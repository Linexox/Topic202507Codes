import os
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from crslab.data.dataset.redial.redial import ReDialDataset
from crslab.model.crs.hollarec.HypergraphLlava.hypergraph_layers.hgnn import HGNN
from torch_geometric.data import Data


class Stage1Config:
    def __init__(self):
        # HGNN output hidden size (internal). HGNN may output this dim.
        self.hg_hidden_size = 256
        # projection dimension (text encoder hidden size)
        self.text_hidden_size = 1024  # will be set from text encoder if available
        self.modalities = ["txt", "img", "vdo", "ado"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class Stage1Dataset(Dataset):
    def __init__(self, redial_dataset: ReDialDataset, modalities: List[str],
                 top_k: int = 10, similarity_threshold: float = 0.0):
        self.dataset = redial_dataset
        self.modalities = modalities
        self.top_k = top_k
        self.sim_thr = similarity_threshold

        # movie list (movie ids as strings in dataset.movie2ind values)
        # Use dataset.ind2movie keys/values alignment
        # dataset.movie2ind maps movie_name -> movie_id (string)
        self.movie_ids = list(self.dataset.movie2ind.values())

    def __len__(self):
        return len(self.movie_ids)

    def _build_hgraph_for_movie(self, movie_id: str, modality: str) -> Data:
        # Gather target + top_k similar (by precomputed similarity_matrices)
        sim_dict = getattr(self.dataset, 'similarity_matrices', None)
        if sim_dict is None or modality not in sim_dict:
            node_ids = [movie_id]
            # raise NotImplementedError("No similarity matrix found for modality "
        else:
            cur_list = sim_dict[modality].get(movie_id, [])
            selected = [sid for sid, score in cur_list if score >= self.sim_thr][:self.top_k]
            node_ids = [movie_id] + selected

        # node features
        features = []
        for nid in node_ids:
            emb = self.dataset.get_embedding(nid, modality, return_zero_if_missing=True)
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float)
            features.append(emb.unsqueeze(0))
        x = torch.cat(features, dim=0)  # [num_nodes, feat_dim]

        # hyperedge_index: connect all nodes to a single hyperedge idx 0
        node_idx = torch.arange(x.size(0), dtype=torch.long)
        hyperedge_idx = torch.zeros(x.size(0), dtype=torch.long)  # single hyperedge
        hyperedge_index = torch.stack([node_idx, hyperedge_idx], dim=0)

        data = Data(x=x, hyperedge_index=hyperedge_index)
        # store mapping (target index 0)
        data.target_idx = 0
        data.node_movie_ids = node_ids
        return data

    def __getitem__(self, idx: int):
        movie_id = self.movie_ids[idx]
        movie_name = self.dataset.ind2movie.get(str(movie_id), str(movie_id))

        out = {
            'movie_id': movie_id,
            'movie_name': movie_name,
        }
        # build a hypergraph Data per modality
        for m in self.modalities:
            data_m = self._build_hgraph_for_movie(movie_id, m)
            out[f"{m}_hgraph_data"] = data_m
        return out


class TextEncoder:
    """Frozen text encoder wrapper. Use a Transformer model and mean-pooling.

    We freeze parameters and return a vector per string.
    """

    def __init__(self, model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # try detect hidden size
        try:
            self.hidden_size = self.model.config.hidden_size
        except Exception:
            # fallback
            self.hidden_size = 768

    def encode(self, texts: List[str]) -> torch.Tensor:
        # return [len(texts), hidden_size]
        with torch.no_grad():
            toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks, return_dict=True)
            last = out.last_hidden_state  # [B, T, D]
            attn_mask = toks['attention_mask'].unsqueeze(-1)
            summed = (last * attn_mask).sum(1)
            denom = attn_mask.sum(1).clamp(min=1e-9)
            emb = summed / denom
            return emb


class Stage1Model(nn.Module):
    """Wraps 4 HGNN towers + input adapters + projection to text space.

    Workflow:
      - For each modality: adapter (linear) maps input embedding dim -> hg_hidden_size
      - HGNN processes modality Data -> node embeddings (num_nodes, hg_hidden_size)
      - Take target node embedding (index 0)
      - Project target embedding -> text_hidden_size (linear)

    Only the HGNN towers and projection layers are trainable in Stage1.
    Text encoder is frozen and used to provide targets.
    """

    def __init__(self, config: Stage1Config, input_dims: Dict[str, int]):
        super().__init__()
        self.cfg = config
        self.modalities = config.modalities

        # adapters map raw embedding dim -> hg_hidden_size
        self.adapters = nn.ModuleDict()
        for m in self.modalities:
            in_dim = input_dims.get(m, None)
            if in_dim is None:
                raise ValueError(f"Missing input dim for modality {m}")
            self.adapters[m] = nn.Linear(in_dim, config.hg_hidden_size)

        # HGNN towers
        self.hgnns = nn.ModuleDict()
        for m in self.modalities:
            self.hgnns[m] = HGNN(
                in_channels=config.hg_hidden_size,
                hidden_channels=config.hg_hidden_size * 2,
                out_channels=config.hg_hidden_size,
                num_layers=2,
                dropout=0.1,
            )

        # final projectors to text space
        self.projectors = nn.ModuleDict()
        for m in self.modalities:
            self.projectors[m] = nn.Linear(config.hg_hidden_size, config.text_hidden_size)

    def forward_single(self, sample: Dict):
        """Process one sample (conversation turn) and return per-modality projected vectors.
        
        sample contains keys like 'txt_hgraph_data' -> Data
        For each modality, we:
          1. Map node features through adapter
          2. Run HGNN to get node embeddings
          3. Aggregate target node embeddings (context_movies)
          4. Project to text space
        """
        out = {}
        for m in self.modalities:
            data: Data = sample.get(f"{m}_hgraph_data", None)
            if data is None or len(getattr(data, 'target_indices', [])) == 0:
                out[m] = None
                continue
            
            # adapter: map node features
            x = data.x.to(self.adapters[m].weight.device)
            x_mapped = self.adapters[m](x)  # [num_nodes, hg_hidden]
            
            # create a new Data with mapped features
            d2 = Data(x=x_mapped, hyperedge_index=data.hyperedge_index.to(x.device))
            emb_nodes = self.hgnns[m](d2)  # [num_nodes, hg_hidden]
            
            # aggregate target nodes (context_movies)
            target_indices = data.target_indices
            if isinstance(target_indices, list):
                target_indices = torch.tensor(target_indices, dtype=torch.long)
            target_embs = emb_nodes[target_indices]  # [num_context_movies, hg_hidden]
            
            # Average pool context movies
            aggregated_emb = target_embs.mean(dim=0)  # [hg_hidden]
            
            # Project to text space
            proj = self.projectors[m](aggregated_emb)  # [text_hidden]
            out[m] = proj
        return out

    def forward_batch(self, batch_samples: List[Dict]):
        # process sequentially (simple and memory-friendly)
        results = {m: [] for m in self.modalities}
        for sample in batch_samples:
            single = self.forward_single(sample)
            for m in self.modalities:
                results[m].append(single[m])
        # stack per modality (if None, skip)
        stacked = {}
        for m in self.modalities:
            if results[m] and results[m][0] is not None:
                stacked[m] = torch.stack(results[m], dim=0)
            else:
                stacked[m] = None
        return stacked


def info_nce_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07):
    # a: [B, D], b: [B, D]
    assert a is not None and b is not None
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = (a @ b.t()) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)


def train_stage1(args):
    device = args.device

    # 1) load dataset
    opt = {
        'load_saved_embeddings': True,
        # leave other options default in ReDialDataset; the caller may customize
    }
    print("Loading ReDialDataset (this may take some seconds)...")
    redial = ReDialDataset(opt=opt, tokenize='llava', restore=False, save=False)

    # infer input dims from dataset
    input_dims = {}
    for m in ["txt", "img", "vdo", "ado"]:
        dim_attr = f"{m}_dim"
        # dataset may have attributes txt_dim etc; fallback to common values
        val = getattr(redial, f"{m}_dim", None)
        if val is None:
            # try to peek at embeddings
            try:
                emb = redial.get_embedding(list(redial.movie2ind.values())[0], m)
                val = emb.numel()
            except Exception:
                val = 768
        input_dims[m] = int(val)
    print("Detected input dims:", input_dims)

    # 2) build Stage1 dataset + dataloader
    dataset = Stage1Dataset(redial, modalities=["txt", "img", "vdo", "ado"], split='train')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=lambda x: x)

    # 3) text encoder
    text_encoder = TextEncoder(args.text_encoder, device=device)

    # set actual text_hidden_size in config
    cfg = Stage1Config()
    cfg.text_hidden_size = text_encoder.hidden_size
    cfg.device = device

    # 4) model
    model = Stage1Model(cfg, input_dims)
    model = model.to(device)

    # 5) optim (train HGNN adapters, HGNNs and projectors)
    # collect parameters
    trainable_params = []
    for name, p in model.named_parameters():
        # by default all parameters in Stage1Model are trainable
        trainable_params.append(p)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)

    # training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch_samples in pbar:
            # batch_samples is a list of sample dicts (size=batch_size)
            # Each sample has 'movie_names' which is a list of movie names
            # Flatten all movie names and encode them
            all_movie_names = []
            sample_movie_counts = []  # track how many movies per sample
            for s in batch_samples:
                names = s['movie_names']
                all_movie_names.extend(names)
                sample_movie_counts.append(len(names))
            
            if len(all_movie_names) == 0:
                continue
            
            # Encode all movie names at once
            all_text_embs = text_encoder.encode(all_movie_names).to(device)  # [total_movies, D]
            
            # Split back into per-sample embeddings and average
            text_embs_per_sample = []
            start_idx = 0
            for count in sample_movie_counts:
                if count == 0:
                    # fallback for empty samples
                    text_embs_per_sample.append(torch.zeros(all_text_embs.size(1), device=device))
                else:
                    sample_embs = all_text_embs[start_idx:start_idx + count]
                    avg_emb = sample_embs.mean(dim=0)  # average over context movies
                    text_embs_per_sample.append(avg_emb)
                    start_idx += count
            
            text_embs = torch.stack(text_embs_per_sample, dim=0)  # [B, D]
            # batch_samples is a list of sample dicts (size=batch_size)
            # Each sample has 'movie_names' which is a list of movie names
            # Flatten all movie names and encode them
            all_movie_names = []
            sample_movie_counts = []  # track how many movies per sample
            for s in batch_samples:
                names = s['movie_names']
                all_movie_names.extend(names)
                sample_movie_counts.append(len(names))
            
            if len(all_movie_names) == 0:
                continue
            
            # Encode all movie names at once
            all_text_embs = text_encoder.encode(all_movie_names).to(device)  # [total_movies, D]
            
            # Split back into per-sample embeddings and average
            text_embs_per_sample = []
            start_idx = 0
            for count in sample_movie_counts:
                if count == 0:
                    # fallback for empty samples
                    text_embs_per_sample.append(torch.zeros(all_text_embs.size(1), device=device))
                else:
                    sample_embs = all_text_embs[start_idx:start_idx + count]
                    avg_emb = sample_embs.mean(dim=0)  # average over context movies
                    text_embs_per_sample.append(avg_emb)
                    start_idx += count
            
            text_embs = torch.stack(text_embs_per_sample, dim=0)  # [B, D]

            # forward through model (sequential per-sample inside)
            stacked = model.forward_batch(batch_samples)

            # compute losses per modality
            loss = 0.0
            count = 0
            for m in cfg.modalities:
                m_proj = stacked.get(m, None)
                if m_proj is None:
                    continue
                l = info_nce_loss(m_proj, text_embs, temperature=args.temperature)
                loss = loss + l
                count += 1

            if count == 0:
                continue
            loss = loss / count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

        print(f"Epoch {epoch} avg loss: {total_loss / len(loader):.4f}")

        # save checkpoints
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"stage1_epoch{epoch}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dims': input_dims,
            'text_encoder': args.text_encoder,
            'cfg': cfg.__dict__,
        }, ckpt_path)
        print("Saved", ckpt_path)

    print("Stage1 training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-encoder', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='text encoder model name or path')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--sim-thr', type=float, default=0.0)
    parser.add_argument('--save-dir', type=str, default='./stage1_ckpt')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))

    args = parser.parse_args()
    train_stage1(args)
