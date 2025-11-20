import os.path as osp
from typing import Dict, Iterable, List, Optional, Union
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from transformers import AutoTokenizer

from crslab.config import DATA_PATH, Config
from crslab.data import get_dataset
from crslab.model.crs.hollarec.HypergraphLlava.hypergraph_layers import HGNN

model_path = 'D:\\.Workspace\\.MODEL\\HF-Model-Backup\\llava-1.5-7b-hf'
assert osp.exists(model_path), f"Model path {model_path} does not exist."
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
# tokenizer.add_tokens(['<eot>'], speical_tokens=True)
tokenizer.add_special_tokens({'additional_special_tokens': ['<eot>']})
print(tokenizer)
print(tokenizer.vocab_size)
# print(f"Tokenizer loaded from {model_path} with vocab size {tokenizer.vocab_size}.")
# tokenizer.
# logger.info(f"***** Tokenizer loaded from {model_path} with vocab size {tokenizer.vocab_size}.")

class Stage1Config:
    """Configuration for Stage 1 training"""
    def __init__(self):
        # Hypergraph dimensions
        # super().__init__()
        opt_dict = {
            'dataset': 'ReDial',
            'txt_dim': 1024,
            'img_dim': 512,
            'vdo_dim': 2048,
            'ado_dim': 1024,
            'hg_hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.1,
            'modalities': ['txt', 'img', 'vdo', 'ado'],
            'text_encoder_name': 'llava-hf/llava-1.5-7b-hf',
            'text_max_length': 64,
            'use_half_precision': False,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'lr': 1e-4,
            'num_epochs': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        for k, v in opt_dict.items():
            # self.__setitem__(k, v)
            self.__setattr__(k, v)
        


class Stage1Dataset(Dataset):
    def __init__(self, config, dataset, split: str = 'train'):
        self.config = config
        self.dataset = dataset
        self.modalities = config.modalities
        
        # Get the appropriate split data
        if split == 'train':
            self.data = dataset.train_data
        elif split == 'valid':
            self.data = dataset.valid_data
        elif split == 'test':
            self.data = dataset.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
        self.filter()
        
    def filter(self):
        filtered_data = []
        for sample in self.data:
            if len(sample['context_movies']) != 0:
                filtered_data.append(sample)
        self.data = filtered_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # print(sample['context_movies'])
        data = {} # {str : modality -> m_data_dict}
        for m in self.modalities:
            cur_hg_edge_list = sample['related_movies'][m] # List[List[str]]
            # logger.info(f"structure of cur_hg_edge_list: {cur_hg_edge_list}")
            
            # print(sample['related_movies'][m])
            cur_hg_target_ids = sample['context_movies']
            cur_hg_node_ids: List[str] = []
            cur_hg_edges: List[int] = []
            for edge_id, edge in enumerate(cur_hg_edge_list):
                anchor = cur_hg_target_ids[edge_id]
                node_ids = list(edge) + [anchor]
                cur_hg_node_ids.extend(node_ids) # 
                cur_hg_edges.extend([edge_id] * len(node_ids))
            
            # node_ids_unique = list(dict.fromkeys(cur_hg_node_ids)) # List[str] - preserves order
            # print(cur_hg_edge_list)
            node_ids_unique = list(set(cur_hg_node_ids))
            # logger.info(f"cur_hg_edge_list: {cur_hg_edge_list}")
            # logger.info(f"cur_hg_node_ids: {cur_hg_node_ids}")
            # logger.info(f"node_ids_unique: {node_ids_unique}")
            # assert len(node_ids_unique)!=0, "Duplicate node IDs found in hypergraph construction."
            # if (len(node_ids_unique))
            node_id2ind = {node_id: i for i, node_id in enumerate(node_ids_unique)} 
            node_id2title = {mid: self.dataset.vocab['ind2movie'][mid] for mid in node_ids_unique} # Dict[str -> str]
            if len(node_ids_unique) == 0:
                logger.error(f"cur_hg_target_ids: {cur_hg_target_ids}")
                logger.error(f"cur_hg_edge_list: {cur_hg_edge_list}")
                logger.error(f"cur_hg_node_ids: {cur_hg_node_ids}")
                logger.error(f"node_ids_unique: {node_ids_unique}")
                logger.error(f"Empty hypergraph constructed for modality {m} in sample index {idx}.")
            
            x = torch.stack([self.dataset.get_embedding(mid, modality=m, return_zero_if_missing=True) for mid in node_ids_unique]) # 按照node_ids_unique的顺序堆叠
            cur_hg_node_ind = [node_id2ind[id] for id in cur_hg_node_ids] # 将原始节点ID为索引
            edge_index = torch.tensor([cur_hg_node_ind, cur_hg_edges], dtype=torch.long)
            hg = Data(x=x, hyperedge_index=edge_index)
            data[m] = {
                'hg' : hg,
                'nodes': node_ids_unique, # List[str]
                'id2ind': node_id2ind, # Dict{str -> int}
                'id2title' : node_id2title, # Dict{str -> str}
                'target_ids' : cur_hg_target_ids, # List[str]
                'neighbor_ids': cur_hg_edge_list, # List[List[str]]
            }
        return data

class Stage1Loader:
    def __init__(self, dataset: Stage1Dataset, config: Stage1Config):
        self.dataset = dataset
        self.config = config
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        batched_data = {m: {} for m in self.config.modalities}
        for sample in batch:
            for m in self.config.modalities:
                for key in sample[m]:
                    if key not in batched_data[m]:
                        batched_data[m][key] = []
                    batched_data[m][key].append(sample[m][key])
        return batched_data

from .text_encoder import Transformer

class Stage1CLIP(nn.Module):
    def __init__(self,config, device='cpu'):
        # self.context_length = config.context_length
        super().__init__()
        self.context_length = 64  # Recommended for movie titles context
        self.config = config
        self.device = device
        self.hgnn = nn.ModuleDict()
        for m in self.config.modalities:
            self.hgnn[m] = HGNN(
                in_channels=getattr(config, f"{m}_dim"),
                hidden_channels=config.hg_hidden_size,
                out_channels=config.hg_hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout
            )
        
        self.transformer = Transformer(
            width = config.hg_hidden_size,
            layers = 2,
            heads = 8,
            attn_mask = self.build_attention_mask()
        )

        # HuggingFace keeps `vocab_size` fixed after `add_special_tokens`, so use the
        # full length (base vocab + added tokens) when allocating the embedding.
        self.vocab_size = len(tokenizer)
        self.token_embedding = nn.Embedding(self.vocab_size, config.hg_hidden_size)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, config.hg_hidden_size))
        self.ln_final = nn.LayerNorm(config.hg_hidden_size)
        self.text_projection = nn.Parameter(torch.empty(config.hg_hidden_size, config.hg_hidden_size))

        # self.dtype = s

        self.optim = optim.Adam(
            [
                {'params': self.token_embedding.weight},
                {'params': self.positional_embedding},
                {'params': self.text_projection},
                {'params': self.transformer.parameters()},
                {'params': self.hgnn.parameters()}
            ],
            lr = config.lr
        )

        self.initialize_parameters()
    
    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        proj_std = (self.config.hg_hidden_size ** -0.5) * ((2 * self.config.num_layers) ** -0.5)
        attn_std = self.config.hg_hidden_size ** -0.5
        fc_std = (2 * self.config.hg_hidden_size) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self) -> torch.Tensor:
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    def _encode_texts_flat(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of strings and return normalized embeddings."""
        device = self.positional_embedding.device

        token_ids = tokenize(texts, context_length=self.context_length).to(device)
        try:
            x = self.token_embedding(token_ids) + self.positional_embedding.unsqueeze(0)
        except IndexError:
            logger.error(f"Token IDs exceed vocabulary size. Max token ID: {token_ids.max().item()}, Vocab size: {self.vocab_size}")
            logger.error(f"Texts: {texts}")
            raise IndexError("Token IDs exceed vocabulary size.")
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        eot_positions = (token_ids != 0).sum(dim=1) - 1
        sent_emb = x[torch.arange(x.size(0), device=device), eot_positions, :]

        if self.text_projection is not None:
            sent_emb = sent_emb @ self.text_projection

        return sent_emb

    def _encode_graph_nodes(self, data: Data, modality: str) -> torch.Tensor:
        """Run modality HGNN on a single Data object and return node embeddings.
        Returns tensor [num_nodes, hidden_dim].
        """
        device = self.positional_embedding.device
        # move data tensors to device
        data = Data(x=data.x.to(device), hyperedge_index=data.hyperedge_index.to(device))
        h_out = self.hgnn[modality](data)
        return h_out

    def forward(self, batched_data: Dict[str, Dict[str, List]], temperature: float = 0.07) -> Dict[str, Union[torch.Tensor, Dict[str, float]]]:
        """Compute alignment loss for a batch of modality-specific hypergraphs."""
        device = self.positional_embedding.device
        total_loss: Optional[torch.Tensor] = None
        modality_metrics: Dict[str, float] = {}
        active_modalities = 0

        for modality in self.config.modalities:
            batch_graphs = batched_data[modality]['hg']
            batch_nodes = batched_data[modality]['nodes'] # unique node ids, List[List[str]]
            batch_id2title = batched_data[modality]['id2title']

            modality_loss = 0.0
            modality_samples = 0
            total_loss = 0.0

            for hg, node_ids, id2title in zip(batch_graphs, batch_nodes, batch_id2title): # batch中逐个处理
                if len(node_ids) == 0:
                    continue

                node_titles = [id2title[node_id] for node_id in node_ids]
                text_emb = self._encode_texts_flat(node_titles)
                graph_emb = self._encode_graph_nodes(hg, modality)

                assert graph_emb.size(0) == text_emb.size(0)

                graph_emb = F.normalize(graph_emb, dim=-1) # [N, D]
                text_emb = F.normalize(text_emb, dim=-1) # [N, D]

                logits = torch.matmul(graph_emb, text_emb.transpose(0, 1)) / temperature
                labels = torch.arange(logits.size(0), device=device)
                loss = 0.5 * (
                    F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.transpose(0, 1), labels)
                )

                modality_loss += loss
                modality_samples += 1

            if modality_samples > 0:
                modality_loss = modality_loss / modality_samples
                modality_metrics[modality] = modality_loss.detach().cpu().item()
                total_loss += modality_loss
                active_modalities += 1

        total_loss = total_loss / active_modalities
        return {'loss': total_loss, 'modality_loss': modality_metrics}

def tokenize(texts: Union[str, List[str]], context_length:int = 64, truncate:bool = True):
    if isinstance(texts, str):
        texts = [texts]
    sot_token = tokenizer.convert_tokens_to_ids('<s>')
    # eot_token = tokenizer.convert_tokens_to_ids('</s>')
    eot_token = tokenizer.convert_tokens_to_ids('<eot>')
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Text length {len(tokens)} exceeds context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


class Stage1Trainer:
    def __init__(self, model: Stage1CLIP, train_loader: Stage1Loader, valid_loader: Stage1Loader, config: Stage1Config):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)
        self.model.device = self.device
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def fit(self):
        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.config.num_epochs}")
            train_loss = self.train_one_epoch(epoch)
            logger.info(f"Epoch {epoch+1} train loss: {train_loss:.4f}")
            if self.valid_loader is not None:
                val_loss = self.validate(epoch)
                logger.info(f"Epoch {epoch+1} valid loss: {val_loss:.4f}")

    def train_one_epoch(self, epoch: int):
        self.model.train()
        epoch_loss = 0.0
        data_loader = self.train_loader.dataloader
        batch_i = 0
        for batch_data in tqdm(data_loader, desc=f"Epoch {epoch+1} Training"):
            self.model.optim.zero_grad()
            outputs = self.model(batch_data)
            loss = outputs['loss']
            if torch.isnan(loss):
                logger.warning("Encountered NaN loss; skipping batch")
                continue
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.model.optim.step()
            epoch_loss += loss.item()
            batch_i += 1
            if batch_i % 10 == 0:
                logger.info(f"Batch {batch_i}, Loss: {loss.item():.4f}")
        avg_loss = epoch_loss / max(1, len(data_loader))
        return avg_loss

    def validate(self, epoch: int):
        if self.valid_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        data_loader = self.valid_loader.dataloader
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc=f"Epoch {epoch+1} Validation"):
                outputs = self.model(batch_data)
                total_loss += outputs['loss'].item()

        avg_loss = total_loss / max(1, len(data_loader))
        return avg_loss


def main(config):
    # Example usage
    # config = Stage1Config()
    # redial_dataset = ReDialDataset(data_path='')
    stage1config = Stage1Config()
    redial_dataset = get_dataset(config, 'llava', restore=False, save=False)
    train_dataset = Stage1Dataset(stage1config, redial_dataset, split='train')
    valid_dataset = Stage1Dataset(stage1config, redial_dataset, split='valid')

    train_loader = Stage1Loader(train_dataset, stage1config)
    valid_loader = Stage1Loader(valid_dataset, stage1config)

    model = Stage1CLIP(stage1config, device=stage1config.device)
    trainer = Stage1Trainer(model, train_loader, valid_loader, stage1config)
    trainer.fit()