import os.path as osp
from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from transformers import AutoTokenizer

from crslab.data import ReDialDataset
from crslab.model.crs.hollarec.HypergraphLlava.hypergraph_layers import HGNN

model_path = 'D:\\.Workspace\\.MODEL\\HF-Model-Backup\\llava-1.5-7b-hf'
assert osp.exists(model_path), f"Model path {model_path} does not exist."
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

class Stage1Config:
    """Configuration for Stage 1 training"""
    def __init__(self):
        # Hypergraph dimensions
        self.txt_dim = 768
        self.img_dim = 512
        self.vdo_dim = 512
        self.ado_dim = 128
        
        # HGNN architecture
        self.hg_hidden_size = 256
        self.num_layers = 2
        self.dropout = 0.1
        
        # Modalities
        self.modalities = ['txt', 'img', 'vdo', 'ado']
        # Text encoder
        self.text_encoder_name: str = 'llava-hf/llava-1.5-7b-hf'
        self.text_max_length: int = 64
        self.use_half_precision: bool = False
        
        # Training
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Stage1Dataset(Dataset):
    def __init__(self, redial_dataset: ReDialDataset, split: str = 'train'):
        self.redial_dataset = redial_dataset
        self.modalities = ['txt', 'img', 'vdo', 'ado']
        
        # Get the appropriate split data
        if split == 'train':
            self.data = redial_dataset.train_data
        elif split == 'valid':
            self.data = redial_dataset.valid_data
        elif split == 'test':
            self.data = redial_dataset.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        data = {} # {str : modality -> m_data_dict}
        for m in ['txt', 'img', 'vdo', 'ado']:
            cur_hg_edge_list = sample['related_movies'][m]
            cur_hg_target_ids = sample['context_movies']
            cur_hg_node_ids: List[str] = []
            cur_hg_edges: List[int] = []
            for edge_id, edge in enumerate(cur_hg_edge_list):
                anchor = cur_hg_target_ids[edge_id]
                node_ids = list(edge) + [anchor]
                cur_hg_node_ids.extend(node_ids)
                cur_hg_edges.extend([edge_id] * len(node_ids))
            
            node_ids_unique = list(dict.fromkeys(cur_hg_node_ids))
            node_id2ind = {node_id: i for i, node_id in enumerate(node_ids_unique)}
            node_titles = [self.redial_dataset.vocab['ind2movie'][mid] for mid in node_ids_unique]
            x = torch.stack([self.redial_dataset.get_embedding(mid, modality=m) for mid in node_ids_unique])
            cur_hg_node_ind = [node_id2ind[id] for id in cur_hg_node_ids]
            # cur_hg_target_ind = [node_id2ind[id] for id in cur_hg_target_id]
            edge_index = torch.tensor([cur_hg_node_ind, cur_hg_edges], dtype=torch.long)
            hg = Data(x=x, hyperedge_index=edge_index)
            data[m] = {
                'hg' : hg,
                'id2ind': node_id2ind, # [str -> int]
                'node_ids': node_ids_unique, # [int]
                'node_titles': node_titles, # [str] mapped by node_ids
                'target_ids' : cur_hg_target_ids # [str]
            }
        return data


        # sample = self.data[idx]
        # data = {}
        # for m in ['txt', 'img', 'vdo', 'ado']:
        #     cur_hg_edge_list = sample['related_movies'][m]
        #     cur_hg_nodes: List[str] = []
        #     cur_hg_edges: List[int] = []
        #     for edge_id, edge in enumerate(cur_hg_edge_list):
        #         anchor = sample['context_movies'][edge_id]
        #         nodes_in_edge = list(edge) + [anchor]
        #         cur_hg_nodes.extend(nodes_in_edge)
        #         cur_hg_edges.extend([edge_id] * len(nodes_in_edge))

        #     node_ids_unique = list(dict.fromkeys(cur_hg_nodes))
        #     node_to_idx = {node: i for i, node in enumerate(node_ids_unique)}
        #     cur_hg_nodes_indices = [node_to_idx[mid] for mid in cur_hg_nodes]
        #     node_ids = node_ids_unique
        #     node_titles = [self.redial_dataset.vocab['ind2movie'][mid] for mid in node_ids]
        #     x = torch.stack([
        #         self.redial_dataset.get_embedding(mid, modality=m)
        #         for mid in node_ids
        #     ])
        #     edge_index = torch.tensor([cur_hg_nodes_indices, cur_hg_edges], dtype=torch.long)
        #     hg = Data(x=x, hyperedge_index=edge_index)
        #     data[m] = {
        #         'hg' : hg,
        #         'movie_id2map': node_to_idx, # [str -> int
        #         'movie_ids': node_ids, # [int]
        #         'movie_names': node_titles # mapped by node_ids
        #     }
        # return data

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
        batched_data = {m: {} for m in ['txt', 'img', 'vdo', 'ado']}
        for sample in batch:
            for m in self.config.modalities:
                for key in sample[m]:
                    if key not in batched_data[m]:
                        batched_data[m][key] = []
                    batched_data[m][key].append(sample[m][key])
        return batched_data

from text_encoder import Transformer

class Stage1Trainer:
    def __init__(self,config):
        # self.context_length = config.context_length
        self.context_length = 64  # Recommended for movie titles context
        self.config = config
        self.hgnn = nn.ModuleDict()
        for m in ['txt', 'img', 'vdo', 'ado']:
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

        self.vocab_size = tokenizer.vocab_size
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
        nn.init.normal_(self.token_embedding, std=0.02)
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

    # def encode_graph(self, hg: Data, m:str):
    #     # Encode the hypergraph using the corresponding HGNN
    #     # hgnn = self.hgnn[modality]
    #     raise NotImplementedError("Use Stage1Trainer.forward() to encode graphs per-modality")

    def _encode_texts_flat(self, flat_texts: List[str]) -> torch.Tensor:
        """Encode a flat list of strings. Returns tensor (N, D).
        Uses </s> (EOT) token position for pooling.
        """
        if len(flat_texts) == 0:
            return torch.empty((0, self.config.hg_hidden_size), device=self.positional_embedding.device)

        token_ids = tokenize(flat_texts, context_length=self.context_length).to(self.positional_embedding.device)
        x = self.token_embedding(token_ids) + self.positional_embedding.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        # EOT is at last non-zero token position (tokenize adds </s>)
        eot_positions = (token_ids != 0).sum(dim=1) - 1
        sent_emb = x[torch.arange(x.size(0), device=x.device), eot_positions, :]

        if hasattr(self, 'text_projection') and self.text_projection is not None:
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

    def forward(self, batched_data: Dict, temperature: float = 0.07):
        """Compute per-modality node-text alignment losses for a batch.

        batched_data: produced by Stage1Loader.collate_fn
        Returns: dict with total loss and per-modality stats
        """
        device = self.positional_embedding.device
        batch_size = len(batched_data['txt']['hg'])

        total_alignment = 0.0
        count_align = 0

        for m in ['txt', 'img', 'vdo', 'ado']:
            for i in range(batch_size):
                data_i: Data = batched_data[m]['hg'][i]
                node_titles: List[str] = batched_data[m]['node_titles'][i]
                
                if len(node_titles) == 0:
                    continue

                node_text_emb = self._encode_texts_flat(node_titles)
                node_graph_emb = self._encode_graph_nodes(data_i, m)
                assert node_graph_emb.size(0) == node_text_emb.size(0), "Dismatched node counts between graph and text encoders."

                g = F.normalize(node_graph_emb, dim=-1)
                t = F.normalize(node_text_emb.to(g.device), dim=-1)

                logits = torch.matmul(g, t.transpose(0, 1)) / temperature
                labels = torch.arange(logits.size(0), device=logits.device)
                loss_a = F.cross_entropy(logits, labels)
                loss_b = F.cross_entropy(logits.transpose(0, 1), labels)
                total_alignment += 0.5 * (loss_a + loss_b)
                count_align += 1

        if count_align == 0:
            return {'loss': torch.tensor(0.0, device=device), 'alignment_loss': 0.0}

        alignment_loss = total_alignment / count_align
        stats = {'loss': alignment_loss, 'alignment_loss': alignment_loss.detach().cpu().item()}
        return stats
        return node_embeddings

    # def encode_text(self, text):
    #     x = self.token_embedding(text)
    #     x = x + self.positional_embedding
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final(x)

    #     # take features from the eot token
    #     x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    #     x = x @ self.text_projection
    #     return x
        
    # def forward(self, batch_data):
    #     mm_hg_embeddings = {}
    #     for m in ['txt', 'img', 'vdo', 'ado']:
    #         m_hg = batch_data[m]['hg']
    #         mm_hg_embeddings[m] = self.encode_graph(m_hg, m)
    #     return mm_hg_embeddings


def tokenize(texts: Union[str, List[str]], context_length:int = 64, truncate:bool = True):
    if isinstance(texts, str):
        texts = [texts]
    sot_token = tokenizer.convert_tokens_to_ids('<s>')
    eot_token = tokenizer.convert_tokens_to_ids('</s>')
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