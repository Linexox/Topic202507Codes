import json
import os.path as osp
import random

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import RGCNConv, HypergraphConv

from crslab.config import DATASET_PATH
from crslab.model.base import BaseModel
from .HypergraphLlava import HypergraphLlavaConfig, HypergraphLlavaModel, HypergraphLlava4Recsys
# from crslab.model.crs.hollarec.hypergraph_llava import HypergraphLlavaCRS
# from crslab.model.utils.modules.attention import

HYPERGRAPH_TRUNCATE_LEN = 50

class HollaRec(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
        """
        Args:
            opt (Config or dict): config for model or the whole system.
            device (torch.device): device for model running.
            vocab (dict): all kinds of useful size, idx and map between token and idx.
            side_data (dict, optional): side data for model. Defaults to None.
        """
        self.device = device
        self.gpu = opt.get("gpu", -1)
        self.dataset = opt.get("dataset", None)
        assert self.dataset in ['redial']

        self.vocab = vocab
        self.n_movies = vocab["n_movies"]
        self.pad_token_idx = vocab["tok2ind"]["<pad>"]
        self.start_token_idx = vocab["tok2ind"]["<s>"]
        self.end_token_idx = vocab["tok2ind"]["</s>"]
        self.image_token_idx = vocab["tok2ind"]["<image>"]
        self.hgraph_token_idx = vocab["tok2ind"]["<hgraph>"]
        self.hg_patch_token_idx = vocab["tok2ind"]["<hg_patch>"]
        self.hg_start_token_idx = vocab["tok2ind"]["<hg_start>"]
        self.hg_end_token_idx = vocab["tok2ind"]["<hg_end>"]
        self.split_token_idx = vocab["tok2ind"].get("<split>", None)
        self.vocab_size = vocab['vocab_size']
        self.token_emb_dim = opt.get("token_emb_dim", 4096)
        self.pretrain_embedding = side_data.get("embedding", None)

        self.movie_emb_dim = opt.get("movie_emb_dim", 256)
        self.user_emb_dim = self.movie_emb_dim
        super(HollaRec, self).__init__(opt, device)
        

        self.side_data = side_data
    
    def build_model(self, *args, **kwargs):
        
        pass

    def build_embedding(self):
        if self.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float),
                freeze=False,
                padding_idx=self.pad_token_idx,
            )
        else:
            self.token_embedding = nn.Embedding(
                self.vocab_size, self.token_emb_dim, padding_idx=self.pad_token_idx
            )
            nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)
        self.movie_embedding = nn.Embedding(self.n_movies, self.token_emb_dim, 0)
        nn.normal_(self.movie_embedding.weight, mean=0, std=self.movie_emb_dim ** -0.5)


    # def encode_user(self, context_tokens, context_movies, related_movies):
        
    #     pass

    def _get_modality_sim_hypergraph(self, session_context_movies, session_related_movies):
        hypergraph_nodes = []
        hypergraph_edges = []
        hypergraph_edge_counter = 0

        hypergraph_truncate_len = HYPERGRAPH_TRUNCATE_LEN
        # 截断与session_context_movies最相似的session_related_movies
        scores = []
        for related_movies in session_related_movies:
            for _, score in related_movies:
                scores.append(score)
        if len(scores) > hypergraph_truncate_len:
            threshold = sorted(scores, reverse=True)[hypergraph_truncate_len - 1]
        
        for related_movies in session_related_movies:
            filtered_movies = [
                (movie_id, score)
                for movie_id, score in related_movies
                if score >= threshold
            ]
            # hypergraph_nodes.extend([movie_id for movie_id, _ in filtered_movies])
            hypergraph_nodes += [movie_id for movie_id, _ in filtered_movies]
            hypergraph_edges += [hypergraph_edge_counter] * len(filtered_movies)
            hypergraph_edge_counter += 1
        hyper_edge_index = torch.tensor([hypergraph_nodes, hypergraph_edges], dtype=torch.long, device=self.device)
        return list(set(hypergraph_nodes)), hyper_edge_index

    def recommend(self, batch, mode):
        context_movies = batch["context_movies"]    # shape: (batch_size, n_context_movies)
        context_tokens = batch["context_tokens"]    # shape: (batch_size, seq_len)
        related_movies = batch["related_movies"]    # shape: (batch_size, n_related_movies)
        
        movie_embedding = self.movie_embedding.weight  # (n_movies, movie_emb_dim)
        
        
        # user_embedding = self.encode_user(
        #     context_tokens,
        #     context_movies,
        #     related_movies
        # )
        # scores = F.linear(user_embedding, movie_embedding)
        pass

    def converse(self, batch, mode):
        pass

    def forward(self, batch, mode, stage):
        if stage == "conv":
            return self.converse(batch, mode)
        elif stage == "rec":
            return self.recommend(batch, mode)