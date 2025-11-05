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
from .HypergraphLlava import HypergraphLlavaConfig, HypergraphLlavaModel, HypergraphLlavaForCausalLM
# from crslab.model.crs.hollarec.hypergraph_llava import HypergraphLlavaCRS
# from crslab.model.utils.modules.attention import


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


    def encode_user(self, context_tokens, context_movies, related_movies):
        pass

    def recommend(self, batch, mode):
        context_movies = batch["context_movies"]  
        context_tokens = batch["context_tokens"]
        related_movies = batch["related_movies"]
        
        movie_embedding = self.movie_embedding.weight  # (n_movies, movie_emb_dim)
        
        user_embedding = self.encode_user(
            context_tokens,
            context_movies,
            related_movies
        )
        scores = F.linear(user_embedding, movie_embedding)
        pass

    def converse(self, batch, mode):
        pass

    def forward(self, batch, mode, stage):
        if stage == "conv":
            return self.converse(batch, mode)
        elif stage == "rec":
            return self.recommend(batch, mode)

    
        