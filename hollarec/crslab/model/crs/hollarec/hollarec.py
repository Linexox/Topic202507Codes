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
from crslab.model.crs.hollarec.hypergraph_llava import HypergraphLlavaCRS
# from crslab.model.utils.modules.attention import


class HollaRec(BaseModel):
    def __init__(self, opt, device, vocab, side_data=None):
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

        super(HollaRec, self).__init__(opt, device)
        

        self.side_data = side_data
    
    def build_model(self, *args, **kwargs):
        