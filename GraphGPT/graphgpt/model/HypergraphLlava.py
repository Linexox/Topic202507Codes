from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, LlavaConfig, LlavaModel

from transformers.modeling_outputs import BaseModelOutputWithPast

from torch_geometric.data import Data
import json
import os.path as osp

DEFAULT_HYPERGRAPH_TOKEN = "<hgraph>"
DEFAULT_HYPERGRAPH_PATCH_TOKEN = "<hg_path>"
DEFAULT_HG_START_TOKEN = "<hg_start>"
DEFAULT_HG_END_TOKEN = "<hg_end>"

class HypergraphLlavaConfig(LlavaConfig):
    model_type = "HypergraphLlava"

class HypergraphPretrainConfig(AutoConfig):
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

class HypergraphLlavaModel(LlavaModel):
    config_class = HypergraphLlavaConfig

    def __init__(self, config: HypergraphLlavaConfig):
        super().__init__(config)
        if config.graph_tower == 'HGNN'
