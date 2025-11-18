import torch
import torch.nn as nn
import torch.nn.functional as F
from text_graph_grounding.text_encoder import Transformer
from transformers import AutoTokenizer
import numpy as np

from crslab.model.crs.hollarec.HypergraphLlava.hypergraph_layers import HGNN

model_path = 'D:\\.Workspace\\.MODEL\\HF-Model-Backup\\llava-1.5-7b-hf'

def cal_cl_loss(s_features, t_features, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * s_features @ t_features.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    ret_loss = (loss_i + loss_t) / 2
    return ret_loss


class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()