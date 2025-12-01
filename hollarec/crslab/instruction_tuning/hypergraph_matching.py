from crslab.data import ReDialDataset
from crslab.model.crs.hollarec.HypergraphLlava import MMHypergraphLlavaModel
from .conversation import default_conversation

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from typing import List, Dict
import dataclasses

class Task_1_Dataset(Dataset):
    def __init__(self, dataset: ReDialDataset, split):
        self.dataset = dataset
        self.split = split
        self.data = getattr(self.dataset, f"{split}_data")