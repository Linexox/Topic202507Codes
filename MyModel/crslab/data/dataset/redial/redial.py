import json
import os 
from copy import copy

from loguru import logger
import torch
from tqdm import tqdm

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from .resources import resources

class ReDialDataset(BaseDataset):
    def __init__(self, opt, tokenize, restore=False, save=False):
        resource = resources[tokenize]
        dpath = os.path.join(DATASET_PATH, 'redial', tokenize)
        super().__init__(opt, dpath, resource, restore, save)

    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        self.train_review, self.valid_review, self.test_review = self._load_raw_review()
        self._load_vocab()
        self._load_other_data()