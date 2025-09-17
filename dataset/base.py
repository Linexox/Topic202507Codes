import os
import json
import random
from abc import ABC, abstractmethod


class BaseDataset:
    def __init__(self, config=None, dataset_path=None):
        self.config = config
        self.dataset_path = dataset_path

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess_data(self):
        pass
