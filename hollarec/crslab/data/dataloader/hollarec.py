from copy import copy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader

class HollaRecDataLoader(BaseDataLoader):
    def __init__(self, opt, dataset, vocab):
        """
        
        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.ind2tok = vocab['ind2tok']
        self.tok2ind = vocab['tok2ind']
        self.n_movies = vocab['n_movies']
        self.pad_token_idx = vocab['tok2ind']['<pad>']
        self.start_token_idx = vocab['tok2ind']['<s>']
        self.end_token_idx = vocab['tok2ind']['</s>']
        self.image_token_idx = vocab['tok2ind']['<image>']
        self.hgraph_token_idx = vocab['tok2ind']['<hgraph>']
        self.hg_patch_token_idx = vocab['tok2ind']['<hg_patch>']
        self.hg_start_token_idx = vocab['tok2ind']['<hg_start>']
        self.hg_end_token_idx = vocab['tok2ind']['<hg_end>']

        def rec_process_fn(self):
            # raise NotImplementedError("Data processing function is not implemented yet.")
            augment_dataset = []
            for data in tqdm(self.dataset, desc='[Dataloader process]'):
                augment_data = copy(data)
                # Add any necessary processing steps here
                augment_dataset.append(augment_data)

            return augment_dataset
        
        def rec_batchify(self, batch):
            raise NotImplementedError("Batchify function is not implemented yet.")
        
        def conv_process_fn(self):
            raise NotImplementedError("Conversation data processing function is not implemented yet.")
        
        def conv_batchify(self, batch):
            raise NotImplementedError("Conversation batchify function is not implemented yet.")
