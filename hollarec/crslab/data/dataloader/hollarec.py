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
        self.ind2tok = vocab["ind2tok"]
        self.tok2ind = vocab["tok2ind"]
        self.n_movies = vocab["n_movies"]
        self.pad_token_idx = vocab["tok2ind"]["<pad>"]
        self.start_token_idx = vocab["tok2ind"]["<s>"]
        self.end_token_idx = vocab["tok2ind"]["</s>"]
        self.image_token_idx = vocab["tok2ind"]["<image>"]
        self.hgraph_token_idx = vocab["tok2ind"]["<hgraph>"]
        self.hg_patch_token_idx = vocab["tok2ind"]["<hg_patch>"]
        self.hg_start_token_idx = vocab["tok2ind"]["<hg_start>"]
        self.hg_end_token_idx = vocab["tok2ind"]["<hg_end>"]

        def rec_process_fn(self):
            augment_dataset = []
            for conv in tqdm(self.dataset, desc="[Dataloader process]"):
                if conv["role"] == "Recommender":
                    for movie in conv["movies"]:
                        augment_conv = {
                            "context_tokens": copy(conv["context_tokens"]),
                            "context_movies": copy(conv["context_movies"]),
                            "response": copy(conv["response"]),
                            "movie": movie,
                            "role": conv["role"],
                            # "conv_id":
                        }
                        augment_dataset.append(augment_conv)

            return augment_dataset

        def rec_batchify(self, batch):
            batch_context_tokens = []
            batch_context_movies = []
            batch_movie = []

            for conv in batch:
                batch_context_tokens.append(conv["context_tokens"])
                batch_context_movies.append(conv["context_movies"])
                batch_movie.append(conv["movie"])

            res = {
                "context_tokens": torch.LongTensor(batch_context_tokens).to(self.opt["device"]),
                "context_movies": torch.LongTensor(batch_context_movies).to(self.opt["device"]),
                "movie": torch.LongTensor(batch_movie).to(self.opt["device"]),
            }

            return res

        def conv_process_fn(self):
            dataset = []
            for conv in tqdm(self.dataset, desc="[Dataloader process]"):
                if conv["role"] == "Recommender":
                    dataset.append(conv)
            return dataset

        def conv_batchify(self, batch):
            batch_context_tokens = []
            batch_context_movies = []
            batch_response = []

            for conv in batch:
                batch_context_tokens.append(conv["context_tokens"])
                batch_context_movies.append(conv["context_movies"])
                batch_response.append(conv["response"])

            res = {
                "context_tokens": torch.LongTensor(batch_context_tokens).to(self.opt["device"]),
                "context_movies": torch.LongTensor(batch_context_movies).to(self.opt["device"]),
                "response": torch.LongTensor(batch_response).to(self.opt["device"]),
            }

            return res
