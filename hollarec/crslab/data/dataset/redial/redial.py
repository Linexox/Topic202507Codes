import os
import json
import pickle
import pandas as pd
from copy import copy
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer
from loguru import logger
from tqdm import tqdm

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from .resources import resources


class ReDialDataset(BaseDataset):
    def __init__(
        self,
        opt,
        tokenize: str = "llava",
        restore: bool = False,
        save: bool = True,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        resource = resources.get(tokenize, None)
        if resource is None:
            logger.info(f"[dataset.redial] Resource for tokenizer '{tokenize}' not found.")
        else:
            logger.info(f"[dataset.redial] Using resource for tokenizer '{tokenize}'.")

        self.opt = opt
        self.dpath = os.path.join(DATASET_PATH, "redial", tokenizer)
        super().__init__(opt, tokenize, restore, save)
        if tokenizer is not None:
            self.tokenizer = tokenizer
            logger.info("[dataset.redial] Using provided tokenizer with special tokens.")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.dpath, "tokenizer")
            )
        self._verify_tokenizer_special_tokens()
            

    def _verify_tokenizer_special_tokens(self):
        DEFAULT_HYPERGRAPH_TOKEN = "<hgraph>"
        DEFAULT_HYPERGRAPH_PATCH_TOKEN = "<hg_patch>"
        DEFAULT_HG_START_TOKEN = "<hg_start>"
        DEFAULT_HG_END_TOKEN = "<hg_end>"
        required_tokens = [
            DEFAULT_HYPERGRAPH_TOKEN,
            DEFAULT_HYPERGRAPH_PATCH_TOKEN,
            DEFAULT_HG_START_TOKEN,
            DEFAULT_HG_END_TOKEN,
        ]
        for token in required_tokens:
            if token not in self.tokenizer.get_vocab():
                raise ValueError(f"Tokenizer is missing required special token: {token}")
        logger.info("[dataset.redial] All required special tokens are present in the tokenizer.")
        self.hg_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_HYPERGRAPH_TOKEN)
        self.hg_start_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_HG_START_TOKEN)
        self.hg_end_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_HG_END_TOKEN)
        self.hg_patch_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_HYPERGRAPH_PATCH_TOKEN)

    def _load_data(self):
        # raise NotImplementedError("Use _load_raw_data instead for ReDialDataset.")
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vacab()
        self._load_other_data()

        vocab = {
            'ind2tok': self.ind2tok,
            'tok2ind': self.tok2ind,
            'ind2movie': self.ind2movie,
            'movie2ind': self.movie2ind,
            'vocab_size': len(self.tok2ind),
            'n_movie': len(self.movie2ind),
        }
        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        if not os.path.exists(self.dpath):
            raise FileNotFoundError(f"Dataset path {self.dpath} does not exist.")

        with open(os.path.join(self.dpath, "train_data.json"), "rb") as f:
            train_data = json.load(f)
            logger.info(f"[dataset.redial] Loaded {len(train_data)} training samples.")
        with open(os.path.join(self.dpath, "valid_data.json"), "rb") as f:
            valid_data = json.load(f)
            logger.info(f"[dataset.redial] Loaded {len(valid_data)} validation samples.")
        with open(os.path.join(self.dpath, "test_data.json"), "rb") as f:
            test_data = json.load(f)
            logger.info(f"[dataset.redial] Loaded {len(test_data)} test samples.")

        return train_data, valid_data, test_data

    def _load_vacab(self):
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            raise ValueError("Tokenizer must be set before loading vocabulary.")
        # self.tok2ind = {token: idx for token, idx in self.tokenizer.get_vocab().items()}
        # self.ind2tok = {idx: token for token, idx in self.tokenizer.get_vocab().items()}
        with open(os.path.join(self.dpath, 'token2ind.json'), 'rb') as f:
            self.token2ind = json.load(f)
            self.ind2tok = {idx: token for token, idx in self.token2ind.items()}
            logger.info("[dataset.redial] Vocabulary loaded from tokenizer.")
            logger.info(f"[dataset.redial] Vocabulary size: {len(self.token2ind)} tokens.")

    def _load_other_data(self):
        if not os.path.exists(self.dpath):
            raise FileNotFoundError(f"Dataset path {self.dpath} does not exist.")
        with open(os.path.join(self.dpath, "movie2ind.json"), "rb") as f:
            self.movie2ind = json.load(f)
            self.ind2movie = {int(movie_id): movie_name for movie_name, movie_id in self.movie2ind.items()}
            logger.info(f"[dataset.redial] Loaded movie vocabulary: {len(self.movie2ind)} movies.")
        
        # with open(os.path.join(self.dpath, "movie_mentioned.csv"), "rb") as f:
        #     movies = pd.read_csv(f)
        #     logger.info(f"[dataset.redial] Loaded {len(movies)} movie samples.")
        #     self.movie2ind = {row["movieName"]: row["movieId"] for _, row in movies.iterrows()}
        #     self.ind2movie = {row["movieId"]: row["movieName"] for _, row in movies.iterrows()}
        #     logger.info(f"[dataset.redial] Loaded movie vocabulary: {len(self.movie2ind)} movies.")

    def _data_preprocess(self, train_data, valid_data, test_data):
        processed_train_data = self._raw_data_process(train_data)
        logger.info("[dataset.redial] Processed training data.")
        processed_valid_data = self._raw_data_process(valid_data)
        logger.info("[dataset.redial] Processed validation data.")
        processed_test_data = self._raw_data_process(test_data)
        logger.info("[dataset.redial] Processed test data.")
        processed_side_data = None
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, raw_data):
        augmented_data = [
            self._merge_conv_data(diag, user_id=diag["user_id"], conv_id=diag["conv_id"])
            for diag in raw_data
        ]
        augmented_conv_dicts = []
        for diag in augmented_data:
            augmented_conv_list = self._augment_and_add(diag)
            augmented_conv_dicts.extend(augmented_conv_list)

        return augmented_conv_dicts

    def _get_movie_mentioned(self, text: str = None, text_token_ids: list = None):
        if text is not None:
            movie_mentioned_list = set()
            for movie_name, movie_id in self.movie2ind.items():
                if movie_name in text:
                    movie_mentioned_list.add(movie_id)

            movie_mentioned_list = list(movie_mentioned_list)
            return movie_mentioned_list
        elif text_token_ids is not None:
            raise NotImplementedError("Movie mention extraction from token IDs is not implemented.")
        else:
            raise ValueError(
                "Either text or text_token_ids must be provided to extract movie mentions."
            )

    def _merge_conv_data(self, conv, user_id, conv_id):
        augmented_data = []
        last_role = None
        for uttr in conv["dialog"]:
            text_token_ids = [
                self.token2id.get(word, self.token2id["<unk>"])
                for word in self.tokenizer.encode(uttr["text"])
            ]
            movie_ids = self._get_movie_mentioned(text=uttr["text"])
            role = uttr["senderWorkerId"]
            if role == last_role:
                augmented_data[-1]["text"] += text_token_ids
                augmented_data[-1]["movie"] += movie_ids
            else:
                augmented_data.append(
                    {
                        "user_id": user_id,
                        "conv_id": conv_id,
                        "role": role,
                        "text_token_ids": text_token_ids,
                        "movie_mentioned": movie_ids,
                    }
                )
            last_role = role

        return augmented_data

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_movies = [], []
        for i, turn in enumerate(raw_conv_dict):
            # role = turn['role']
            turn_tokens = turn["text_token_ids"]
            turn_movies = turn["movie_mentioned"]

            if len(context_tokens) > 0:
                conv_dict = {
                    "role": turn["role"],
                    "context_tokens": copy(context_tokens),
                    "context_movies": copy(context_movies),
                    "movies": turn_movies,
                    "response_tokens": turn_tokens,
                }
                augmented_conv_dicts.append(conv_dict)
            context_tokens.append(turn_tokens)
            context_movies += turn_movies
        return augmented_conv_dicts
