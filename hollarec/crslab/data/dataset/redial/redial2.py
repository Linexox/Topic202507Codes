import os
import json
import pickle
import numpy as np
import pandas as pd
from copy import copy
from typing import Dict, List, Optional

import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer
from loguru import logger
from tqdm import tqdm

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from .resources import resources


class ReDialDataset2(BaseDataset):
    """
    ~_data structure:
    [
        {
            "role": "Seeker" or "Recommender",
            "context_tokens": [[his1], [his2], ...],
            "context_movies": [...],
            "movies": [...],
            "response": [cur],
        },
        ...
    ]
    """

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
        self.dpath = os.path.join(DATASET_PATH, "redial", tokenize)
        logger.info(f"[dataset.redial] Dataset path set to {self.dpath}.")

        if tokenizer is not None:
            self.tokenizer = tokenizer
            logger.info("[dataset.redial] Using provided tokenizer with special tokens.")
        else:
            logger.info(f"[dataset.redial] Loading tokenizer from {os.path.join(self.dpath, 'tokenizer')}.")
            assert os.path.exists(os.path.join(self.dpath, "tokenizer")), \
                f"Tokenizer path {os.path.join(self.dpath, 'tokenizer')} does not exist."
            self.tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(self.dpath, "tokenizer"),
            )
            # print(self.tokenizer)

        super().__init__(opt, dpath=self.dpath, resource=resource, restore=restore, save=save)

    def _load_data(self):
        # raise NotImplementedError("Use _load_raw_data instead for ReDialDataset.")
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vacab()
        self._load_other_data()

        vocab = {
            "ind2tok": self.ind2tok,
            "tok2ind": self.tok2ind,
            "id2movie": self.id2movie,
            "movie2id": self.movie2id,
            'idx2id': self.idx2id,
            'movie2idx': self.movie2idx,
            "vocab_size": len(self.tok2ind),
            "n_movies": len(self.id2movie),
        }
        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        if not os.path.exists(self.dpath):
            logger.error(f"Dataset path {self.dpath} does not exist.")
            raise FileNotFoundError(f"Dataset `path` {self.dpath} does not exist.")

        with open(os.path.join(self.dpath, "train_data.json"), "r", encoding="utf-8") as f:
            train_data = json.load(f)
            logger.info(f"[dataset.redial] Loaded {len(train_data)} training samples.")
        with open(os.path.join(self.dpath, "valid_data.json"), "r", encoding="utf-8") as f:
            valid_data = json.load(f)
            logger.info(f"[dataset.redial] Loaded {len(valid_data)} validation samples.")
        with open(os.path.join(self.dpath, "test_data.json"), "r", encoding="utf-8") as f:
            test_data = json.load(f)
            logger.info(f"[dataset.redial] Loaded {len(test_data)} test samples.")

        return train_data, valid_data, test_data

    def _load_vacab(self):
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            logger.error("Tokenizer must be set before loading vocabulary.")
            raise ValueError("Tokenizer must be set before loading vocabulary.")
        with open(os.path.join(self.dpath, "token2ind.json"), "r", encoding="utf-8") as f:
            self.tok2ind = json.load(f)
            self.ind2tok = {idx: token for token, idx in self.tok2ind.items()}
            logger.info("[dataset.redial] Vocabulary loaded from tokenizer.")
            logger.info(f"[dataset.redial] Vocabulary size: {len(self.tok2ind)} tokens.")

    def _load_other_data(self):
        if not os.path.exists(self.dpath):
            logger.error(f"Dataset path {self.dpath} does not exist.")
            raise FileNotFoundError(f"Dataset path {self.dpath} does not exist.")
        with open(os.path.join(self.dpath, "movie2ind.json"), "r", encoding="utf-8") as f:
            movie2id = json.load(f) # movie2ind: {movie_name: movie_id}: {"The Godfather": "123", ...}
        with open(os.path.join(self.dpath, "ind2movie.json"), "r", encoding="utf-8") as f:
            id2movie = json.load(f)
        
        self.movie2id = { movie_name: int(movie_id) for movie_name, movie_id in movie2id.items()}
        self.id2movie = {int(movie_id): movie_name for movie_id, movie_name in id2movie.items()}
        self.idx2id = [int(movie_id) for movie_id in self.id2movie.keys()]
        self.movie2idx = {int(id): int(idx) for idx, id in enumerate(self.idx2id)}
        logger.info(f"[dataset.redial] Loaded movie vocabulary: {len(self.movie2id)} movies.")
        
        if self.opt.get("load_saved_embeddings", True):
            self.txt_dim = self.opt.get("txt_dim", 0)
            self.img_dim = self.opt.get("img_dim", 0)
            self.vdo_dim = self.opt.get("vdo_dim", 0)
            self.ado_dim = self.opt.get("ado_dim", 0)
            logger.info(f"modalities dimensions - txt: {self.txt_dim}, img: {self.img_dim}, vdo: {self.vdo_dim}, ado: {self.ado_dim}")
            self._load_embeddings()
            self._build_hyperedges()
        else:
            logger.error("Loading embeddings on-the-fly is not implemented yet.")
            raise NotImplementedError("Loading embeddings on-the-fly is not implemented yet.")
    
    def _load_embeddings(self):
        print(os.listdir(os.path.join(self.dpath, "embeddings")))
        self.embeddings = {}
        # ~_embeddings.pt structure: { movie_id (int) : embedding (torch.Tensor) }
        self.embeddings['txt'] = torch.load(os.path.join(self.dpath, "embeddings", "txt_embeddings.pt"), map_location='cpu') 
        logger.info("[dataset.redial] txt embeddings loaded from files.")
        self.embeddings['img'] = torch.load(os.path.join(self.dpath, "embeddings", "img_embeddings.pt"), map_location='cpu')
        logger.info("[dataset.redial] img embeddings loaded from files.")
        self.embeddings['vdo'] = torch.load(os.path.join(self.dpath, "embeddings", "vdo_embeddings.pt"), map_location='cpu')
        logger.info("[dataset.redial] vdo embeddings loaded from files.")
        self.embeddings['ado'] = torch.load(os.path.join(self.dpath, "embeddings", "ado_embeddings.pt"), map_location='cpu')
        logger.info("[dataset.redial] ado embeddings loaded from files.")
        self.zero_embeddings = {
            'txt': torch.zeros(self.txt_dim),
            'img': torch.zeros(self.img_dim),
            'vdo': torch.zeros(self.vdo_dim),
            'ado': torch.zeros(self.ado_dim)
        }
        self.movie_embs = { 'txt': None, 'img': None, 'vdo': None, 'ado': None }
        for m in ['txt', 'img', 'vdo', 'ado']: # converted keys from str to int
            converted = {}
            for k, v in self.embeddings[m].items():
                int_k = int(k)
                converted[int_k] = v.float()
            self.embeddings[m] = converted
            self.movie_embs[m] = torch.stack([
                self.get_embedding(mv_id, m, return_zero_if_missing=True)
                for mv_id in self.idx2id
            ], dim=0).float()

    def get_embedding(self, movie_id , modality:str='txt', return_zero_if_missing:bool=True):
        assert modality in ['txt', 'img', 'vdo', 'ado'], "Modality must be one of 'txt', 'img', 'vdo', 'ado'."
        if isinstance(movie_id, str):
            movie_id = int(movie_id)
        if movie_id in self.embeddings[modality]:
            return self.embeddings[modality][movie_id]
        elif return_zero_if_missing:
            return self.zero_embeddings[modality]
        else:
            raise KeyError(f"Embedding for movie_id {movie_id} in modality {modality} not found.")
    
    def _compute_sim_mat(self, modality: str):
        X = self.movie_embs[modality]
        X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
        sim_mat = torch.mm(X_norm, X_norm.t())

        return sim_mat

    def _build_hyperedges(self):
        self.m_hyperedges = [
            { 'txt': [], 'img': [], 'vdo': [], 'ado': [] }
            for _ in range(len(self.idx2id))
        ]
        hyperedge_top_k = self.opt.get("hyperedge_top_k", 100)
        for m in ['txt', 'img', 'vdo', 'ado']:
            m_sim_mat = self._compute_sim_mat(modality=m)
            for idx, id in enumerate(self.idx2id):
                sim_scores = np.array(m_sim_mat[idx])  # shape: (num_movies, )
                sim_scores[idx] = -1.00  # 排除自己
                sim_list = np.argsort(-sim_scores)[:hyperedge_top_k].tolist()  # top-k 相似电影索引列表
                self.m_hyperedges[idx][m] = sim_list
        logger.info("[dataset.redial] Hypergraph construction completed.")

    def _data_preprocess(self, train_data, valid_data, test_data):
        logger.info("[dataset.redial] Processing training data.")
        processed_train_data = self._raw_data_process(train_data)
        logger.info("[dataset.redial] Processing valid data.")
        processed_valid_data = self._raw_data_process(valid_data)
        logger.info("[dataset.redial] Processing test data.")
        processed_test_data = self._raw_data_process(test_data)
        processed_side_data = None
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, raw_data):
        augmented_data = [
            self._merge_conv_data(diag, user_id=diag["user_id"], conv_id=diag["conv_id"])
            for diag in raw_data
        ]
        augmented_conv_dicts = []
        for diag in tqdm(augmented_data, desc="Processing conversations"):
            augmented_conv_list = self._augment_and_add(diag)
            augmented_conv_dicts.extend(augmented_conv_list)
        return augmented_conv_dicts

    def _merge_conv_data(self, conv, user_id, conv_id):
        augmented_data = []
        last_role = None
        for uttr in conv["dialog"]:
            text_token_ids = [
                self.tok2ind.get(token, self.tok2ind["<unk>"]) for token in uttr["text"]
            ]
            role = uttr["role"]
            if role == last_role:
                augmented_data[-1]["text"] += text_token_ids
                augmented_data[-1]["movies"] += uttr["movies"]
            else:
                augmented_data.append({
                        "user_id": user_id,
                        "conv_id": conv_id,
                        "role": role,
                        "text": text_token_ids,
                        "movies": uttr["movies"],
                    })
            last_role = role

        return augmented_data

    def _augment_and_add(self, raw_conv_dict):
        """
        将对话转为历史->当前轮次的形式:
        {[uttr1]},
        {[uttr1, uttr2]},
        {[uttr1, uttr2, uttr3]},
        ...
        同时添加基于多模态相似度的超边物品（每个模态分别扩展）
        """
        augmented_conv_dicts = []
        context_tokens, context_movies = [], []
        conv_id = raw_conv_dict[0]["conv_id"]
        user_id = raw_conv_dict[0]["user_id"]
        
        hyperedge_top_k = self.opt.get("hyperedge_top_k", 10)  # 每个电影扩展10个相似电影
        
        for i, turn in enumerate(raw_conv_dict):
            # role = turn['role']
            turn_tokens = turn["text"]
            turn_movies = turn["movies"]

            if len(context_tokens) > 0:
                related_movies = {'txt': [], 'img': [], 'ado': [], 'vdo': []}
                for m in ['txt', 'img', 'ado', 'vdo']:
                    for mv in context_movies:
                        if related_movies.get(m) is None:
                            related_movies[m] = []
                        related_movies[m].append(
                            self._get_related_movies(mv, m, k=hyperedge_top_k, sample_method='topk')
                        )
                        
                    assert len(related_movies[m]) == len(context_movies), \
                        f"Length mismatch in related movies for modality {m} at turn {i}"

                conv_dict = { # final dict
                    "role": turn["role"],
                    "movies": turn_movies,
                    "response": turn_tokens,
                    "user_id": user_id,
                    "conv_id": conv_id,
                    "context_tokens": copy(context_tokens),
                    "context_movies": copy(context_movies), # shape: (N, )
                    "related_movies": copy(related_movies), # shape: {modality : List[List[str]]}, corresponding to context_movies
                }
                augmented_conv_dicts.append(conv_dict)
            
            context_tokens.append(turn_tokens)
            context_movies += turn_movies
            context_movies = list(set(context_movies))  # 去重
        return augmented_conv_dicts

    # def _add_related_movies(self, context_movies, m: str, k=20, sample_method='topk'):
    #     assert sample_method in ['topk','rand']
    #     related_movies = []
    #     for mv_id in context_movies:
    #         mv_idx = self.movie2idx.get(mv_id)
    #         n_sample = min(k, len(self.m_hyperedges[mv_idx][m]))
    #         if sample_method == 'rand':
    #             m_hyperedge = np.random.choice(
    #                 self.m_hyperedges[mv_idx][m],
    #                 size=n_sample,
    #                 replace=False
    #             ).tolist()
    #         else: # 'topk'
    #             m_hyperedge = self.m_hyperedges[mv_idx][m][:n_sample]
    #         m_hyperedge = [self.idx2id[idx] for idx in m_hyperedge]
    #         related_movies.append(m_hyperedge)
    #     return related_movies
    def _get_related_movies(self, mv_id, m: str, k=10, sample_method='topk'):
        assert sample_method in ['topk','rand']
        mv_idx = self.movie2idx.get(mv_id)
        n_sample = min(k, len(self.m_hyperedges[mv_idx][m]))
        if sample_method == 'rand':
            m_hyperedge = np.random.choice(
                self.m_hyperedges[mv_idx][m],
                size=n_sample,
                replace=False
            ).tolist()
        else: # 'topk'
            m_hyperedge = self.m_hyperedges[mv_idx][m][:n_sample]
        m_hyperedge = [self.idx2id[idx] for idx in m_hyperedge]
        return m_hyperedge