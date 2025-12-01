# import os
# import json
# import pickle
# import pandas as pd
# from copy import copy
# from typing import Dict, List, Optional

# import torch
# from torch.nn.functional import cosine_similarity
# from transformers import AutoTokenizer
# from loguru import logger
# from tqdm import tqdm

# from crslab.config import DATASET_PATH
# from crslab.data.dataset.base import BaseDataset
# from .resources import resources


# class ReDialDataset(BaseDataset):
#     """
#     ~_data structure:
#     [
#         {
#             "role": "Seeker" or "Recommender",
#             "context_tokens": [[his1], [his2], ...],
#             "context_movies": [...],
#             "movies": [...],
#             "response": [cur],
#         },
#         ...
#     ]
#     """

#     def __init__(
#         self,
#         opt,
#         tokenize: str = "llava",
#         restore: bool = False,
#         save: bool = True,
#         tokenizer: Optional[AutoTokenizer] = None,
#     ):
#         resource = resources.get(tokenize, None)
#         if resource is None:
#             logger.info(f"[dataset.redial] Resource for tokenizer '{tokenize}' not found.")
#         else:
#             logger.info(f"[dataset.redial] Using resource for tokenizer '{tokenize}'.")

#         self.opt = opt
#         self.dpath = os.path.join(DATASET_PATH, "redial", tokenize)
#         logger.info(f"[dataset.redial] Dataset path set to {self.dpath}.")

#         if tokenizer is not None:
#             self.tokenizer = tokenizer
#             logger.info("[dataset.redial] Using provided tokenizer with special tokens.")
#         else:
#             logger.info(f"[dataset.redial] Loading tokenizer from {os.path.join(self.dpath, 'tokenizer')}.")
#             assert os.path.exists(os.path.join(self.dpath, "tokenizer")), \
#                 f"Tokenizer path {os.path.join(self.dpath, 'tokenizer')} does not exist."
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 os.path.join(self.dpath, "tokenizer"),
#             )
#             # print(self.tokenizer)
#         # self._verify_tokenizer_special_tokens()

#         super().__init__(opt, dpath=self.dpath, resource=resource, restore=restore, save=save)

#     def _verify_tokenizer_special_tokens(self):
#         DEFAULT_HYPERGRAPH_TOKEN = "<hgraph>"
#         DEFAULT_HYPERGRAPH_PATCH_TOKEN = "<hg_patch>"
#         DEFAULT_HG_START_TOKEN = "<hg_start>"
#         DEFAULT_HG_END_TOKEN = "<hg_end>"
#         required_tokens = [
#             DEFAULT_HYPERGRAPH_TOKEN,
#             DEFAULT_HYPERGRAPH_PATCH_TOKEN,
#             DEFAULT_HG_START_TOKEN,
#             DEFAULT_HG_END_TOKEN,
#         ]
#         for token in required_tokens:
#             if token not in self.tokenizer.get_vocab():
#                 print(self.tokenizer.all_special_tokens)
#                 logger.error(f"Tokenizer is missing required special token: {token}")
#                 raise ValueError(f"Tokenizer is missing required special token: {token}")
#         logger.info("[dataset.redial] All required special tokens are present in the tokenizer.")
#         self.hg_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_HYPERGRAPH_TOKEN)
#         self.hg_start_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_HG_START_TOKEN)
#         self.hg_end_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_HG_END_TOKEN)
#         self.hg_patch_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_HYPERGRAPH_PATCH_TOKEN)

#     def _load_data(self):
#         # raise NotImplementedError("Use _load_raw_data instead for ReDialDataset.")
#         train_data, valid_data, test_data = self._load_raw_data()
#         self._load_vacab()
#         self._load_other_data()

#         vocab = {
#             "ind2tok": self.ind2tok,
#             "tok2ind": self.tok2ind,
#             "ind2movie": self.ind2movie,
#             "movie2ind": self.movie2ind,
#             "vocab_size": len(self.tok2ind),
#             "n_movies": len(self.ind2movie),
#         }
#         return train_data, valid_data, test_data, vocab

#     def _load_raw_data(self):
#         if not os.path.exists(self.dpath):
#             logger.error(f"Dataset path {self.dpath} does not exist.")
#             raise FileNotFoundError(f"Dataset `path` {self.dpath} does not exist.")

#         with open(os.path.join(self.dpath, "train_data.json"), "r", encoding="utf-8") as f:
#             train_data = json.load(f)
#             logger.info(f"[dataset.redial] Loaded {len(train_data)} training samples.")
#         with open(os.path.join(self.dpath, "valid_data.json"), "r", encoding="utf-8") as f:
#             valid_data = json.load(f)
#             logger.info(f"[dataset.redial] Loaded {len(valid_data)} validation samples.")
#         with open(os.path.join(self.dpath, "test_data.json"), "r", encoding="utf-8") as f:
#             test_data = json.load(f)
#             logger.info(f"[dataset.redial] Loaded {len(test_data)} test samples.")

#         return train_data, valid_data, test_data

#     def _load_vacab(self):
#         if not hasattr(self, "tokenizer") or self.tokenizer is None:
#             logger.error("Tokenizer must be set before loading vocabulary.")
#             raise ValueError("Tokenizer must be set before loading vocabulary.")
#         # self.tok2ind = {token: idx for token, idx in self.tokenizer.get_vocab().items()}
#         # self.ind2tok = {idx: token for token, idx in self.tokenizer.get_vocab().items()}
#         with open(os.path.join(self.dpath, "token2ind.json"), "r", encoding="utf-8") as f:
#             self.tok2ind = json.load(f)
#             self.ind2tok = {idx: token for token, idx in self.tok2ind.items()}
#             logger.info("[dataset.redial] Vocabulary loaded from tokenizer.")
#             logger.info(f"[dataset.redial] Vocabulary size: {len(self.tok2ind)} tokens.")

#     def _load_other_data(self):
#         if not os.path.exists(self.dpath):
#             logger.error(f"Dataset path {self.dpath} does not exist.")
#             raise FileNotFoundError(f"Dataset path {self.dpath} does not exist.")
#         with open(os.path.join(self.dpath, "movie2ind.json"), "r", encoding="utf-8") as f:
#             self.movie2ind = json.load(f) # movie2ind: {movie_name: movie_id}: {"The Godfather": "123", ...}
#             self.movie2ind = { movie_name: int(movie_id) for movie_name, movie_id in self.movie2ind.items() }
#         with open(os.path.join(self.dpath, "ind2movie.json"), "r", encoding="utf-8") as f:
#             self.id2movie = json.load(f)
#             self.id2movie = {int(movie_id): movie_name for movie_id, movie_name in self.id2movie.items()}
#         self.ind2id = [int(movie_id) for movie_id in self.id2movie.keys()]
#         logger.info(f"[dataset.redial] Loaded movie vocabulary: {len(self.movie2ind)} movies.")
        
#         if self.opt.get("load_saved_embeddings", False):
#             self.txt_dim = self.opt.get("txt_dim", 0)
#             self.img_dim = self.opt.get("img_dim", 0)
#             self.vdo_dim = self.opt.get("vdo_dim", 0)
#             self.ado_dim = self.opt.get("ado_dim", 0)
#             logger.info(f"modalities dimensions - txt: {self.txt_dim}, img: {self.img_dim}, vdo: {self.vdo_dim}, ado: {self.ado_dim}")
#             self._load_embeddings()
                
#             logger.info("[dataset.redial] Initialized lazy loading for multi-modal embeddings.")
#             logger.info(f"[dataset.redial] Total movies: {len(self.movie2ind)}")
#             logger.info(f"[dataset.redial] Total movie_ids : {len(self.ind2movie)}")
#             self._precompute_similarity()

#         else:
#             logger.error("Loading embeddings on-the-fly is not implemented yet.")
#             raise NotImplementedError("Loading embeddings on-the-fly is not implemented yet.")
    
#     def _load_embeddings(self):
#         print(os.listdir(os.path.join(self.dpath, "embeddings")))
#         self.embeddings = {}
#         self.embeddings['txt'] = torch.load(os.path.join(self.dpath, "embeddings", "txt_embeddings.pt"), map_location='cpu')
#         logger.info("[dataset.redial] txt embeddings loaded from files.")
#         self.embeddings['img'] = torch.load(os.path.join(self.dpath, "embeddings", "img_embeddings.pt"), map_location='cpu')
#         logger.info("[dataset.redial] img embeddings loaded from files.")
#         self.embeddings['vdo'] = torch.load(os.path.join(self.dpath, "embeddings", "vdo_embeddings.pt"), map_location='cpu')
#         logger.info("[dataset.redial] vdo embeddings loaded from files.")
#         self.embeddings['ado'] = torch.load(os.path.join(self.dpath, "embeddings", "ado_embeddings.pt"), map_location='cpu')
#         logger.info("[dataset.redial] ado embeddings loaded from files.")
        
#         for modality in ['txt', 'img', 'vdo', 'ado']:
#             converted = {}
#             for key, value in self.embeddings[modality].items():
#                 int_key = int(key)
#                 converted[int_key] = value.float()
#             self.embeddings[modality] = converted
        
#         self.movie_embs = { 'txt': None, 'img': None, 'vdo': None, 'ado': None }
#         for modality in ['txt', 'img', 'vdo', 'ado']:
#             self.movie_embs[modality] = torch.stack([
#                 # self.embeddings[modality][movie_id]
#                 self.get_embedding(movie_id, modality, return_zero_if_missing=True)
#                 for movie_id in self.ind2id
#             ], dim=0).float()
#         self.zero_embeddings = {
#             'txt': torch.zeros(self.txt_dim),
#             'img': torch.zeros(self.img_dim),
#             'vdo': torch.zeros(self.vdo_dim),
#             'ado': torch.zeros(self.ado_dim)
#         }



#     def get_embedding(self, movie_id , modality:str='txt', return_zero_if_missing:bool=True):
#         assert modality in ['txt', 'img', 'vdo', 'ado'], "Modality must be one of 'txt', 'img', 'vdo', 'ado'."
#         if isinstance(movie_id, str):
#             movie_id = int(movie_id)
        
#         if movie_id in self.embeddings[modality]:
#             return self.embeddings[modality][movie_id]

#         if return_zero_if_missing:
#             return self.zero_embeddings[modality]
#         else:
#             return None
    

#     # def _precompute_similarity(self):
#     #     """
#     #     self.similarity_matrices: {
#     #         'txt': { '<movie_id>': [(<similar_movie_id_1>, <similarity_score_1>), ...], ... },
#     #         'img': { '<movie_id>': [(<similar_movie_id_1>, <similarity_score_1>), ...], ... },
#     #         'vdo': { '<movie_id>': [(<similar_movie_id_1>, <similarity_score_1>), ...], ... },
#     #         'ado': { '<movie_id>': [(<similar_movie_id_1>, <similarity_score_1>), ...], ... },
#     #     }
#     #     """
#     #     restore_similarity_topk = self.opt.get("restore_similarity_topk", 50)
#     #     self.similarity_matrices = {}
#     #     for modality in ['txt', 'img', 'vdo', 'ado']:
#     #         all_embs = []
#     #         movie_ids = []
#     #         for movie_id in self.movie2ind.values():
#     #             emb = self.get_embedding(movie_id, modality, return_zero_if_missing=True) # ！！！这里使用零向量填充缺失值，后续可考虑使用其他模态交集
#     #             all_embs.append(emb.unsqueeze(0))  # 添加一个维度以便堆叠
#     #             movie_ids.append(movie_id)
#     #         all_embs_tensor = torch.cat(all_embs, dim=0) # shape: (num_movies, emb_dim)
#     #         all_embs_norm = torch.nn.functional.normalize(all_embs_tensor, p=2, dim=1)
#     #         sim_matrix = torch.mm(all_embs_norm, all_embs_norm.t()) # shape: (num_movies, num_movies)
#     #         for i in range(len(movie_ids)):
#     #             sim_matrix[i, i] = -float('inf')

#     #         modality_sim_dict = {}
#     #         for i, movie_id in enumerate(movie_ids):
#     #             k = min(restore_similarity_topk, len(movie_ids) - 1)
#     #             topk_values, topk_indices = torch.topk(sim_matrix[i], k, dim=0)
#     #             for value, index in zip(topk_values, topk_indices):
#     #                 similar_id = movie_ids[index.item()]
#     #                 if movie_id not in modality_sim_dict:
#     #                     modality_sim_dict[movie_id] = []
#     #                 modality_sim_dict[movie_id].append((similar_id, value.item()))
#     #         self.similarity_matrices[modality] = modality_sim_dict
#     #     logger.info("[dataset.redial] Precomputed similarity matrices for all modalities.")
#     #     # print({modality: self.similarity_matrices[modality] for modality in self.similarity_matrices})
#     def _compute_sim_mat(self, modality: str):
#         X = self.movie_embs[modality]
#         X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
#         sim_mat = torch.mm(X_norm, X_norm.t())

#         return sim_mat

#     def _build_hypergraph(self):


    
#     def _data_preprocess(self, train_data, valid_data, test_data):
#         logger.info("[dataset.redial] Processing training data.")
#         processed_train_data = self._raw_data_process(train_data)
#         logger.info("[dataset.redial] Processing valid data.")
#         processed_valid_data = self._raw_data_process(valid_data)
#         logger.info("[dataset.redial] Processing test data.")
#         processed_test_data = self._raw_data_process(test_data)
#         processed_side_data = None
#         return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

#     def _raw_data_process(self, raw_data):
#         augmented_data = [
#             self._merge_conv_data(diag, user_id=diag["user_id"], conv_id=diag["conv_id"])
#             for diag in raw_data
#         ]
#         augmented_conv_dicts = []
#         for diag in tqdm(augmented_data, desc="Processing conversations"):
#             augmented_conv_list = self._augment_and_add(diag)
#             augmented_conv_dicts.extend(augmented_conv_list)
     
#         return augmented_conv_dicts

#     def _merge_conv_data(self, conv, user_id, conv_id):
#         augmented_data = []
#         last_role = None
#         for uttr in conv["dialog"]:
#             text_token_ids = [
#                 self.tok2ind.get(token, self.tok2ind["<unk>"]) for token in uttr["text"]
#             ]
#             role = uttr["role"]
#             if role == last_role:
#                 augmented_data[-1]["text"] += text_token_ids
#                 augmented_data[-1]["movies"] += uttr["movies"]
#             else:
#                 augmented_data.append({
#                         "user_id": user_id,
#                         "conv_id": conv_id,
#                         "role": role,
#                         "text": text_token_ids,
#                         "movies": uttr["movies"],
#                     })
#             last_role = role

#         return augmented_data

#     def _augment_and_add(self, raw_conv_dict):
#         """
#         将对话转为历史->当前轮次的形式:
#         {[uttr1]},
#         {[uttr1, uttr2]},
#         {[uttr1, uttr2, uttr3]},
#         ...
#         同时添加基于多模态相似度的超边物品（每个模态分别扩展）
#         """
#         augmented_conv_dicts = []
#         context_tokens, context_movies = [], []
#         conv_id = raw_conv_dict[0]["conv_id"]
#         user_id = raw_conv_dict[0]["user_id"]
        
#         hyperedge_modalities = ['txt', 'img', 'vdo', 'ado']  # 所有模态
#         hyperedge_top_k = self.opt.get("hyperedge_top_k", 5)  # 每个电影扩展5个相似电影
#         hyperedge_threshold = self.opt.get("hyperedge_threshold", 0.0)  # 相似度阈值
        
#         for i, turn in enumerate(raw_conv_dict):
#             # role = turn['role']
#             turn_tokens = turn["text"]
#             turn_movies = turn["movies"]

#             if len(context_tokens) > 0:
#                 related_movies = {}
#                 for modality in hyperedge_modalities:
#                     related_movies[modality] = self._add_related_movies(
#                         context_movies,
#                         modality=modality,
#                         top_k=hyperedge_top_k,
#                         similarity_threshold=hyperedge_threshold
#                     )
#                     assert len(related_movies[modality]) == len(context_movies), \
#                         f"Length mismatch in related movies for modality {modality} at turn {i}"

#                 conv_dict = { # final dict
#                     "role": turn["role"],
#                     "movies": turn_movies,
#                     "response": turn_tokens,
#                     "user_id": user_id,
#                     "conv_id": conv_id,
#                     "context_tokens": copy(context_tokens),
#                     "context_movies": copy(context_movies), # shape: (N, )
#                     "related_movies": copy(related_movies), # shape: {modality : List[List[str]]}, corresponding to context_movies
#                 }
#                 augmented_conv_dicts.append(conv_dict)
            
#             context_tokens.append(turn_tokens)
#             context_movies += turn_movies
#             context_movies = list(set(context_movies))  # 去重
#         return augmented_conv_dicts

#     def _add_related_movies(self, context_movies, modality='txt', top_k=10, similarity_threshold=0.0):
#         """
#         得到context_movies中每个电影对应的模态topk相似电影列表
#         Return:
#             related_movies: List[List[str]]
#         """
#         related_movies = []
#         for movie_id in context_movies:
#             cur_related = []
#             movie_sim_list = self.similarity_matrices[modality].get(movie_id, [])
#             # print(f"Movie ID: {movie_id}, Similar Movies: {movie_sim_list}")
#             # logger.info(f"Movie ID: {movie_id}, Similar Movies: {movie_sim_list}") #############################
#             # movie_sim_list = sorted(movie_sim_list, key=lambda x: x[1], reverse=True)
#             if len(movie_sim_list) != 0:
#                 for similar_id, value in movie_sim_list:
#                     if value >= similarity_threshold:
#                         cur_related.append(similar_id)
#                     if len(cur_related) >= top_k:
#                         break
#             if len(cur_related) == 0:
#                 pass
#             related_movies.append(cur_related)
#             # exit()
#         # if len(context_movies) != 0:###############################
#         #     exit()
#         return related_movies