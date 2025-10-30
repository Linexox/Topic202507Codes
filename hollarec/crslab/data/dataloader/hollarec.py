from copy import copy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


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
        self.split_token_idx = vocab["tok2ind"].get("<split>", None)

        self.context_truncate = opt.get("context_truncate", None)
        self.response_truncate = opt.get("response_truncate", None)

    def rec_process_fn(self):
        """
        推荐任务数据预处理：为每个提及的电影创建一个训练样本
        """
        augment_dataset = []
        for conv in self.dataset:
            if conv["role"] == "Recommender":
                for movie in conv["movies"]:
                    augment_conv = {
                        "context_tokens": copy(conv["context_tokens"]),
                        "context_movies": copy(conv["context_movies"]),
                        "response": copy(conv["response"]),
                        "movie": movie,
                        "role": conv["role"],
                        "user_id": conv["user_id"],
                        "conv_id": conv["conv_id"],
                    }
                    augment_dataset.append(augment_conv)

        return augment_dataset

    def rec_batchify(self, batch):
        """
        推荐任务批处理
        """
        batch_context_tokens = []
        batch_context_movies = []
        batch_movie = []

        for conv in batch:
            batch_context_tokens.append(conv["context_tokens"])
            batch_context_movies.append(conv["context_movies"])
            batch_movie.append(conv["movie"])

        res = {
            "context_tokens": batch_context_tokens,
            "context_movies": batch_context_movies,
            "movie": torch.tensor(batch_movie, dtype=torch.long),
        }

        return res

    def conv_process_fn(self):
        """
        对话生成任务数据预处理：过滤出 Recommender 角色的对话样本
        """
        dataset = []
        for conv in self.dataset:
            if conv["role"] == "Recommender":
                dataset.append(conv)
        return dataset

    def conv_batchify(self, batch):
        """
        对话生成任务的批处理

        处理步骤:
        1. 合并多轮上下文并截断
        2. 为response添加start/end token并截断
        3. Padding到相同长度
        4. 收集电影ID和超图数据
        """
        batch_context_tokens = []
        batch_context_movies = []
        batch_response = []
        batch_user_id = []
        batch_conv_id = []

        for conv in batch:
            # 1. 处理上下文：合并多轮对话 -> 截断
            context = truncate(
                merge_utt(
                    conv["context_tokens"],
                    start_token_idx=self.start_token_idx,
                    split_token_idx=self.split_token_idx,
                    final_token_idx=self.end_token_idx,
                ),
                self.context_truncate,
                truncate_tail=False,  # 保留最近的对话
            )
            batch_context_tokens.append(context)

            # 2. 处理response：截断 -> 添加start/end token
            response = add_start_end_token_idx(
                truncate(conv["response"], self.response_truncate - 2),
                start_token_idx=self.start_token_idx,
                end_token_idx=self.end_token_idx,
            )
            batch_response.append(response)

            # 3. 收集电影ID
            batch_context_movies.append(conv["context_movies"])

            # 4. 收集元信息
            batch_user_id.append(conv["user_id"])
            batch_conv_id.append(conv["conv_id"])

        # Padding 并转为 tensor
        res = {
            "context_tokens": padded_tensor(
                batch_context_tokens, self.pad_token_idx, pad_tail=False  # 左padding，保留最近对话
            ),
            "context_movies": batch_context_movies,  # 保持为list，模型内部处理
            "response": padded_tensor(
                batch_response, self.pad_token_idx, pad_tail=True  # 右padding
            ),
            "user_id": batch_user_id,
            "conv_id": batch_conv_id,
        }

        return res

    def policy_batchify(self, *args, **kwargs):
        pass
