import os
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from base import BaseDataset

# class ABC_Dataset:
#     def __init__(self, dataset_path):
#         self.dataset_path = dataset_path


class DuRecDial_Dataset(BaseDataset):
    """
    Dataset info:
    - train/dev/test
        - list of dict
        - dict keys: ['goal', 'user_profile', 'conversation', 'goal_topic_list',
                      'goal_type_list', 'situation', 'knowledge']
        - conversation: list of str
        - knowledge: list of list, [[h,r,t], ...]
    """

    def __init__(self, config=None, dataset_path=None):
        super().__init__(config, dataset_path)
        self.modality = ["txt", "img", "vdo", "ado"]
        self.modality_folder = ["txt_emb", "img_emb", "vdo_emb", "audio_emb"]
        self.train_data, self.dev_data, self.test_data = self.load_data()
        self.movie_id2name, self.movie_name2id = self._get_movie_map()
        self.user_id2name, self.user_name2id = self._get_user_map()
        self.data_preprocess()
        # self.user_

    def _load_raw_data(self, path):
        """load raw .txt data, and turn to list of dict"""
        with open(path, "r", encoding="utf-8") as f:
            raw_data_txt = f.readlines()
        raw_data = [json.loads(line) for line in raw_data_txt]
        return raw_data

    def load_data(self):
        train_path = os.path.join(self.dataset_path, "en_train.txt")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data file not found: {train_path}")
        train_data = self._load_raw_data(train_path)

        dev_path = os.path.join(self.dataset_path, "en_dev.txt")
        if not os.path.exists(dev_path):
            raise FileNotFoundError(f"Development data file not found: {dev_path}")
        dev_data = self._load_raw_data(dev_path)

        test_path = os.path.join(self.dataset_path, "en_test.txt")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data file not found: {test_path}")
        test_data = self._load_raw_data(test_path)

        return train_data, dev_data, test_data

    def _get_movie_map(self):
        """
        get movie id to name and name to id mapping
        requires movies_with_mentions.csv
        returns two dicts: movie_id2name, movie_name2id
        """

        file_path = os.path.join(self.dataset_path, "movies_with_mentions.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Mentioned movies list file not found: {file_path}")

        df = pd.read_csv(file_path, encoding="ANSI")
        movie_id2name = {}
        movie_name2id = {}
        for _, row in df.iterrows():
            movie_id = row["movie_id"]
            movie_name = row["movie_name"]
            movie_id2name[movie_id] = movie_name
            movie_name2id[movie_name] = movie_id
        return movie_id2name, movie_name2id

    def _get_user_map(self):
        """
        get user id to name and name to id mapping
        requires train/dev/test data to be loaded, so call after _load_data
        returns two dicts: user_id2name, user_name2id
        """

        if not (self.train_data and self.dev_data and self.test_data):
            raise ValueError("Data not loaded. Cannot create user mapping.")
        user_set = set(
            [
                conv["user_profile"]["Name"]
                for conv in self.train_data + self.dev_data + self.test_data
            ]
        )
        user_id2name = {}
        user_name2id = {}
        for user_id, user_name in enumerate(user_set):
            user_id2name[user_id] = user_name
            user_name2id[user_name] = user_id
        return user_id2name, user_name2id

    def _get_modality_emb(self):
        if not self.movie_id2name:
            raise ValueError("Movie mapping not created. Cannot create modality embeddings.")

        modality_emb = {i: {m: None for m in self.modality} for i in self.movie_id2name.keys()}
        for m, m_folder in zip(self.modality, self.modality_folder):
            emb_path = os.path.join(self.dataset_path, m_folder)
            if not os.path.exists(emb_path):
                raise FileNotFoundError(f"Modality embedding folder not found: {emb_path}")
            for movie_id in tqdm(self.movie_id2name):
                if os.path.exists(os.path.join(emb_path, f"{movie_id}.pt")):
                    emb = torch.load(os.path.join(emb_path, f"{movie_id}.pt"))
                    modality_emb[movie_id][m] = emb
            print(
                f"Loaded {m} embeddings for {sum(1 for v in modality_emb.values() if v[m] is not None)} movies."
            )

        return modality_emb

    def data_preprocess(self):
        self._get_modality_emb()
        pass


if __name__ == "__main__":
    path = "data\\DuRecDial"
    dataset = DuRecDial_Dataset(config=None, dataset_path=path)
    print(len(dataset.train_data))
    print(len(dataset.dev_data))
    print(len(dataset.test_data))
