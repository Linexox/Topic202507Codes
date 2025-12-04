from crslab.data import ReDialDataset2
from crslab.model.crs.hollarec.HypergraphLlava import MMHypergraphLlavaModel
from .conversation import default_conversation

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import AutoTokenizer
import numpy as np

from typing import List, Dict
import dataclasses

np.random.seed(42)

def apply_map(lst, dct):
    return [dct[item] for item in lst]

class Task_1_Dataset(Dataset):
    def __init__(self, dataset: ReDialDataset2, split, k_values=[5]):
        self.dataset = dataset
        split_data = getattr(self.dataset, f"{split}_data")
        self.data = []
        for k in k_values:
            self.data.extend(self._prepare_data(split_data, k))
    
    def _get_hypergraph(
        self,
        hyperedges: Dict[str, List[List[int]]],
    ):
        hypergraphs = {'txt': None, 'img': None, 'vdo': None, 'ado': None}
        hypergraph_nodes = {'txt': [], 'img': [], 'vdo': [], 'ado': []}
        for m in ['txt', 'img', 'vdo', 'ado']:
            all_nodes = np.unique(np.concatenate(hyperedges[m])).tolist()
            node_feat = torch.stack([self.dataset.get_embedding(mv, m) for mv in all_nodes], dim=0)
            edge_index = [[mid, i] for i, hedge in enumerate(hyperedges[m]) for mid in hedge]
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            hypergraphs[m] = Data(x=node_feat, edge_index=edge_index)
            hypergraph_nodes[m] = all_nodes
        return hypergraphs, hypergraph_nodes

    def _prepare_data(self, split_data: List[Dict], k: int):
        data = []
        for conv in split_data:
            if len(conv['movies']) == 0:
                continue
            center_mv = conv['movies'].copy()
            all_mv = conv['movies'].copy()
            m_hyperedges = {'txt': [], 'img': [], 'vdo': [], 'ado': []}
            for m in ['txt', 'img', 'vdo', 'ado']:
                for mv in center_mv:
                    hyperedge = self.dataset._get_related_movies(mv, m, k=k, sample_method='rand')
                    m_hyperedges[m].append([mv]+hyperedge)
                    all_mv.extend(hyperedge)
            all_mv = list(set(all_mv))
            mv_titles = apply_map(all_mv, self.dataset.id2movie)
            data.append({
                'mv_titles': mv_titles,
                'center_mv': center_mv,
                'm_hyperedges': m_hyperedges,
                'all_mv': all_mv,
            })
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        conv = default_conversation.copy()
        data = self.data[i]
        shuffled_mv_titles = data['mv_titles'].copy()
        np.random.shuffle(shuffled_mv_titles)
        hgraphs, hgraphs_node = self._get_hypergraph(data['m_hyperedges'])
        conv.append_message(
            conv.roles[0],
            ("Given a shuffled list of movie titles: " + \
             f"[{', '.join(shuffled_mv_titles)}], " + \
             f"and the following hypergraph connections:\n" +
             f"Text-based Hypergraph: {"<txt_hg_start>"+"<txt_hg_patch>"*hgraphs['txt'].x.size(0)+"<txt_hg_end>"} \n" +
             f"Image-based Hypergraph: {"<img_hg_start>"+"<img_hg_patch>"*hgraphs['img'].x.size(0)+"<img_hg_end>"} \n" +
             f"Video-based Hypergraph: {"<vdo_hg_start>"+"<vdo_hg_patch>"*hgraphs['vdo'].x.size(0)+"<vdo_hg_end>"} \n" +
             f"Audio-based Hypergraph: {"<ado_hg_start>"+"<ado_hg_patch>"*hgraphs['ado'].x.size(0)+"<ado_hg_end>"} \n" +
             "each hyperedge connects a set of related movies. \n" +
             "Your task is to identify and reorder the movie titles back to their original order based on the hypergraph connections. ")
        )
        txt_node_titles = apply_map(hgraphs_node['txt'], self.datase.id2movie)
        vdo_node_titles = apply_map(hgraphs_node['vdo'], self.datase.id2movie)
        ado_node_titles = apply_map(hgraphs_node['ado'], self.datase.id2movie)
        img_node_titles = apply_map(hgraphs_node['img'], self.datase.id2movie)
        conv.append_message(
            conv.roles[1],
            ("Based on the hypergraph connections provided, I have identified the following groupings of related movies: \n" +
             f"Text-based Hypergraph Nodes: {txt_node_titles} \n" +
             f"Image-based Hypergraph Nodes: {img_node_titles} \n" +
             f"Video-based Hypergraph Nodes: {vdo_node_titles} \n" +
             f"Audio-based Hypergraph Nodes: {ado_node_titles} \n" +
             "Using these groupings, I have reordered the movie titles to their original sequence." )
        )
        prompt = conv.get_prompt()
        return {
            'prompt': prompt,
            'hgraphs': hgraphs,
            'hgraphs_node': hgraphs_node,
            'shuffled_mv_titles': shuffled_mv_titles,
            'center_mv': data['center_mv'],
            'all_mv': data['all_mv'],
        }

class Task_1_DataLoader:
    def __init__(self, dataset: ReDialDataset2, split, k_values=[5], batch_size=1):
        self.dataset = Task_1_Dataset(dataset, split, k_values)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        prompts = [item['prompt'] for item in batch]
        hgraphs = {m: [item['hgraphs'][m] for item in batch] for m in ['txt', 'img', 'vdo', 'ado']}
        hgraphs_node = {m: [item['hgraphs_node'][m] for item in batch] for m in ['txt', 'img', 'vdo', 'ado']}
        shuffled_mv_titles = [item['shuffled_mv_titles'] for item in batch]
        center_mv = [item['center_mv'] for item in batch]
        all_mv = [item['all_mv'] for item in batch]
        return {
            'prompts': prompts,
            'hgraphs': hgraphs,
            'hgraphs_node': hgraphs_node,
            'shuffled_mv_titles': shuffled_mv_titles,
            'center_mv': center_mv,
            'all_mv': all_mv
        }
    
    def __iter__(self):
        return iter(self.dataloader)

        
class Task_1_Trainer:
    def __init__(self, dataset: ReDialDataset2, split, k_values=[5], batch_size=1):
        self.dataset = dataset
        self.dataloader = Task_1_DataLoader(dataset, split, k_values, batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained("hollarec/hollarec-llama-7b")
        self.model = MMHypergraphLlavaModel.from_pretrained("hollarec/hollarec-mm-hypergraph-llava")
    
    def train(self):
        for batch in self.dataloader:
            prompts = batch['prompts']
            hgraphs = batch['hgraphs']
            hgraphs_node = batch['hgraphs_node']
            shuffled_mv_titles = batch['shuffled_mv_titles']
            center_mv = batch['center_mv']
            all_mv = batch['all_mv']
            # Training logic goes here