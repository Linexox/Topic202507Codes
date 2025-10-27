# @Time   : 2025/10/27
# @Author : Your Name
# @Email  : your.email@example.com

r"""
HypergraphLlava ReDial Dataset
==============================
ReDial dataset adapted for HypergraphLlava CRS model.
Uses transformers AutoTokenizer (LLaVA's tokenizer) instead of custom vocabulary.
"""

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from transformers import AutoTokenizer
from loguru import logger
from tqdm import tqdm


class HypergraphLlavaReDialDataset(Dataset):
    """
    ReDial dataset for HypergraphLlava CRS model
    
    This dataset:
    1. Uses LLaVA's tokenizer (from transformers)
    2. Supports both recommendation and conversation tasks
    3. Loads hypergraph data dynamically
    4. Returns properly formatted batches for the model
    
    Args:
        data_path: Path to preprocessed data (pkl file)
        tokenizer: HypergraphLlava tokenizer (AutoTokenizer)
        side_data: Dict containing item mappings and metadata
        task: 'rec' or 'conv'
        hypergraph_dir: Directory containing hypergraph embeddings
        max_length: Maximum sequence length
        use_hypergraph: Whether to load hypergraph data
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        side_data: Dict,
        task: str = 'rec',
        hypergraph_dir: Optional[str] = None,
        max_context_len: int = 512,
        max_response_len: int = 128,
        use_hypergraph: bool = True
    ):
        super().__init__()
        
        assert task in ['rec', 'conv'], f"task must be 'rec' or 'conv', got {task}"
        
        self.task = task
        self.tokenizer = tokenizer
        self.side_data = side_data
        self.hypergraph_dir = hypergraph_dir
        self.max_context_len = max_context_len
        self.max_response_len = max_response_len
        self.use_hypergraph = use_hypergraph
        
        # Load preprocessed data
        logger.info(f'Loading {task} data from {data_path}')
        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        
        logger.info(f'Loaded {len(self.samples)} samples for {task} task')
        
        # Cache for hypergraph data (避免重复加载)
        self.hypergraph_cache = {}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.task == 'rec':
            return self._get_rec_sample(sample)
        else:  # conv
            return self._get_conv_sample(sample)
    
    def _get_rec_sample(self, sample: Dict):
        """
        Get recommendation sample
        
        Returns:
            Dict with keys:
                - input_ids: Tensor [seq_len]
                - attention_mask: Tensor [seq_len]
                - item: int (item index)
                - graph_data: Data or None
        """
        # sample format: {'context_tokens': List[int], 'item': int, 'conv_id': int, 'turn_id': int}
        
        # 注意：context_tokens 是旧词表的ID，需要重新编码
        # 但在 notebook 中我们应该保存原始文本而不是 token IDs
        # 这里假设你已经修改了预处理，保存的是文本
        
        # 如果保存的是 token IDs (旧格式)，需要先转回文本
        # 这里我们提供一个临时方案：直接用旧 token IDs（需要你更新预处理）
        
        # TODO: 更新预处理以保存原始文本
        # 暂时使用占位符
        context_text = "[Placeholder: Update preprocessing to save text instead of token IDs]"
        
        # 使用 LLaVA tokenizer 编码
        encoding = self.tokenizer(
            context_text,
            max_length=self.max_context_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'item': torch.tensor(sample['item'], dtype=torch.long),
        }
        
        # Load hypergraph data if needed
        if self.use_hypergraph and self.hypergraph_dir:
            original_movie_id = self.side_data['id2item'][sample['item']]
            graph_data = self._load_hypergraph(original_movie_id)
            result['graph_data'] = graph_data
        else:
            result['graph_data'] = None
        
        return result
    
    def _get_conv_sample(self, sample: Dict):
        """
        Get conversation sample
        
        Returns:
            Dict with keys:
                - input_ids: Tensor [context_len]
                - response_ids: Tensor [response_len]
                - attention_mask: Tensor [context_len]
                - graph_data: Data or None
        """
        # sample format: {'context_tokens': List[int], 'response': List[int], ...}
        
        # TODO: 更新预处理以保存原始文本
        context_text = "[Placeholder: Update preprocessing]"
        response_text = "[Placeholder: Update preprocessing]"
        
        # Encode context
        context_encoding = self.tokenizer(
            context_text,
            max_length=self.max_context_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode response
        response_encoding = self.tokenizer(
            response_text,
            max_length=self.max_response_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': context_encoding['input_ids'].squeeze(0),
            'response': response_encoding['input_ids'].squeeze(0),
            'attention_mask': context_encoding['attention_mask'].squeeze(0),
            'graph_data': None  # 对话任务通常不需要超图
        }
        
        return result
    
    def _load_hypergraph(self, movie_id: int) -> Optional[Data]:
        """
        Load hypergraph data for a movie
        
        Args:
            movie_id: Original movie ID
        
        Returns:
            torch_geometric.Data or Dict[str, Data] or None
        """
        if movie_id in self.hypergraph_cache:
            return self.hypergraph_cache[movie_id]
        
        # Try to load multimodal embeddings
        features = {}
        
        for modality in ['img_emb', 'vdo_emb', 'audio_emb']:
            emb_path = os.path.join(self.hypergraph_dir, modality, f"{movie_id}.pt")
            if os.path.exists(emb_path):
                try:
                    feat = torch.load(emb_path, map_location='cpu')
                    features[modality.replace('_emb', '')] = feat
                except Exception as e:
                    logger.warning(f"Failed to load {emb_path}: {e}")
        
        if not features:
            self.hypergraph_cache[movie_id] = None
            return None
        
        # Build hypergraph Data
        graph_data = self._build_hypergraph_data(features)
        self.hypergraph_cache[movie_id] = graph_data
        
        return graph_data
    
    def _build_hypergraph_data(self, features: Dict[str, torch.Tensor]):
        """
        Build hypergraph from multimodal features
        
        Args:
            features: Dict[modality, tensor]
        
        Returns:
            Data or Dict[str, Data]
        """
        if len(features) == 1:
            # Single modality
            modality, feat = list(features.items())[0]
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            
            num_nodes = feat.size(0)
            
            # Create fully connected edges
            if num_nodes > 1:
                edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
                # Add reverse edges
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            else:
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            
            return Data(x=feat, edge_index=edge_index, num_nodes=num_nodes)
        
        else:
            # Multiple modalities
            multi_modal_data = {}
            
            for modality, feat in features.items():
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                
                num_nodes = feat.size(0)
                
                if num_nodes > 1:
                    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
                    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                else:
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                
                multi_modal_data[modality] = Data(
                    x=feat, 
                    edge_index=edge_index, 
                    num_nodes=num_nodes
                )
            
            return multi_modal_data


def collate_fn_rec(batch: List[Dict]) -> Dict:
    """
    Collate function for recommendation task
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Dict with batched tensors
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    items = torch.stack([item['item'] for item in batch])
    
    # Collect graph data
    graph_data = [item['graph_data'] for item in batch if item['graph_data'] is not None]
    if not graph_data:
        graph_data = None
    
    return {
        'context_tokens': input_ids,  # 使用 context_tokens 以兼容 CRSLab
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'item': items,
        'graph_data': graph_data
    }


def collate_fn_conv(batch: List[Dict]) -> Dict:
    """
    Collate function for conversation task
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Dict with batched tensors
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    response = torch.stack([item['response'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Collect graph data
    graph_data = [item['graph_data'] for item in batch if item['graph_data'] is not None]
    if not graph_data:
        graph_data = None
    
    return {
        'context_tokens': input_ids,  # 兼容 CRSLab
        'input_ids': input_ids,
        'response': response,
        'attention_mask': attention_mask,
        'graph_data': graph_data
    }


class HypergraphLlavaReDialDataLoader:
    """
    DataLoader wrapper for CRSLab compatibility
    
    Provides get_rec_data() and get_conv_data() methods
    """
    
    def __init__(
        self,
        train_rec_path: str,
        test_rec_path: str,
        train_conv_path: str,
        test_conv_path: str,
        tokenizer: AutoTokenizer,
        side_data: Dict,
        hypergraph_dir: Optional[str] = None,
        max_context_len: int = 512,
        max_response_len: int = 128,
        use_hypergraph: bool = True
    ):
        self.tokenizer = tokenizer
        self.side_data = side_data
        self.hypergraph_dir = hypergraph_dir
        self.max_context_len = max_context_len
        self.max_response_len = max_response_len
        self.use_hypergraph = use_hypergraph
        
        # Create datasets
        self.train_rec_dataset = HypergraphLlavaReDialDataset(
            train_rec_path, tokenizer, side_data, task='rec',
            hypergraph_dir=hypergraph_dir,
            max_context_len=max_context_len,
            use_hypergraph=use_hypergraph
        )
        
        self.test_rec_dataset = HypergraphLlavaReDialDataset(
            test_rec_path, tokenizer, side_data, task='rec',
            hypergraph_dir=hypergraph_dir,
            max_context_len=max_context_len,
            use_hypergraph=use_hypergraph
        )
        
        self.train_conv_dataset = HypergraphLlavaReDialDataset(
            train_conv_path, tokenizer, side_data, task='conv',
            hypergraph_dir=hypergraph_dir,
            max_context_len=max_context_len,
            max_response_len=max_response_len,
            use_hypergraph=False  # 对话任务通常不需要超图
        )
        
        self.test_conv_dataset = HypergraphLlavaReDialDataset(
            test_conv_path, tokenizer, side_data, task='conv',
            hypergraph_dir=hypergraph_dir,
            max_context_len=max_context_len,
            max_response_len=max_response_len,
            use_hypergraph=False
        )
    
    def get_rec_data(self, batch_size: int, shuffle: bool = True, split: str = 'train'):
        """
        Get recommendation data iterator (CRSLab compatible)
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            split: 'train' or 'test'
        
        Yields:
            Dict with batched tensors
        """
        dataset = self.train_rec_dataset if split == 'train' else self.test_rec_dataset
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_rec,
            num_workers=0,  # 设为0避免多进程问题
            pin_memory=True
        )
        
        for batch in dataloader:
            yield batch
    
    def get_conv_data(self, batch_size: int, shuffle: bool = True, split: str = 'train'):
        """
        Get conversation data iterator (CRSLab compatible)
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            split: 'train' or 'test'
        
        Yields:
            Dict with batched tensors
        """
        dataset = self.train_conv_dataset if split == 'train' else self.test_conv_dataset
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_conv,
            num_workers=0,
            pin_memory=True
        )
        
        for batch in dataloader:
            yield batch


def create_hypergraph_llava_dataloader(
    data_dir: str,
    tokenizer_path: str = "llava-hf/llava-1.5-7b-hf",
    hypergraph_dir: Optional[str] = None,
    max_context_len: int = 512,
    max_response_len: int = 128,
    use_hypergraph: bool = True
) -> Tuple[HypergraphLlavaReDialDataLoader, Dict, AutoTokenizer]:
    """
    Factory function to create dataloader
    
    Args:
        data_dir: Directory containing processed data
        tokenizer_path: Path to LLaVA tokenizer
        hypergraph_dir: Directory containing hypergraph embeddings
        max_context_len: Maximum context length
        max_response_len: Maximum response length
        use_hypergraph: Whether to use hypergraph data
    
    Returns:
        (dataloader, side_data, tokenizer)
    """
    # Load tokenizer
    logger.info(f'Loading tokenizer from {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    
    # Add special tokens if needed
    special_tokens = {
        'additional_special_tokens': ['<hg_start>', '<hg_end>', '<hg_patch>']
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        logger.info(f'Added {num_added} special tokens to tokenizer')
    
    # Load side_data
    side_data_path = os.path.join(data_dir, 'side_data.pkl')
    logger.info(f'Loading side_data from {side_data_path}')
    with open(side_data_path, 'rb') as f:
        side_data = pickle.load(f)
    
    # Create dataloader
    dataloader = HypergraphLlavaReDialDataLoader(
        train_rec_path=os.path.join(data_dir, 'train_rec.pkl'),
        test_rec_path=os.path.join(data_dir, 'test_rec.pkl'),
        train_conv_path=os.path.join(data_dir, 'train_conv.pkl'),
        test_conv_path=os.path.join(data_dir, 'test_conv.pkl'),
        tokenizer=tokenizer,
        side_data=side_data,
        hypergraph_dir=hypergraph_dir,
        max_context_len=max_context_len,
        max_response_len=max_response_len,
        use_hypergraph=use_hypergraph
    )
    
    logger.info('DataLoader created successfully')
    logger.info(f'  Train rec samples: {len(dataloader.train_rec_dataset)}')
    logger.info(f'  Test rec samples: {len(dataloader.test_rec_dataset)}')
    logger.info(f'  Train conv samples: {len(dataloader.train_conv_dataset)}')
    logger.info(f'  Test conv samples: {len(dataloader.test_conv_dataset)}')
    logger.info(f'  Vocabulary size: {len(tokenizer)}')
    logger.info(f'  Number of items: {side_data["n_entity"]}')
    
    return dataloader, side_data, tokenizer
