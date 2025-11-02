"""
DataLoader for ModalityAdaptor Pre-training
用于多模态适配器预训练的数据加载器
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List
from loguru import logger


class ModalityPretrainDataLoader:
    """
    用于 ModalityAdaptor 预训练的 DataLoader
    封装 PyTorch DataLoader 并提供便捷接口
    """
    
    def __init__(
        self,
        train_dataset,
        valid_dataset,
        test_dataset,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        """
        Args:
            train_dataset: 训练数据集
            valid_dataset: 验证数据集
            test_dataset: 测试数据集
            batch_size: 批次大小
            num_workers: 数据加载线程数
            pin_memory: 是否使用固定内存（GPU训练时推荐True）
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 创建 DataLoader
        self.train_loader = self._create_dataloader(
            train_dataset, shuffle=True
        )
        self.valid_loader = self._create_dataloader(
            valid_dataset, shuffle=False
        )
        self.test_loader = self._create_dataloader(
            test_dataset, shuffle=False
        )
        
        logger.info(f"ModalityPretrainDataLoader initialized:")
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Valid batches: {len(self.valid_loader)}")
        logger.info(f"  Test batches: {len(self.test_loader)}")
        logger.info(f"  Batch size: {batch_size}")
    
    def _create_dataloader(self, dataset, shuffle: bool):
        """创建 PyTorch DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.pin_memory,
            drop_last=False  # 保留最后一个不完整的batch
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """
        批处理函数，将多个样本组合成一个batch
        
        Args:
            batch: List of samples from dataset
        
        Returns:
            dict: {
                'movie_ids': List[str],
                'txt_feat': tensor (batch_size, txt_dim),
                'img_feat': tensor (batch_size, img_dim),
                'vdo_feat': tensor (batch_size, vdo_dim),
                'ado_feat': tensor (batch_size, ado_dim)
            }
        """
        # 收集 movie_ids
        movie_ids = [item['movie_id'] for item in batch]
        
        # 堆叠各模态特征
        collated = {'movie_ids': movie_ids}
        
        # 获取所有特征键（除了 movie_id）
        feat_keys = [k for k in batch[0].keys() if k != 'movie_id']
        
        for key in feat_keys:
            # 堆叠成 (batch_size, feature_dim)
            collated[key] = torch.stack([item[key] for item in batch])
        
        return collated
    
    def get_train_loader(self):
        """返回训练数据加载器"""
        return self.train_loader
    
    def get_valid_loader(self):
        """返回验证数据加载器"""
        return self.valid_loader
    
    def get_test_loader(self):
        """返回测试数据加载器"""
        return self.test_loader


def get_modality_pretrain_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42
):
    """
    便捷函数：从主数据集创建预训练用的 DataLoader
    
    Args:
        dataset: 主数据集（包含 embeddings）
        batch_size: 批次大小
        num_workers: 数据加载线程数
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        ModalityPretrainDataLoader 实例
    """
    from crslab.data.dataset.modality_pretrain import create_pretrain_datasets
    
    # 创建数据集
    train_dataset, valid_dataset, test_dataset = create_pretrain_datasets(
        dataset, train_ratio, valid_ratio, seed
    )
    
    # 创建 DataLoader
    dataloader = ModalityPretrainDataLoader(
        train_dataset,
        valid_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return dataloader
