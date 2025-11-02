# from .HypergraphLlava4Recsys import ModalityAdaptor
from .HypergraphLlava4Recsys import ModalityAdaptor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict
from loguru import logger
from tqdm import tqdm
import sys
import os
import signal
import atexit
import traceback


# 全局变量记录当前状态
current_epoch = 0
current_batch = 0


def _signal_handler(signum, frame):
    """捕获系统信号（如强制终止）"""
    logger.critical(f"Received signal {signum}! Program is being terminated.")
    logger.critical(f"Last known state: Epoch {current_epoch}, Batch {current_batch}")
    sys.exit(1)


def _on_exit():
    """程序退出时的清理函数"""
    logger.warning(f"Program exiting. Last state: Epoch {current_epoch}, Batch {current_batch}")
    sys.stdout.flush()
    sys.stderr.flush()


# 注册信号处理器和退出处理器
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
try:
    signal.signal(signal.SIGBREAK, _signal_handler)  # Windows特有
except AttributeError:
    pass

atexit.register(_on_exit)

class ModalityAdaptorDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # 获取所有电影ID列表
        self.movie_ids = list(dataset.movie2ind.values())
        
    def __len__(self):
        return len(self.movie_ids)
    
    def __getitem__(self, idx):
        movie_id = self.movie_ids[idx]
        # 使用延迟加载方法获取 embeddings
        return {
            'txt': self.dataset.get_embedding(movie_id, 'txt'),
            'img': self.dataset.get_embedding(movie_id, 'img'),
            'vdo': self.dataset.get_embedding(movie_id, 'vdo'),
            'ado': self.dataset.get_embedding(movie_id, 'ado')
        }
        
        
 
class ModalityAdaptorDataLoader:
    def __init__(self, adaptor_dataset, batch_size=16):
        self.adaptor_dataset = adaptor_dataset
        self.batch_size = batch_size
    
    def __iter__(self):
        dataloader = DataLoader(
            self.adaptor_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # 关键：在 Windows 上使用单进程，避免多进程问题
            pin_memory=False  # 避免 CUDA pinned memory 问题
        )
        for batch in dataloader:
            yield batch
        

def pretrain_modality_adaptor(
    config,
    dataset, 
    device: torch.device, 
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 1e-4
):
    global current_epoch, current_batch
    
    # 添加文件日志
    log_file = os.path.join(os.getcwd(), "pretrain_crash_debug.log")
    file_handler_id = logger.add(log_file, level="DEBUG", mode="w")
    logger.info(f"=== Training session started ===")
    logger.info(f"Debug log: {log_file}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    # 固定随机种子
    import random, numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Seed: {seed}")
    
    logger.info("Initializing ModalityAdaptor model...")
    model = ModalityAdaptor(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    adaptor_dataset = ModalityAdaptorDataset(dataset)
    dataloader = ModalityAdaptorDataLoader(adaptor_dataset, batch_size=batch_size)
    
    logger.info(f"Starting ModalityAdaptor pre-training...")
    logger.info(f"Total samples: {len(adaptor_dataset)}, Batch size: {batch_size}")
    logger.info(f"Device: {device}")
    
    try:
        for epoch in range(num_epochs):
            current_epoch = epoch + 1
            current_batch = 0
            logger.info(f"="*50)
            logger.info(f"Epoch {current_epoch}/{num_epochs}")
            logger.info(f"="*50)
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {current_epoch}/{num_epochs}")):
                current_batch = batch_idx + 1
                
                if current_batch % 50 == 0:
                    logger.info(f"E{current_epoch} B{current_batch}: Processing...")
                    sys.stdout.flush()
                
                try:
                    projected_emb, alignment_loss = model(
                        batch,
                        return_alignment_loss=True,
                        device=device
                    )
                    
                    if torch.isnan(alignment_loss) or torch.isinf(alignment_loss):
                        logger.error(f"Invalid loss at E{current_epoch} B{current_batch}")
                        continue
                    
                    alignment_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                except Exception as e:
                    logger.error(f"Error at E{current_epoch} B{current_batch}: {type(e).__name__}: {e}")
                    logger.error(traceback.format_exc())
                    raise

            logger.info(f"Epoch {current_epoch} completed, Loss: {alignment_loss.item():.4f}")
        
        logger.info("="*50)
        logger.info("Training completed successfully!")
        logger.info("="*50)
        torch.save(model.state_dict(), config['mm_proj_weight_path'])
        logger.info(f"Model saved to {config['mm_proj_weight_path']}")
        
    except KeyboardInterrupt:
        logger.warning(f"Interrupted at E{current_epoch} B{current_batch}")
        raise
    except Exception as e:
        logger.critical(f"FATAL at E{current_epoch} B{current_batch}")
        logger.critical(f"{type(e).__name__}: {e}")
        logger.critical(traceback.format_exc())
        raise
    finally:
        logger.remove(file_handler_id)
        logger.info(f"Log saved: {log_file}")