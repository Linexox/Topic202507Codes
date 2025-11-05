# from .HypergraphLlava4Recsys import ModalityAdaptor
from .HypergraphLlava4Recsys import ModalityAdaptor

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict
from loguru import logger
from tqdm import tqdm

from crslab.config import PRETRAIN_PATH

current_epoch = 0
current_batch = 0

class ModalityAdaptorDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.movie_ids = self.filter(list(dataset.movie2ind.values()))
        # self.movie_ids = list(dataset.movie2ind.values())
        # print(type(self.movie_ids[0]))

    def filter(self, raw_movie_ids):
        movie_ids = [
            mid for mid in raw_movie_ids
            if self.dataset.get_embedding(mid, 'txt', return_zero_if_missing=False) is not None and
               self.dataset.get_embedding(mid, 'img', return_zero_if_missing=False) is not None and
               self.dataset.get_embedding(mid, 'vdo', return_zero_if_missing=False) is not None and
               self.dataset.get_embedding(mid, 'ado', return_zero_if_missing=False) is not None
        ]
        return movie_ids
    
    def __len__(self):
        return len(self.movie_ids)
    
    def __getitem__(self, idx):
        movie_id = self.movie_ids[idx]
        # print(idx, movie_id)
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
    
    def __len__(self):
        return (len(self.adaptor_dataset) + self.batch_size - 1) // self.batch_size
        

def pretrain_modality_adaptor(
    config,
    dataset, 
    device: torch.device, 
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4
):
    global current_epoch, current_batch
    
    logger.info("Initializing ModalityAdaptor model...")
    model = ModalityAdaptor(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    adaptor_dataset = ModalityAdaptorDataset(dataset)
    dataloader = ModalityAdaptorDataLoader(adaptor_dataset, batch_size=batch_size)
    
    logger.info(f"Starting ModalityAdaptor pre-training...")
    logger.info(f"Total samples: {len(adaptor_dataset)}, Batch size: {batch_size}")
    logger.info(f"Device: {device}")

    # logger.info(f"Check dataset sample modalities:")
    # sample = adaptor_dataset[0]
    # for modality in ['txt', 'img', 'vdo', 'ado']:
    #     logger.info(f"  {modality} : {sample[modality]}")

    
    for epoch in range(num_epochs):
        current_epoch = epoch + 1
        current_batch = 0
        logger.info(f"="*50)
        logger.info(f"Epoch {current_epoch}/{num_epochs}")
        logger.info(f"="*50)
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {current_epoch}/{num_epochs}"):
            projected_emb, alignment_loss = model(
                batch,
                return_alignment_loss=True,
                device=device
            )
            
            if torch.isnan(alignment_loss) or torch.isinf(alignment_loss):
                logger.error(f"Invalid loss at E{current_epoch} B{current_batch}")
                continue
            epoch_loss += alignment_loss.item()
            alignment_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if current_batch % 50 == 0:
                logger.info(f"E{current_epoch} B{current_batch}: Loss: {alignment_loss.item():.4f}")
            current_batch += 1

        logger.info(f"Epoch {current_epoch} completed, Loss: {epoch_loss/len(dataloader):.4f}")
    
    logger.info("="*50)
    logger.info("Training completed successfully!")
    logger.info("="*50)
    # torch.save(model.state_dict(), config['mm_proj_weight_path'])
    torch.save({
        'txt_proj': model.txt_proj.state_dict(),
        'img_proj': model.img_proj.state_dict(),
        'vdo_proj': model.vdo_proj.state_dict(),
        'ado_proj': model.ado_proj.state_dict()
    }, os.path.join(PRETRAIN_PATH, 'modality_adaptor_proj_weights.pth'))
    logger.info(f"Model saved to {os.path.join(PRETRAIN_PATH, 'modality_adaptor_proj_weights.pth')}")