from unittest.mock import Base
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import json
from transformers import AutoModel, AutoTokenizer
from loguru import logger

from crslab.model.base import BaseModel
from .HypergraphLlava import HypergraphLlavaModel, HypergraphLlavaConfig

class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(MultiModalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive loss between two sets of features.

        Args:
            features_a (torch.Tensor): Tensor of shape (batch_size, feature_dim)
            features_b (torch.Tensor): Tensor of shape (batch_size, feature_dim)

        Returns:
            torch.Tensor: Contrastive loss value
        """
        batch_size = features_a.size(0)
        
        # Normalize the features
        features_a = F.normalize(features_a, p=2, dim=-1)
        features_b = F.normalize(features_b, p=2, dim=-1)

        pos_sim = self.cosine_similarity(features_a, features_b) 
        exp_pos_sim = torch.exp(pos_sim / self.temperature) # (batch_size,)
        
        neg_sim_matrix = torch.matmul(features_a, features_b.t()) # (batch_size, batch_size)
        exp_neg_sim = torch.exp(neg_sim_matrix / self.temperature).sum(dim=1) - exp_pos_sim # (batch_size,)
        
        exp_neg_sim = torch.clamp(exp_neg_sim, min=1e-8) 
        loss = -torch.sum(torch.log(exp_pos_sim / exp_neg_sim)) / batch_size
        return loss

class ModalityAdaptor(nn.Module):
    def __init__(self, config):
        super(ModalityAdaptor, self).__init__()
        self.config = config
        self.unified_dim = config.get('mm_unified_dim')
        self.hidden_size = config.get('mm_hidden_size')
        self.txt_dim = config.get('txt_dim')
        self.img_dim = config.get('img_dim')
        self.vdo_dim = config.get('vdo_dim')
        self.ado_dim = config.get('ado_dim')

        self.txt_proj = nn.Linear(self.txt_dim, self.unified_dim)
        self.img_proj = nn.Linear(self.img_dim, self.unified_dim)
        self.vdo_proj = nn.Linear(self.vdo_dim, self.unified_dim)
        self.ado_proj = nn.Linear(self.ado_dim, self.unified_dim)
        
        if config.get('load_mm_proj_weights', False):
            self._load_mm_proj_weights()
            self.freeze_projections()
        else:
            self.unfreeze_projections()
        
        self.alignment_loss_fn = MultiModalContrastiveLoss(temperature=config.get('mm_contrastive_temp', 0.07))
    
    def _load_mm_proj_weights(self):
        proj_weights_path = self.config.get('mm_proj_weight_path', None)
        try:
            state_dict = torch.load(proj_weights_path, map_location='cpu')
            self.txt_proj.load_state_dict(state_dict['txt_proj'])
            self.img_proj.load_state_dict(state_dict['img_proj'])
            self.vdo_proj.load_state_dict(state_dict['vdo_proj'])
            self.ado_proj.load_state_dict(state_dict['ado_proj'])
            logger.info(f"[ModalityAdaptor] Loaded pre-trained projection weights from {proj_weights_path}.")
        except Exception as e:
            logger.error(f"[ModalityAdaptor] Failed to load projection weights: {e}")
            raise e
        
    def forward(self, batch_data: Dict, return_alignment_loss=False, device='cpu'):
        # 优化：合并 device 和 dtype 转换为一次操作，减少内存复制
        # .to() 如果已经在目标 device 且类型正确，不会创建新副本
        txt_feat = batch_data['txt'].to(device=device, dtype=torch.float32, non_blocking=True)
        img_feat = batch_data['img'].to(device=device, dtype=torch.float32, non_blocking=True)
        vdo_feat = batch_data['vdo'].to(device=device, dtype=torch.float32, non_blocking=True)
        ado_feat = batch_data['ado'].to(device=device, dtype=torch.float32, non_blocking=True)
        
        # 投影
        projected_txt = self.txt_proj(txt_feat)
        projected_img = self.img_proj(img_feat)
        projected_vdo = self.vdo_proj(vdo_feat)
        projected_ado = self.ado_proj(ado_feat)
        
        # 优化：如果不需要单独访问，可以直接传递 tuple 而不是 dict，节省内存
        projected_emb = {
            'txt': projected_txt,
            'img': projected_img,
            'vdo': projected_vdo,
            'ado': projected_ado
        }
        
        alignment_loss = None
        if return_alignment_loss:
            alignment_loss = self.compute_alignment_loss(projected_emb)
        return projected_emb, alignment_loss
    
    def compute_alignment_loss(self, projected_emb: Dict) -> torch.Tensor:
        txt_feat = projected_emb['txt']
        img_feat = projected_emb['img']
        vdo_feat = projected_emb['vdo']
        ado_feat = projected_emb['ado']

        loss_txt_img = self.alignment_loss_fn(txt_feat, img_feat)
        loss_txt_vdo = self.alignment_loss_fn(txt_feat, vdo_feat)
        loss_txt_ado = self.alignment_loss_fn(txt_feat, ado_feat)
        loss_img_vdo = self.alignment_loss_fn(img_feat, vdo_feat)
        loss_img_ado = self.alignment_loss_fn(img_feat, ado_feat)
        loss_vdo_ado = self.alignment_loss_fn(vdo_feat, ado_feat)
        
        total_loss = (loss_txt_img + loss_txt_vdo + loss_txt_ado +
                      loss_img_vdo + loss_img_ado + loss_vdo_ado) / 6.0
        return total_loss
        
    def freeze_projections(self):
        for proj in [self.txt_proj, self.img_proj, self.vdo_proj, self.ado_proj]:
            for param in proj.parameters():
                param.requires_grad = False
        logger.info("[ModalityAdaptor] Projection layers frozen.")
        
    def unfreeze_projections(self):
        for proj in [self.txt_proj, self.img_proj, self.vdo_proj, self.ado_proj]:
            for param in proj.parameters():
                param.requires_grad = True
        logger.info("[ModalityAdaptor] Projection layers unfrozen.")

    def get_fusion_embedding(self, projected_emb: Dict, method='mean') -> torch.Tensor:
        """
        Fuse the projected embeddings from different modalities.

        Args:
            projected_emb (Dict): Dictionary containing projected embeddings for each modality.
        """
        if method == 'mean':
            fused_emb = (projected_emb['txt'] + projected_emb['img'] +
                         projected_emb['vdo'] + projected_emb['ado']) / 4.0
        else:
            raise NotImplementedError(f"Fusion method '{method}' not implemented.")
        return fused_emb

class HypergraphLlava4Recsys(BaseModel):
    # config_class = HypergraphLlavaConfig
    def __init__(self, opt, device, vocab):
        self.device = device
        self.gpu = opt.get("gpu", -1)
        assert self.dataset in ["redial"]
        #vocab
        self.pad_token_id = vocab['tok2id']['<pad>']
        self.start_token_id = vocab['tok2id']['<s>']
        self.end_token_id = vocab['tok2id']['</s>']
        self.vocab_size = vocab['vocab_size']
        self.token_emb_dim = opt.get('token_emb_dim', 4096)
        
        super().__init__(opt, device)
    
    def build_model(self, *args, **kwargs):
        self.modality_adaptor = ModalityAdaptor(self.config)
        pass

    def converse(self, batch, mode):
        pass

    def recommend(self, batch, mode):
        context_tokens = batch['context_tokens']
        context_movies = batch['context_movies']
        batch_size = len(context_tokens)
        
        pass

    def modality_finetune(self, batch, mode):
        pass

    def graph_finetune(self, batch, mode):
        pass

    def instruction_finetune(self, batch, mode):
        pass

    def forward(self, batch, mode, stage):
        if len(self.gpu)>=2:
            logger.error("Not support multi-gpu training for HypergraphLlava4Recsys model yet.")
            raise NotImplementedError
        if stage=='conv':
            return self.converse(batch, mode)
        if stage == 'rec':
            return self.recommend(batch, mode)
        pass