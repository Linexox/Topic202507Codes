"""
HypergraphLlava4Recsys: 为推荐系统设计的超图多模态大语言模型
包含两个核心组件：
1. MultiModalAdapter: 多模态对齐适配器
2. HypergraphLlava: 超图感知的MLLM

训练策略：
Stage 1: 预训练MultiModalAdapter完成多模态对齐
Stage 2: 冻结Adapter，训练GraphTower完成图-文本对齐  
Stage 3: 端到端微调整个CRS模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import json
from transformers import AutoModel, AutoTokenizer

from .HypergraphLlava import HypergraphLlavaModel, HypergraphLlavaConfig
from .multimodal_alignment import MultiModalAlignmentLoss


class MultiModalAdapter(nn.Module):
    """
    多模态适配器：将不同模态特征对齐到统一空间
    负责Text-Image-Audio-Video的对齐
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 各模态编码器
        self.text_encoder = self._build_text_encoder()
        self.image_encoder = self._build_image_encoder() 
        self.audio_encoder = self._build_audio_encoder()
        self.video_encoder = self._build_video_encoder()
        
        # 统一投影层 - 都投影到同一维度
        self.unified_dim = config.mm_hidden_size  # 例如768
        self.text_projector = nn.Linear(config.text_hidden_size, self.unified_dim)
        self.image_projector = nn.Linear(config.image_hidden_size, self.unified_dim)
        self.audio_projector = nn.Linear(config.audio_hidden_size, self.unified_dim)
        self.video_projector = nn.Linear(config.video_hidden_size, self.unified_dim)
        
        # 模态融合层
        self.modality_fusion = nn.MultiheadAttention(
            embed_dim=self.unified_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 对齐损失函数
        self.alignment_loss = MultiModalAlignmentLoss()
    
    def _build_text_encoder(self):
        """构建文本编码器 - 使用BERT或RoBERTa"""
        return AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    def _build_image_encoder(self):
        """构建图像编码器 - 使用CLIP或DINOv2"""
        return AutoModel.from_pretrained('openai/clip-vit-base-patch32')
    
    def _build_audio_encoder(self):
        """构建音频编码器 - 使用Wav2Vec2"""
        return AutoModel.from_pretrained('facebook/wav2vec2-base-960h')
    
    def _build_video_encoder(self):
        """构建视频编码器 - 使用VideoMAE"""
        return AutoModel.from_pretrained('MCG-NJU/videomae-base')
    
    def encode_single_modality(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        编码单个模态的特征
        
        Args:
            data: {
                'text': [batch_size, seq_len],
                'image': [batch_size, 3, H, W], 
                'audio': [batch_size, audio_len],
                'video': [batch_size, frames, 3, H, W]
            }
        
        Returns:
            modal_features: {
                'text': [batch_size, unified_dim],
                'image': [batch_size, unified_dim],
                'audio': [batch_size, unified_dim], 
                'video': [batch_size, unified_dim]
            }
        """
        modal_features = {}
        
        # 文本编码
        if 'text' in data:
            text_output = self.text_encoder(data['text'])
            text_feat = text_output.last_hidden_state.mean(dim=1)  # [batch_size, text_hidden]
            modal_features['text'] = self.text_projector(text_feat)
        
        # 图像编码
        if 'image' in data:
            image_output = self.image_encoder(data['image'])
            image_feat = image_output.last_hidden_state.mean(dim=1)  # [batch_size, image_hidden]
            modal_features['image'] = self.image_projector(image_feat)
        
        # 音频编码
        if 'audio' in data:
            audio_output = self.audio_encoder(data['audio'])
            audio_feat = audio_output.last_hidden_state.mean(dim=1)  # [batch_size, audio_hidden]
            modal_features['audio'] = self.audio_projector(audio_feat)
        
        # 视频编码
        if 'video' in data:
            # 视频处理：[batch_size, frames, 3, H, W] -> [batch_size*frames, 3, H, W]
            batch_size, frames = data['video'].shape[:2]
            video_reshaped = data['video'].view(-1, *data['video'].shape[2:])
            video_output = self.video_encoder(video_reshaped)
            video_feat = video_output.last_hidden_state.mean(dim=1)  # [batch_size*frames, video_hidden]
            video_feat = video_feat.view(batch_size, frames, -1).mean(dim=1)  # [batch_size, video_hidden]
            modal_features['video'] = self.video_projector(video_feat)
        
        return modal_features
    
    def fuse_modalities(self, modal_features: Dict[str, torch.Tensor], 
                       fusion_strategy: str = 'attention') -> torch.Tensor:
        """
        融合多模态特征
        
        Args:
            modal_features: 各模态特征字典
            fusion_strategy: 融合策略 ['attention', 'average', 'concat']
        
        Returns:
            fused_features: [batch_size, unified_dim] 融合后的特征
        """
        if fusion_strategy == 'average':
            # 简单平均
            features_list = list(modal_features.values())
            return torch.stack(features_list, dim=1).mean(dim=1)
        
        elif fusion_strategy == 'attention':
            # 注意力融合
            features_list = list(modal_features.values())
            stacked_features = torch.stack(features_list, dim=1)  # [batch_size, num_modalities, unified_dim]
            
            fused_features, _ = self.modality_fusion(
                stacked_features, stacked_features, stacked_features
            )
            return fused_features.mean(dim=1)  # [batch_size, unified_dim]
        
        elif fusion_strategy == 'concat':
            # 拼接后降维
            features_list = list(modal_features.values())
            concat_features = torch.cat(features_list, dim=1)  # [batch_size, unified_dim * num_modalities]
            return nn.Linear(concat_features.shape[1], self.unified_dim)(concat_features)
        
        else:
            raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")
    
    def forward(self, batch_data: Dict, return_alignment_loss: bool = False):
        """
        前向传播
        
        Args:
            batch_data: 批次数据
            return_alignment_loss: 是否返回对齐损失（训练Stage1时使用）
        
        Returns:
            如果return_alignment_loss=True，返回(fused_features, alignment_loss)
            否则返回fused_features
        """
        # 编码各模态
        modal_features = self.encode_single_modality(batch_data)
        
        # 融合模态特征
        fused_features = self.fuse_modalities(modal_features, fusion_strategy='attention')
        
        if return_alignment_loss:
            # 计算对齐损失（Stage1训练时使用）
            alignment_loss = self.alignment_loss(modal_features, fused_features)
            return fused_features, alignment_loss
        else:
            return fused_features


class HypergraphLlava4Recsys(nn.Module):
    """
    为推荐系统设计的超图多模态大语言模型
    """
    
    def __init__(self, config: HypergraphLlavaConfig):
        super().__init__()
        self.config = config
        
        # 组件1: 多模态适配器
        self.mm_adapter = MultiModalAdapter(config)
        
        # 组件2: 超图MLLM
        self.hypergraph_llava = HypergraphLlavaModel(config)
        
        # 图-文本对齐投影层
        self.graph_text_projector = nn.Linear(
            config.hg_hiddens_size, 
            config.mm_hidden_size  # 与多模态特征维度一致
        )
        
        # 对比学习损失（用于图-文本对齐）
        self.graph_text_alignment_loss = nn.CrossEntropyLoss()
        
    def get_item_multimodal_features(self, item_data: Dict, 
                                   modality_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        获取物品的多模态特征表示
        
        Args:
            item_data: 物品的多模态数据
            modality_weights: 各模态权重 {'text': 0.3, 'image': 0.4, 'audio': 0.2, 'video': 0.1}
        
        Returns:
            item_features: [num_items, mm_hidden_size] 物品特征
        """
        # 通过多模态适配器编码
        modal_features = self.mm_adapter.encode_single_modality(item_data)
        
        if modality_weights is None:
            # 默认策略：注意力融合
            item_features = self.mm_adapter.fuse_modalities(modal_features, 'attention')
        else:
            # 加权融合策略
            weighted_features = []
            for modality, features in modal_features.items():
                weight = modality_weights.get(modality, 0.25)  # 默认平均权重
                weighted_features.append(weight * features)
            item_features = torch.sum(torch.stack(weighted_features), dim=0)
        
        return item_features
    
    def compute_graph_text_alignment_loss(self, 
                                        hypergraph_node_features: torch.Tensor,
                                        item_multimodal_features: torch.Tensor,
                                        node_item_mapping: torch.LongTensor) -> torch.Tensor:
        """
        计算图-文本对齐损失（类似GollaRec的text-graph alignment）
        
        Args:
            hypergraph_node_features: [num_nodes, hg_hidden_size] 超图节点特征
            item_multimodal_features: [num_items, mm_hidden_size] 物品多模态特征  
            node_item_mapping: [num_nodes] 节点到物品的映射
        
        Returns:
            alignment_loss: 对齐损失
        """
        # 将超图节点特征投影到多模态空间
        projected_node_features = self.graph_text_projector(hypergraph_node_features)
        
        # 根据映射关系获取对应的物品特征
        mapped_item_features = item_multimodal_features[node_item_mapping]
        
        # 计算相似度矩阵
        projected_node_features = F.normalize(projected_node_features, dim=1)
        mapped_item_features = F.normalize(mapped_item_features, dim=1)
        
        similarity_matrix = torch.mm(projected_node_features, mapped_item_features.t()) / 0.07  # temperature=0.07
        
        # 对比学习损失
        labels = torch.arange(projected_node_features.size(0), device=projected_node_features.device)
        loss = self.graph_text_alignment_loss(similarity_matrix, labels)
        
        return loss
    
    def forward(self, 
                input_ids: torch.LongTensor,
                attention_mask: torch.Tensor,
                item_multimodal_data: Dict,
                hypergraph_data: Optional[List] = None,
                training_stage: str = 'stage3',
                **kwargs) -> Dict:
        """
        前向传播
        
        Args:
            input_ids: 输入token序列
            attention_mask: 注意力掩码
            item_multimodal_data: 物品多模态数据
            hypergraph_data: 超图数据
            training_stage: 训练阶段 ['stage1', 'stage2', 'stage3']
        """
        outputs = {}
        
        if training_stage == 'stage1':
            # Stage 1: 只训练多模态适配器
            item_features, alignment_loss = self.mm_adapter(
                item_multimodal_data, return_alignment_loss=True
            )
            outputs['multimodal_features'] = item_features
            outputs['alignment_loss'] = alignment_loss
            
        elif training_stage == 'stage2':
            # Stage 2: 冻结适配器，训练图-文本对齐
            with torch.no_grad():
                item_features = self.mm_adapter(item_multimodal_data)
            
            # 通过超图模型获取节点特征
            hypergraph_outputs = self.hypergraph_llava(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_data=hypergraph_data
            )
            
            # 计算图-文本对齐损失
            if hypergraph_data is not None:
                # 这里需要根据您的数据格式调整
                node_features = hypergraph_outputs.last_hidden_state  # 或其他合适的特征
                node_item_mapping = kwargs.get('node_item_mapping')  # 需要提供映射关系
                
                graph_text_loss = self.compute_graph_text_alignment_loss(
                    node_features, item_features, node_item_mapping
                )
                outputs['graph_text_loss'] = graph_text_loss
            
        else:  # stage3
            # Stage 3: 端到端训练整个模型
            item_features = self.mm_adapter(item_multimodal_data)
            
            hypergraph_outputs = self.hypergraph_llava(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_data=hypergraph_data,
                **kwargs
            )
            
            outputs.update(hypergraph_outputs)
            outputs['item_features'] = item_features
        
        return outputs


# 配置类扩展
class HypergraphLlava4RecsysConfig(HypergraphLlavaConfig):
    """扩展配置，添加多模态适配器相关参数"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 多模态适配器配置
        self.mm_hidden_size = kwargs.get('mm_hidden_size', 768)
        self.text_hidden_size = kwargs.get('text_hidden_size', 384)
        self.image_hidden_size = kwargs.get('image_hidden_size', 512)
        self.audio_hidden_size = kwargs.get('audio_hidden_size', 768)
        self.video_hidden_size = kwargs.get('video_hidden_size', 768)
        
        # 训练策略配置
        self.stage1_epochs = kwargs.get('stage1_epochs', 3)
        self.stage2_epochs = kwargs.get('stage2_epochs', 2) 
        self.stage3_epochs = kwargs.get('stage3_epochs', 5)


if __name__ == "__main__":
    # 使用示例
    config = HypergraphLlava4RecsysConfig(
        vocab_size=32000,
        hidden_size=4096,
        mm_hidden_size=768,
        hg_hiddens_size=768
    )
    
    model = HypergraphLlava4Recsys(config)
    print("✅ HypergraphLlava4Recsys 初始化完成")
    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")