"""
Multi-modal Alignment for HypergraphLlava
基于GollaRec的思路，实现多模态对齐预训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

class MultiModalAlignmentLoss(nn.Module):
    """
    Multi-modal alignment loss inspired by GollaRec
    Aligns Text-Image-Audio-Video-Hypergraph features
    """
    
    def __init__(self, temperature: float = 0.07, hypergraph_weight: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.hypergraph_weight = hypergraph_weight
        
    def compute_contrastive_loss(self, features_a, features_b, labels=None):
        """
        计算对比学习损失 (类似GollaRec的ITC loss)
        
        Args:
            features_a: [batch_size, hidden_dim]
            features_b: [batch_size, hidden_dim] 
            labels: [batch_size] 正样本对的标签
        """
        features_a = F.normalize(features_a, dim=1)
        features_b = F.normalize(features_b, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.mm(features_a, features_b.t()) / self.temperature
        
        if labels is None:
            # 对角线为正样本对
            labels = torch.arange(features_a.size(0), device=features_a.device)
        
        # 计算InfoNCE损失
        loss_a2b = F.cross_entropy(similarity, labels)
        loss_b2a = F.cross_entropy(similarity.t(), labels)
        
        return (loss_a2b + loss_b2a) / 2
    
    def forward(self, modal_features: Dict[str, torch.Tensor], 
                hypergraph_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算多模态对齐损失
        
        Args:
            modal_features: {
                'text': [batch_size, hidden_dim],
                'image': [batch_size, hidden_dim], 
                'audio': [batch_size, hidden_dim],
                'video': [batch_size, hidden_dim]
            }
            hypergraph_features: [batch_size, hidden_dim] 超图特征
        
        Returns:
            losses: 各种对齐损失
        """
        losses = {}
        
        # 1. Text-Image 对齐 (类似GollaRec的ITC)
        if 'text' in modal_features and 'image' in modal_features:
            losses['text_image'] = self.compute_contrastive_loss(
                modal_features['text'], modal_features['image']
            )
        
        # 2. Text-Audio 对齐 (新增)
        if 'text' in modal_features and 'audio' in modal_features:
            losses['text_audio'] = self.compute_contrastive_loss(
                modal_features['text'], modal_features['audio']
            )
        
        # 3. Text-Video 对齐 (新增)
        if 'text' in modal_features and 'video' in modal_features:
            losses['text_video'] = self.compute_contrastive_loss(
                modal_features['text'], modal_features['video']
            )
        
        # 4. 各模态与超图对齐 (类似GollaRec的text-graph alignment)
        for modality, features in modal_features.items():
            losses[f'{modality}_hypergraph'] = self.compute_contrastive_loss(
                features, hypergraph_features
            )
        
        # 5. 总损失计算
        total_loss = 0
        for loss_name, loss_value in losses.items():
            if 'hypergraph' in loss_name:
                total_loss += self.hypergraph_weight * loss_value
            else:
                total_loss += loss_value
        
        losses['total'] = total_loss
        return losses


class HypergraphModalityEncoder(nn.Module):
    """
    多模态特征编码器，为每种模态生成统一维度的特征
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 各模态特征投影层
        self.text_projector = nn.Linear(config.hidden_size, config.hg_hiddens_size)
        self.image_projector = nn.Linear(config.vision_config.hidden_size, config.hg_hiddens_size)
        
        # Audio和Video编码器 (需要根据您的特征提取器调整)
        self.audio_projector = nn.Linear(768, config.hg_hiddens_size)  # 假设audio特征是768维
        self.video_projector = nn.Linear(1024, config.hg_hiddens_size)  # 假设video特征是1024维
        
        # 模态融合层
        self.modality_fusion = nn.MultiheadAttention(
            embed_dim=config.hg_hiddens_size,
            num_heads=8,
            dropout=0.1
        )
        
    def encode_modalities(self, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """
        编码各种模态特征
        
        Args:
            batch_data: {
                'text_features': [batch_size, seq_len, hidden_size],
                'image_features': [batch_size, patch_num, vision_hidden_size],
                'audio_features': [batch_size, audio_len, audio_dim],
                'video_features': [batch_size, frame_num, video_dim]
            }
        """
        modal_features = {}
        
        # 文本特征 (取CLS token或平均池化)
        if 'text_features' in batch_data:
            text_feat = batch_data['text_features'].mean(dim=1)  # [batch_size, hidden_size]
            modal_features['text'] = self.text_projector(text_feat)
        
        # 图像特征 (平均池化)
        if 'image_features' in batch_data:
            image_feat = batch_data['image_features'].mean(dim=1)  # [batch_size, vision_hidden_size]
            modal_features['image'] = self.image_projector(image_feat)
        
        # 音频特征 (时序平均)
        if 'audio_features' in batch_data:
            audio_feat = batch_data['audio_features'].mean(dim=1)  # [batch_size, audio_dim]
            modal_features['audio'] = self.audio_projector(audio_feat)
        
        # 视频特征 (帧平均)
        if 'video_features' in batch_data:
            video_feat = batch_data['video_features'].mean(dim=1)  # [batch_size, video_dim]
            modal_features['video'] = self.video_projector(video_feat)
        
        return modal_features
    
    def fuse_modalities(self, modal_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        使用注意力机制融合多模态特征
        """
        # 将所有模态特征堆叠
        features_list = list(modal_features.values())
        stacked_features = torch.stack(features_list, dim=1)  # [batch_size, num_modalities, hidden_size]
        
        # 通过多头注意力融合
        fused_features, _ = self.modality_fusion(
            stacked_features, stacked_features, stacked_features
        )
        
        # 平均池化得到最终表示
        return fused_features.mean(dim=1)  # [batch_size, hidden_size]


class HypergraphInstructionDataset(torch.utils.data.Dataset):
    """
    超图指令数据集，用于第二阶段的指令微调
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """
        加载超图指令数据
        数据格式示例:
        {
            "instruction": "Based on the user's interaction hypergraph, recommend suitable items.",
            "input": "User has interacted with items: [movie_1, movie_2, book_1]",
            "output": "I recommend: movie_3, book_2",
            "hypergraph_data": {...}  # 超图结构数据
        }
        """
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建GoT提示 (类似GollaRec的GoT prompting)
        got_prompt = self.build_got_prompt(item)
        
        # Tokenize
        tokenized = self.tokenizer(
            got_prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'hypergraph_data': item.get('hypergraph_data', None),
            'labels': tokenized['input_ids'].squeeze()  # 自回归训练
        }
    
    def build_got_prompt(self, item):
        """
        构建Graph-of-Thought提示 (超图版本)
        """
        instruction = item['instruction']
        user_input = item['input']
        expected_output = item['output']
        
        # GoT推理步骤 (针对超图的推理链)
        got_steps = [
            "Step 1: Analyze hypergraph structure - identify user interaction patterns",
            "Step 2: Extract multi-modal features - consider text, image, audio, video",
            "Step 3: Propagate through hyperedges - aggregate neighborhood information", 
            "Step 4: Generate recommendations - rank items based on hypergraph reasoning"
        ]
        
        prompt = f"""<|system|>
You are a hypergraph-aware recommendation assistant. Use the provided interaction hypergraph to make informed recommendations.

<|user|>
{instruction}

Input: {user_input}

<hg_start>{'<hg_patch>' * 50}<hg_end>

Reasoning steps:
{chr(10).join(got_steps)}

<|assistant|>
{expected_output}"""
        
        return prompt


# 使用示例
if __name__ == "__main__":
    # 初始化组件
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    # 模拟配置
    class Config:
        hidden_size = 4096
        hg_hiddens_size = 768
        vision_config = type('', (), {'hidden_size': 1024})()
    
    config = Config()
    
    # 初始化编码器和损失函数
    encoder = HypergraphModalityEncoder(config)
    alignment_loss = MultiModalAlignmentLoss()
    
    print("✅ 多模态对齐组件初始化完成")
    print("📝 可以开始预训练和指令微调了")