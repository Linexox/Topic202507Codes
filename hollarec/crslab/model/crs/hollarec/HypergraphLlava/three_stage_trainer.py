"""
HypergraphLlava4Recsys 三阶段训练策略
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import json
from tqdm import tqdm

class ThreeStageTrainer:
    """
    三阶段训练器
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def stage1_multimodal_alignment(self, dataloader, epochs=3):
        """
        Stage 1: 多模态对齐预训练
        目标: 让Text-Image-Audio-Video特征对齐
        训练: 只训练MultiModalAdapter
        冻结: HypergraphLlava保持冻结
        """
        print("🚀 Stage 1: 多模态对齐预训练")
        
        # 冻结HypergraphLlava，只训练MultiModalAdapter
        for param in self.model.hypergraph_llava.parameters():
            param.requires_grad = False
        for param in self.model.mm_adapter.parameters():
            param.requires_grad = True
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4
        )
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Stage 1 Epoch {epoch+1}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # 只计算多模态对齐损失
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    item_multimodal_data=batch['item_data'],
                    training_stage='stage1'
                )
                
                loss = outputs['alignment_loss']
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            print(f"Stage 1 Epoch {epoch+1} 完成，平均损失: {total_loss/len(dataloader):.4f}")
        
        # 保存Stage 1检查点
        torch.save(self.model.mm_adapter.state_dict(), 'stage1_mm_adapter.pth')
        print("✅ Stage 1 完成，多模态适配器已保存")
    
    def stage2_graph_text_alignment(self, dataloader, epochs=2):
        """
        Stage 2: 图-文本对齐
        目标: 让超图节点特征与物品多模态特征对齐
        训练: 只训练GraphTower和graph_text_projector
        冻结: MultiModalAdapter保持冻结
        """
        print("🎯 Stage 2: 图-文本对齐")
        
        # 冻结MultiModalAdapter，训练GraphTower
        for param in self.model.mm_adapter.parameters():
            param.requires_grad = False
        for param in self.model.hypergraph_llava.get_graph_tower().parameters():
            param.requires_grad = True
        for param in self.model.graph_text_projector.parameters():
            param.requires_grad = True
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=5e-5
        )
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Stage 2 Epoch {epoch+1}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # 计算图-文本对齐损失
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    item_multimodal_data=batch['item_data'],
                    hypergraph_data=batch['hypergraph_data'],
                    node_item_mapping=batch['node_item_mapping'],
                    training_stage='stage2'
                )
                
                loss = outputs['graph_text_loss']
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            print(f"Stage 2 Epoch {epoch+1} 完成，平均损失: {total_loss/len(dataloader):.4f}")
        
        # 保存Stage 2检查点
        torch.save({
            'graph_tower': self.model.hypergraph_llava.get_graph_tower().state_dict(),
            'graph_text_projector': self.model.graph_text_projector.state_dict()
        }, 'stage2_graph_alignment.pth')
        print("✅ Stage 2 完成，图-文本对齐已保存")
    
    def stage3_end_to_end_crs(self, dataloader, epochs=5):
        """
        Stage 3: 端到端CRS训练
        目标: 整个对话推荐系统的端到端优化
        训练: 所有参数都可训练（小学习率）
        任务: 对话生成 + 推荐准确性
        """
        print("🎉 Stage 3: 端到端CRS训练")
        
        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 使用更小的学习率
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Stage 3 Epoch {epoch+1}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # 端到端训练
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'], 
                    item_multimodal_data=batch['item_data'],
                    hypergraph_data=batch['hypergraph_data'],
                    labels=batch['labels'],  # 对话生成标签
                    training_stage='stage3'
                )
                
                # 综合损失：对话生成损失 + 推荐损失
                dialogue_loss = outputs.loss if hasattr(outputs, 'loss') else 0
                rec_loss = self.compute_recommendation_loss(outputs, batch)
                
                total_loss_val = dialogue_loss + 0.5 * rec_loss  # 加权组合
                total_loss_val.backward()
                optimizer.step()
                
                total_loss += total_loss_val.item()
                progress_bar.set_postfix({
                    'total': f"{total_loss_val.item():.4f}",
                    'dialogue': f"{dialogue_loss:.4f}" if isinstance(dialogue_loss, torch.Tensor) else "0.0000",
                    'rec': f"{rec_loss.item():.4f}"
                })
            
            print(f"Stage 3 Epoch {epoch+1} 完成，平均损失: {total_loss/len(dataloader):.4f}")
        
        # 保存最终模型
        torch.save(self.model.state_dict(), 'final_hollarec_model.pth')
        print("🎊 Stage 3 完成，HollaRec训练完成！")
    
    def compute_recommendation_loss(self, outputs, batch):
        """计算推荐损失"""
        # 这里需要根据您的推荐任务设计
        # 例如：ranking loss, pointwise loss, etc.
        if 'item_features' in outputs:
            # 简单示例：使用余弦相似度
            item_features = outputs['item_features']
            target_items = batch.get('target_items', None)
            
            if target_items is not None:
                # 计算相似度并用ranking loss
                similarities = torch.cosine_similarity(
                    item_features.unsqueeze(1), 
                    target_items.unsqueeze(0), 
                    dim=2
                )
                # 简化的ranking loss
                return -torch.mean(similarities)
            else:
                return torch.tensor(0.0, device=item_features.device)
        else:
            return torch.tensor(0.0, device=self.device)
    
    def full_training_pipeline(self, stage1_data, stage2_data, stage3_data):
        """
        完整的三阶段训练流水线
        """
        print("🚀 开始HollaRec三阶段训练")
        
        try:
            # Stage 1: 多模态对齐
            stage1_loader = DataLoader(stage1_data, batch_size=16, shuffle=True)
            self.stage1_multimodal_alignment(stage1_loader, epochs=3)
            
            # Stage 2: 图-文本对齐  
            stage2_loader = DataLoader(stage2_data, batch_size=8, shuffle=True)
            self.stage2_graph_text_alignment(stage2_loader, epochs=2)
            
            # Stage 3: 端到端CRS
            stage3_loader = DataLoader(stage3_data, batch_size=4, shuffle=True)
            self.stage3_end_to_end_crs(stage3_loader, epochs=5)
            
            print("🎉 HollaRec三阶段训练全部完成！")
            
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            raise


# 数据准备示例
class MultiModalDataset:
    """多模态数据集"""
    
    def __init__(self, stage='stage1'):
        self.stage = stage
    
    def __getitem__(self, idx):
        if self.stage == 'stage1':
            # Stage 1数据：多模态对齐数据
            return {
                'input_ids': torch.randint(0, 1000, (128,)),
                'attention_mask': torch.ones(128),
                'item_data': {
                    'text': torch.randn(1, 50),
                    'image': torch.randn(1, 3, 224, 224),
                    'audio': torch.randn(1, 16000),
                    'video': torch.randn(1, 8, 3, 224, 224)
                }
            }
        elif self.stage == 'stage2':
            # Stage 2数据：图-文本对齐数据
            return {
                'input_ids': torch.randint(0, 1000, (128,)),
                'attention_mask': torch.ones(128),
                'item_data': {
                    'text': torch.randn(10, 50),
                    'image': torch.randn(10, 3, 224, 224)
                },
                'hypergraph_data': [torch.randn(20, 768)],  # 简化的超图数据
                'node_item_mapping': torch.randint(0, 10, (20,))
            }
        else:  # stage3
            # Stage 3数据：完整CRS数据
            return {
                'input_ids': torch.randint(0, 1000, (128,)),
                'attention_mask': torch.ones(128),
                'labels': torch.randint(0, 1000, (128,)),
                'item_data': {
                    'text': torch.randn(10, 50),
                    'image': torch.randn(10, 3, 224, 224)
                },
                'hypergraph_data': [torch.randn(20, 768)],
                'target_items': torch.randn(5, 768)
            }
    
    def __len__(self):
        return 1000  # 示例数据量


if __name__ == "__main__":
    from hypergraph_llava_4_recsys import HypergraphLlava4Recsys, HypergraphLlava4RecsysConfig
    
    # 初始化配置和模型
    config = HypergraphLlava4RecsysConfig()
    model = HypergraphLlava4Recsys(config)
    
    # 初始化训练器
    trainer = ThreeStageTrainer(model, config)
    
    # 准备数据
    stage1_data = MultiModalDataset('stage1')
    stage2_data = MultiModalDataset('stage2') 
    stage3_data = MultiModalDataset('stage3')
    
    # 执行完整训练
    trainer.full_training_pipeline(stage1_data, stage2_data, stage3_data)