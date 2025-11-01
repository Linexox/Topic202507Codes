"""
Two-Stage Fine-tuning Pipeline for HypergraphLlava
基于GollaRec的两阶段微调方案：
Stage 1: Multi-modal Alignment Pre-training
Stage 2: Hypergraph Instruction Tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
import os
from tqdm import tqdm
import logging
from typing import Dict, List, Optional
import wandb

from HypergraphLlava import HypergraphLlavaConfig, HypergraphLlavaForCausalLM
from multimodal_alignment import MultiModalAlignmentLoss, HypergraphModalityEncoder, HypergraphInstructionDataset

class HypergraphLlavaTrainer:
    """
    HypergraphLlava两阶段训练器
    """
    
    def __init__(self, config_path: str, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        self.config = HypergraphLlavaConfig(**config_dict)
        
        # 初始化模型和tokenizer
        self.model = HypergraphLlavaForCausalLM(self.config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 添加特殊token
        self.add_hypergraph_tokens()
        
        # 初始化多模态组件
        self.modality_encoder = HypergraphModalityEncoder(self.config).to(self.device)
        self.alignment_loss_fn = MultiModalAlignmentLoss().to(self.device)
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def add_hypergraph_tokens(self):
        """添加超图相关的特殊token"""
        special_tokens = [
            "<hgraph>", "<hg_patch>", "<hg_start>", "<hg_end>"
        ]
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 设置token ids
        self.hg_token_id = self.tokenizer.convert_tokens_to_ids("<hgraph>")
        self.hg_patch_token_id = self.tokenizer.convert_tokens_to_ids("<hg_patch>")
        self.hg_start_token_id = self.tokenizer.convert_tokens_to_ids("<hg_start>")
        self.hg_end_token_id = self.tokenizer.convert_tokens_to_ids("<hg_end>")
        
    def stage1_multimodal_alignment(self, 
                                   data_path: str,
                                   epochs: int = 3,
                                   batch_size: int = 4,
                                   learning_rate: float = 1e-4,
                                   save_path: str = "./stage1_checkpoint"):
        """
        第一阶段：多模态对齐预训练
        目标：让模型理解文本-图像-音频-视频-超图之间的对应关系
        """
        self.logger.info("🚀 Starting Stage 1: Multi-modal Alignment Pre-training")\n        \n        # 准备数据\n        alignment_dataset = self.prepare_alignment_dataset(data_path)\n        dataloader = DataLoader(alignment_dataset, batch_size=batch_size, shuffle=True)\n        \n        # 优化器设置\n        optimizer = AdamW([\n            {'params': self.model.parameters(), 'lr': learning_rate},\n            {'params': self.modality_encoder.parameters(), 'lr': learning_rate * 2}\n        ])\n        \n        # 学习率调度\n        num_training_steps = len(dataloader) * epochs\n        scheduler = get_linear_schedule_with_warmup(\n            optimizer, \n            num_warmup_steps=num_training_steps // 10,\n            num_training_steps=num_training_steps\n        )\n        \n        # 训练循环\n        self.model.train()\n        for epoch in range(epochs):\n            total_loss = 0\n            progress_bar = tqdm(dataloader, desc=f"Stage 1 Epoch {epoch+1}/{epochs}")\n            \n            for batch in progress_bar:\n                optimizer.zero_grad()\n                \n                # 前向传播\n                losses = self.alignment_forward_pass(batch)\n                loss = losses['total']\n                \n                # 反向传播\n                loss.backward()\n                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)\n                optimizer.step()\n                scheduler.step()\n                \n                total_loss += loss.item()\n                progress_bar.set_postfix({\n                    'loss': f"{loss.item():.4f}",\n                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"\n                })\n                \n                # 记录到wandb\n                if hasattr(self, 'use_wandb') and self.use_wandb:\n                    wandb.log({\n                        'stage1/loss': loss.item(),\n                        'stage1/lr': scheduler.get_last_lr()[0],\n                        **{f'stage1/{k}': v.item() for k, v in losses.items() if k != 'total'}\n                    })\n            \n            avg_loss = total_loss / len(dataloader)\n            self.logger.info(f"Stage 1 Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")\n        \n        # 保存检查点\n        self.save_stage1_checkpoint(save_path)\n        self.logger.info(f"✅ Stage 1 completed. Checkpoint saved to {save_path}")\n    \n    def stage2_hypergraph_instruction_tuning(self,\n                                            instruction_data_path: str,\n                                            epochs: int = 5,\n                                            batch_size: int = 2,\n                                            learning_rate: float = 5e-5,\n                                            save_path: str = "./stage2_checkpoint"):\n        """\n        第二阶段：超图指令微调\n        目标：让模型能够基于超图结构进行推荐和对话\n        """\n        self.logger.info("🎯 Starting Stage 2: Hypergraph Instruction Tuning")\n        \n        # 准备指令数据\n        instruction_dataset = HypergraphInstructionDataset(\n            instruction_data_path, self.tokenizer\n        )\n        dataloader = DataLoader(instruction_dataset, batch_size=batch_size, shuffle=True)\n        \n        # 优化器设置 (较小的学习率)\n        optimizer = AdamW(self.model.parameters(), lr=learning_rate)\n        \n        # 学习率调度\n        num_training_steps = len(dataloader) * epochs\n        scheduler = get_linear_schedule_with_warmup(\n            optimizer,\n            num_warmup_steps=num_training_steps // 20,\n            num_training_steps=num_training_steps\n        )\n        \n        # 训练循环\n        self.model.train()\n        for epoch in range(epochs):\n            total_loss = 0\n            progress_bar = tqdm(dataloader, desc=f"Stage 2 Epoch {epoch+1}/{epochs}")\n            \n            for batch in progress_bar:\n                optimizer.zero_grad()\n                \n                # 前向传播\n                outputs = self.model(\n                    input_ids=batch['input_ids'].to(self.device),\n                    attention_mask=batch['attention_mask'].to(self.device),\n                    graph_data=batch.get('hypergraph_data', None),\n                    labels=batch['labels'].to(self.device)\n                )\n                \n                loss = outputs.loss\n                \n                # 反向传播\n                loss.backward()\n                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)\n                optimizer.step()\n                scheduler.step()\n                \n                total_loss += loss.item()\n                progress_bar.set_postfix({\n                    'loss': f"{loss.item():.4f}",\n                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"\n                })\n                \n                # 记录到wandb\n                if hasattr(self, 'use_wandb') and self.use_wandb:\n                    wandb.log({\n                        'stage2/loss': loss.item(),\n                        'stage2/lr': scheduler.get_last_lr()[0]\n                    })\n            \n            avg_loss = total_loss / len(dataloader)\n            self.logger.info(f"Stage 2 Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")\n        \n        # 保存最终模型\n        self.save_final_model(save_path)\n        self.logger.info(f"🎉 Stage 2 completed. Final model saved to {save_path}")\n    \n    def alignment_forward_pass(self, batch):\n        """\n        第一阶段的前向传播逻辑\n        """\n        # 编码各模态特征\n        modal_features = self.modality_encoder.encode_modalities(batch)\n        \n        # 获取超图特征\n        if 'hypergraph_data' in batch and batch['hypergraph_data'] is not None:\n            # 通过HGNN获取超图特征\n            hypergraph_output = self.model.model.get_graph_tower()(batch['hypergraph_data'])\n            hypergraph_features = self.model.model.graph_projector(hypergraph_output)\n        else:\n            # 创建dummy特征\n            hypergraph_features = torch.zeros(\n                len(modal_features['text']), self.config.hg_hiddens_size\n            ).to(self.device)\n        \n        # 计算对齐损失\n        alignment_losses = self.alignment_loss_fn(modal_features, hypergraph_features)\n        \n        return alignment_losses\n    \n    def prepare_alignment_dataset(self, data_path: str):\n        """\n        准备多模态对齐数据集\n        数据格式应包含：\n        - text_features: 文本特征\n        - image_features: 图像特征  \n        - audio_features: 音频特征\n        - video_features: 视频特征\n        - hypergraph_data: 超图数据\n        """\n        # 这里需要根据您的实际数据格式实现\n        # 示例实现\n        class AlignmentDataset(torch.utils.data.Dataset):\n            def __init__(self, data_path):\n                with open(data_path, 'r') as f:\n                    self.data = json.load(f)\n            \n            def __len__(self):\n                return len(self.data)\n            \n            def __getitem__(self, idx):\n                item = self.data[idx]\n                return {\n                    'text_features': torch.tensor(item['text_features']),\n                    'image_features': torch.tensor(item['image_features']),\n                    'audio_features': torch.tensor(item.get('audio_features', [])),\n                    'video_features': torch.tensor(item.get('video_features', [])),\n                    'hypergraph_data': item.get('hypergraph_data', None)\n                }\n        \n        return AlignmentDataset(data_path)\n    \n    def save_stage1_checkpoint(self, save_path: str):\n        """保存第一阶段检查点"""\n        os.makedirs(save_path, exist_ok=True)\n        \n        # 保存模型\n        torch.save(self.model.state_dict(), os.path.join(save_path, "model.pth"))\n        torch.save(self.modality_encoder.state_dict(), os.path.join(save_path, "modality_encoder.pth"))\n        \n        # 保存配置\n        with open(os.path.join(save_path, "config.json"), 'w') as f:\n            json.dump(self.config.to_dict(), f, indent=2)\n        \n        # 保存tokenizer\n        self.tokenizer.save_pretrained(save_path)\n    \n    def save_final_model(self, save_path: str):\n        """保存最终模型"""\n        os.makedirs(save_path, exist_ok=True)\n        \n        # 保存完整模型\n        self.model.save_pretrained(save_path)\n        self.tokenizer.save_pretrained(save_path)\n        \n        # 保存训练配置\n        training_config = {\n            "model_type": "HypergraphLlava",\n            "stage1_completed": True,\n            "stage2_completed": True,\n            "supports_modalities": ["text", "image", "audio", "video", "hypergraph"]\n        }\n        \n        with open(os.path.join(save_path, "training_config.json"), 'w') as f:\n            json.dump(training_config, f, indent=2)\n    \n    def load_stage1_checkpoint(self, checkpoint_path: str):\n        """加载第一阶段检查点"""\n        self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pth")))\n        self.modality_encoder.load_state_dict(torch.load(os.path.join(checkpoint_path, "modality_encoder.pth")))\n        self.logger.info(f"✅ Stage 1 checkpoint loaded from {checkpoint_path}")\n    \n    def full_pipeline(self,\n                      stage1_data_path: str,\n                      stage2_data_path: str, \n                      stage1_epochs: int = 3,\n                      stage2_epochs: int = 5,\n                      use_wandb: bool = False):\n        """\n        完整的两阶段训练流水线\n        """\n        if use_wandb:\n            wandb.init(project="hypergraph-llava", name="two-stage-training")\n            self.use_wandb = True\n        \n        try:\n            # 第一阶段：多模态对齐\n            self.stage1_multimodal_alignment(\n                data_path=stage1_data_path,\n                epochs=stage1_epochs\n            )\n            \n            # 第二阶段：指令微调\n            self.stage2_hypergraph_instruction_tuning(\n                instruction_data_path=stage2_data_path,\n                epochs=stage2_epochs\n            )\n            \n            self.logger.info("🎉 两阶段训练完成！")\n            \n        except Exception as e:\n            self.logger.error(f"训练过程中出现错误: {e}")\n            raise\n        \n        finally:\n            if use_wandb:\n                wandb.finish()


# 使用示例\nif __name__ == "__main__":\n    # 初始化训练器\n    trainer = HypergraphLlavaTrainer(\n        config_path="./config_example.json",\n        model_name="llava-hf/llava-1.5-7b-hf"\n    )\n    \n    # 执行完整的两阶段训练\n    trainer.full_pipeline(\n        stage1_data_path="./data/alignment_data.json",\n        stage2_data_path="./data/instruction_data.jsonl",\n        stage1_epochs=3,\n        stage2_epochs=5,\n        use_wandb=True\n    )