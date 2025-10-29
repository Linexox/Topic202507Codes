# @Time   : 2025/10/26
# @Author : Your Name
# @Email  : your.email@example.com

r"""
HypergraphLlava System
=====================
Training and evaluation system for HypergraphLlava CRS model
"""

import os
import torch
from loguru import logger
from tqdm import tqdm

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class HypergraphLlavaSystem(BaseSystem):
    """
    System for training and evaluating HypergraphLlava CRS model
    
    This system handles:
    1. Two-stage training: Recommendation -> Conversation
    2. Joint evaluation on both tasks
    3. Hypergraph data preparation
    """
    
    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, 
                 vocab, side_data, restore_system=False, interact=False, 
                 debug=False, tensorboard=False):
        
        super().__init__(opt, train_dataloader, valid_dataloader, test_dataloader, 
                        vocab, side_data, restore_system, interact, debug, tensorboard)
        
        # Training configurations
        self.rec_epoch = self.opt.get('rec_epoch', 10)
        self.conv_epoch = self.opt.get('conv_epoch', 10)
        self.rec_batch_size = self.opt.get('rec_batch_size', 128)
        self.conv_batch_size = self.opt.get('conv_batch_size', 32)
        
        # Optimizer configurations
        self.rec_optim_opt = self.opt.get('rec', {})
        self.conv_optim_opt = self.opt.get('conv', {})
        
        # Training strategy
        self.freeze_llava_for_rec = self.opt.get('freeze_llava_for_rec', True)
        self.use_hypergraph = self.opt.get('use_hypergraph', True)
    
    def rec_evaluate(self, rec_predict, item_label):
        """
        Evaluate recommendation predictions
        
        Args:
            rec_predict: [batch_size, n_items] score tensor
            item_label: [batch_size] ground truth item IDs
        """
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        
        for rec_rank, label in zip(rec_ranks, item_label):
            item = self.item_ids[label]
            self.evaluator.rec_evaluate(rec_rank, item)
    
    def conv_evaluate(self, prediction, response):
        """
        Evaluate conversation generation
        
        Args:
            prediction: [batch_size, seq_len] predicted token IDs
            response: [batch_size, seq_len] ground truth token IDs
        """
        prediction = prediction.tolist()
        response = response.tolist()
        
        for pred, resp in zip(prediction, response):
            pred_str = ind2txt(pred, self.ind2tok)
            resp_str = ind2txt(resp, self.ind2tok)
            self.evaluator.gen_evaluate(pred_str, [resp_str])
    
    def step(self, batch, stage, mode):
        """
        Single training/evaluation step
        
        Args:
            batch: Input batch
            stage: 'rec' or 'conv'
            mode: 'train', 'valid', or 'test'
        """
        assert stage in ['rec', 'conv']
        assert mode in ['train', 'valid', 'test']
        
        # Prepare hypergraph data if available
        if self.use_hypergraph and 'hypergraph_data' in batch:
            batch['graph_data'] = self._prepare_graph_data(batch['hypergraph_data'])
        
        if stage == 'rec':
            # Recommendation task
            if mode != 'test':
                loss, rec_scores = self.model.recommend(batch, mode)
                if mode == 'train':
                    self.backward(loss)
                else:
                    self.evaluator.optim_metrics.add('rec_loss', AverageMetric(loss))
                
                # Evaluate predictions
                self.rec_evaluate(rec_scores, batch['item'])
            else:
                rec_scores = self.model.recommend(batch, mode)
                self.rec_evaluate(rec_scores, batch['item'])
        
        elif stage == 'conv':
            # Conversation task
            if mode != 'test':
                loss, preds = self.model.converse(batch, mode)
                if mode == 'train':
                    self.backward(loss)
                else:
                    self.evaluator.optim_metrics.add('gen_loss', AverageMetric(loss))
                    self.evaluator.gen_metrics.add('ppl', PPLMetric(loss))
                
                # Evaluate predictions
                self.conv_evaluate(preds, batch['response'])
            else:
                preds = self.model.converse(batch, mode)
                self.conv_evaluate(preds, batch['response'])
    
    def _prepare_graph_data(self, hypergraph_batch):
        """
        Prepare hypergraph data for the model
        
        Args:
            hypergraph_batch: List of hypergraph representations
        
        Returns:
            List of torch_geometric.data.Data objects
        """
        from torch_geometric.data import Data
        
        graph_data_list = []
        
        for hg in hypergraph_batch:
            if isinstance(hg, Data):
                # Already in correct format
                graph_data_list.append(hg.to(self.device))
            elif isinstance(hg, dict):
                # Dictionary format: {modality: features}
                # Convert to Data objects
                modality_graphs = {}
                for modality, features in hg.items():
                    # Assume features is dict with 'x', 'edge_index', 'edge_attr'
                    graph = Data(
                        x=features['x'].to(self.device),
                        edge_index=features['edge_index'].to(self.device),
                        edge_attr=features.get('edge_attr', None)
                    )
                    if graph.edge_attr is not None:
                        graph.edge_attr = graph.edge_attr.to(self.device)
                    modality_graphs[modality] = graph
                
                graph_data_list.append(modality_graphs)
            else:
                raise ValueError(f"Unsupported hypergraph format: {type(hg)}")
        
        return graph_data_list
    
    def train_recommender(self):
        """Train recommendation task"""
        logger.info('[Start training recommendation]')
        
        # Freeze LLaVA backbone if specified
        if self.freeze_llava_for_rec:
            self.model.freeze_llava_backbone()
            self.model.unfreeze_graph_tower()
        
        # Initialize optimizer
        self.init_optim(self.rec_optim_opt, self.model.parameters())
        
        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {epoch}]')
            
            # Training
            logger.info('[Train]')
            self.model.train()
            pbar = tqdm(
                self.train_dataloader.get_rec_data(batch_size=self.rec_batch_size, shuffle=True),
                desc=f'Epoch {epoch}'
            )
            for batch in pbar:
                self.step(batch, stage='rec', mode='train')
                pbar.set_postfix({'loss': f'{self.evaluator.optim_metrics.get("rec_loss", 0):.4f}'})
            
            self.evaluator.report(epoch=epoch, mode='train')
            
            # Validation
            logger.info('[Valid]')
            self.model.eval()
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(batch_size=self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='valid')
                
                self.evaluator.report(epoch=epoch, mode='valid')
                
                # Early stopping
                metric = self.evaluator.optim_metrics.get('rec_loss', float('inf'))
                if self.early_stop(metric):
                    logger.info('[Early stopping triggered for recommendation]')
                    break
        
        # Test
        logger.info('[Test]')
        self.model.eval()
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(batch_size=self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            
            self.evaluator.report(mode='test')
        
        # Save best model
        self.save_model()
        logger.info('[Recommendation training finished]')
    
    def train_conversation(self):
        """Train conversation task"""
        logger.info('[Start training conversation]')
        
        # Unfreeze all parameters for conversation fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Initialize optimizer
        self.init_optim(self.conv_optim_opt, self.model.parameters())
        
        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {epoch}]')
            
            # Training
            logger.info('[Train]')
            self.model.train()
            pbar = tqdm(
                self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=True),
                desc=f'Epoch {epoch}'
            )
            for batch in pbar:
                self.step(batch, stage='conv', mode='train')
                pbar.set_postfix({'loss': f'{self.evaluator.optim_metrics.get("gen_loss", 0):.4f}'})
            
            self.evaluator.report(epoch=epoch, mode='train')
            
            # Validation
            logger.info('[Valid]')
            self.model.eval()
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='valid')
                
                self.evaluator.report(epoch=epoch, mode='valid')
                
                # Early stopping
                metric = self.evaluator.optim_metrics.get('gen_loss', float('inf'))
                if self.early_stop(metric):
                    logger.info('[Early stopping triggered for conversation]')
                    break
        
        # Test
        logger.info('[Test]')
        self.model.eval()
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            
            self.evaluator.report(mode='test')
        
        # Save best model
        self.save_model()
        logger.info('[Conversation training finished]')
    
    def fit(self):
        """
        Main training loop: Two-stage training
        Stage 1: Train recommendation with frozen LLaVA
        Stage 2: Fine-tune conversation with all parameters
        """
        if self.opt.get('train_recommendation', True):
            self.train_recommender()
        
        if self.opt.get('train_conversation', True):
            self.train_conversation()
    
    def interact(self):
        """Interactive mode for testing"""
        logger.info('[Interactive mode]')
        self.model.eval()
        
        print("Enter 'quit' to exit")
        while True:
            # Get user input
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            
            # Tokenize input
            input_ids = self.tokenizer.encode(user_input, return_tensors='pt').to(self.device)
            
            # Prepare batch (no hypergraph for simple interaction)
            batch = {
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids),
                'graph_data': None
            }
            
            # Generate response
            with torch.no_grad():
                response_ids = self.model.converse(batch, mode='test')
            
            # Decode response
            response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
            print(f"Bot: {response_text}")
    
    def save_model(self):
        """Save model checkpoint"""
        save_dir = self.opt.get('save_dir', './save')
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, 'hypergraph_llava_crs_best.pth')
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch_i,
        }, save_path)
        
        logger.info(f'[Model saved to {save_path}]')
    
    def load_model(self, model_path):
        """Load model checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'[Model loaded from {model_path}]')
