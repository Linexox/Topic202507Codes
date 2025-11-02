import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch_geometric.data import Data
from typing import Optional, List, Union, Dict

from crslab.model.base import BaseModel
from crslab.model.HypergraphLlava import (
    HypergraphLlavaConfig,
    HypergraphLlavaForCausalLM
)


class HypergraphLlavaCRSModel(BaseModel):
    """
    HypergraphLlava model for Conversational Recommendation System
    
    This model supports three main tasks:
    1. Recommendation: Predict items based on dialogue context and hypergraph
    2. Conversation: Generate natural language responses
    
    Args:
        opt: Configuration dictionary
        device: Device to run the model
        vocab: Vocabulary object
        side_data: Side information (e.g., item database, knowledge graph)
    """
    
    def __init__(self, opt, device, vocab, side_data):
        self.opt = opt
        self.device = device
        self.vocab = vocab
        self.side_data = side_data
        
        # Model configurations
        self.hidden_size = opt.get('hidden_size', 768)
        self.n_items = side_data['n_entity']  # Number of items to recommend
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        
        # Hypergraph configurations
        self.hg_hidden_size = opt.get('hg_hidden_size', 768)
        self.use_hypergraph = opt.get('use_hypergraph', True)
        
        # Task-specific configurations
        self.enable_recommendation = opt.get('enable_recommendation', True)
        self.enable_conversation = opt.get('enable_conversation', True)
        
        super().__init__(opt, device, vocab=vocab, side_data=side_data)
    
    def build_model(self):
        """Build the HypergraphLlava CRS model"""
        self._build_hypergraph_llava()
        
        if self.enable_recommendation:
            self._build_recommendation_head()
        
        if self.enable_conversation:
            self._build_conversation_head()
        
        logger.info('[Build HypergraphLlava CRS model]')
    
    def _build_hypergraph_llava(self):
        """Initialize HypergraphLlava backbone"""
        # Load pretrained LLaVA config
        llava_model_path = self.opt.get('llava_model_path', 'llava-hf/llava-1.5-7b-hf')
        
        # Create HypergraphLlava configuration
        config = HypergraphLlavaConfig.from_pretrained(llava_model_path)
        
        # Add hypergraph-specific configurations
        config.graph_tower = self.opt.get('graph_tower', 'HGNN')
        config.hg_hiddens_size = self.hg_hidden_size
        config.hg_num_layers = self.opt.get('hg_num_layers', 2)
        config.hg_dropout = self.opt.get('hg_dropout', 0.1)
        config.use_graph_proj = True
        
        # Initialize model
        self.hypergraph_llava = HypergraphLlavaForCausalLM(config)
        
        # Optionally load pretrained weights
        if self.opt.get('pretrained_model_path'):
            self._load_pretrained_weights()
        
        logger.debug('[Build HypergraphLlava backbone]')
    
    def _load_pretrained_weights(self):
        """Load pretrained HypergraphLlava weights"""
        pretrained_path = self.opt['pretrained_model_path']
        logger.info(f'Loading pretrained weights from {pretrained_path}')
        
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        
        # Load state dict with prefix handling
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.hypergraph_llava.load_state_dict(state_dict, strict=False)
        logger.info('[Loaded pretrained weights]')
    
    def _build_recommendation_head(self):
        """Build recommendation head for item prediction"""
        # Projection layer: hidden_size -> item embedding
        self.rec_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.opt.get('rec_dropout', 0.1)),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Item scoring layer
        self.item_scorer = nn.Linear(self.hidden_size, self.n_items)
        
        # Recommendation loss
        self.rec_loss_fn = nn.CrossEntropyLoss()
        
        logger.debug('[Build recommendation head]')
    
    def _build_conversation_head(self):
        """Build conversation head (already included in HypergraphLlava)"""
        # HypergraphLlava already has lm_head for conversation
        # We use it directly
        self.conv_loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)
        
        logger.debug('[Build conversation head]')
    
    def recommend(self, batch, mode):
        """
        Recommendation task: Predict items to recommend
        
        Args:
            batch: Dict containing:
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - graph_data: List of hypergraph Data objects (optional)
                - item_label: [batch_size] (ground truth item IDs)
            mode: 'train', 'valid', or 'test'
        
        Returns:
            loss (Tensor): Recommendation loss (if mode != 'test')
            rec_scores (Tensor): Item scores [batch_size, n_items]
        """
        input_ids = batch.get('context_tokens', batch.get('input_ids')).to(self.device)
        attention_mask = batch.get('attention_mask').to(self.device) if 'attention_mask' in batch else None
        graph_data = batch.get('graph_data', None)
        
        # Forward through HypergraphLlava to get contextual representations
        outputs = self.hypergraph_llava.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_data=graph_data,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract [CLS] or last token representation for recommendation
        # Option 1: Use last hidden state's first token
        last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Option 2: Pool over sequence
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            context_repr = sum_hidden / sum_mask  # [batch_size, hidden_size]
        else:
            # Simple average pooling
            context_repr = last_hidden.mean(dim=1)  # [batch_size, hidden_size]
        
        # Project to recommendation space
        rec_repr = self.rec_projector(context_repr)  # [batch_size, hidden_size]
        
        # Score all items
        rec_scores = self.item_scorer(rec_repr)  # [batch_size, n_items]
        
        # Compute loss if labels provided
        loss = None
        if mode != 'test' and 'item' in batch:
            item_labels = batch['item'].to(self.device)  # [batch_size]
            loss = self.rec_loss_fn(rec_scores, item_labels)
        
        if mode == 'test':
            return rec_scores
        else:
            return loss, rec_scores
    
    def converse(self, batch, mode):
        """
        Conversation task: Generate dialogue responses
        
        Args:
            batch: Dict containing:
                - input_ids: [batch_size, context_len]
                - attention_mask: [batch_size, context_len]
                - response: [batch_size, response_len] (labels)
                - graph_data: List of hypergraph Data objects (optional)
            mode: 'train', 'valid', or 'test'
        
        Returns:
            loss (Tensor): Language modeling loss (if mode != 'test')
            predictions (Tensor): Generated token IDs (if mode == 'test')
        """
        input_ids = batch.get('context_tokens', batch.get('input_ids')).to(self.device)
        attention_mask = batch.get('attention_mask').to(self.device) if 'attention_mask' in batch else None
        graph_data = batch.get('graph_data', None)
        
        if mode != 'test':
            # Training/validation: Teacher forcing
            response = batch['response'].to(self.device)  # [batch_size, response_len]
            
            # Concatenate context and response for autoregressive training
            # Format: [context] [response]
            full_input_ids = torch.cat([input_ids, response], dim=1)
            
            # Create labels: -100 for context (ignored), response tokens shifted
            labels = torch.full_like(full_input_ids, -100)
            labels[:, input_ids.size(1):] = response
            
            # Forward pass
            outputs = self.hypergraph_llava(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                labels=labels,
                graph_data=graph_data,
                return_dict=True
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Get predictions
            preds = torch.argmax(logits[:, input_ids.size(1)-1:-1, :], dim=-1)
            
            return loss, preds
        
        else:
            # Test: Autoregressive generation
            preds = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_data=graph_data,
                max_new_tokens=self.opt.get('max_gen_len', 50)
            )
            
            return preds
    
    def generate(self, input_ids, attention_mask=None, graph_data=None, max_new_tokens=50):
        """
        Generate response using greedy decoding
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            graph_data: List of hypergraph Data
            max_new_tokens: Maximum number of tokens to generate
        
        Returns:
            generated_ids: [batch_size, generated_len]
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Start with context
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.hypergraph_llava(
                input_ids=generated,
                attention_mask=attention_mask,
                graph_data=graph_data,
                return_dict=True
            )
            
            # Get next token logits
            next_token_logits = outputs['logits'][:, -1, :]  # [batch_size, vocab_size]
            
            # Greedy selection
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)
            
            # Stop if all sequences generated EOS
            if (next_tokens == self.end_token_idx).all():
                break
        
        # Return only generated part (exclude context)
        generated_only = generated[:, input_ids.size(1):]
        
        return generated_only
    
    def freeze_llava_backbone(self):
        """Freeze LLaVA parameters for efficient fine-tuning"""
        for param in self.hypergraph_llava.model.vision_tower.parameters():
            param.requires_grad = False
        for param in self.hypergraph_llava.model.language_model.parameters():
            param.requires_grad = False
        
        logger.info('[Frozen LLaVA backbone (vision + language)]')
    
    def unfreeze_graph_tower(self):
        """Unfreeze hypergraph tower for training"""
        for param in self.hypergraph_llava.model.graph_tower.parameters():
            param.requires_grad = True
        for param in self.hypergraph_llava.model.graph_projector.parameters():
            param.requires_grad = True
        
        logger.info('[Unfrozen graph tower and projector]')
    
    def forward(self, batch, mode, stage='rec'):
        """
        Unified forward pass
        
        Args:
            batch: Input batch
            mode: 'train', 'valid', or 'test'
            stage: 'rec' (recommendation) or 'conv' (conversation)
        
        Returns:
            Task-specific outputs
        """
        if stage == 'rec':
            return self.recommend(batch, mode)
        elif stage == 'conv':
            return self.converse(batch, mode)
        else:
            raise ValueError(f"Unknown stage: {stage}")


# Alias for compatibility
HypergraphLlavaCRS = HypergraphLlavaCRSModel
