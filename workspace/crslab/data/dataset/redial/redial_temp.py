# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2021/1/3, 2020/12/19, 2025/10/27
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou, HypergraphLlava Project
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail

r"""
ReDial (HypergraphLlava Version)
=================================
References:
    Li, Raymond, et al. `"Towards deep conversational recommendations."`_ in NeurIPS 2018.

.. _`"Towards deep conversational recommendations."`:
   https://papers.nips.cc/paper/2018/hash/800de15c79c8d840f4e78d3af937d4d4-Abstract.html

NOTE: This version is adapted for HypergraphLlava CRS model:
    - Uses transformers AutoTokenizer (LLaVA) instead of custom vocab
    - Loads raw text data preprocessed from notebook
    - No resources.py dependency
    - Adds hypergraph special tokens
"""

import os
import pickle
from typing import Dict, List, Optional  # Added Optional

import torch
from transformers import AutoTokenizer
from loguru import logger
from tqdm import tqdm

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset


class ReDialDataset(BaseDataset):
    """
    HypergraphLlava-compatible ReDial dataset
    
    Attributes:
        train_data: List[Dict] - training samples
        valid_data: List[Dict] - validation samples  
        test_data: List[Dict] - test samples
        side_data: Dict - item mappings and metadata
        vocab: Dict - tokenizer info and special token IDs
        tokenizer: AutoTokenizer - LLaVA tokenizer instance
    
    Notes:
        Inherits from BaseDataset but overrides __init__ to skip resource download
        and use LLaVA tokenizer. The following attributes MUST be set for CRSLab compatibility:
        - self.train_data
        - self.valid_data
        - self.test_data
        - self.side_data
        - self.vocab
    """
    
    def __init__(
        self, 
        opt, 
        tokenize: str = 'llava',  # Changed to llava_model_path if needed
        restore: bool = False, 
        save: bool = False,
        tokenizer: Optional[AutoTokenizer] = None  # NEW: Accept pre-initialized tokenizer
    ):
        """
        Initialize dataset with LLaVA tokenizer
        
        Args:
            opt (Config or dict): Configuration (must contain 'llava_model_path')
            tokenize (str): Tokenizer type (ignored, kept for interface compatibility)
            restore (bool): Whether to restore saved processed data
            save (bool): Whether to save processed data
            tokenizer (AutoTokenizer, optional): Pre-initialized tokenizer with special tokens.
                If None, will load fresh tokenizer and add hypergraph tokens (NOT recommended
                for production use - use pre-initialized tokenizer from model instead)
        """
        # Set basic attributes (normally done by parent __init__)
        self.opt = opt
        self.dpath = os.path.join(DATASET_PATH, "redial", "hypergraph_llava")
        
        # TODO: Load or use provided tokenizer
        if tokenizer is not None:
            # Use pre-initialized tokenizer (RECOMMENDED)
            logger.info("Using provided tokenizer (with special tokens already added)")
            self.tokenizer = tokenizer
            self._verify_special_tokens()
        else:
            # Load fresh tokenizer and add tokens (for standalone testing only)
            llava_model_path = opt.get('llava_model_path', 'liuhaotian/llava-v1.5-7b')
            logger.warning(
                f"Loading fresh tokenizer from {llava_model_path}. "
                f"For production, pass pre-initialized tokenizer from model!"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(llava_model_path)
            self._add_special_tokens()
        
        # Load or process data
        if restore:
            logger.info("Restoring dataset from saved file...")
            self.train_data, self.valid_data, self.test_data, self.side_data, self.vocab = self._load_from_restore()
        else:
            logger.info("Loading and processing dataset...")
            # Load raw data (our notebook-processed pickles)
            train_data, valid_data, test_data, side_data = self._load_raw_data()
            logger.info('[Finish data load]')
            
            # Process data
            self.train_data, self.valid_data, self.test_data, self.side_data = self._data_preprocess(
                train_data, valid_data, test_data, side_data
            )
            
            # Build vocab dict for compatibility
            self.vocab = self._build_vocab()
            logger.info('[Finish data preprocess]')
        
        # Save if requested
        if save:
            data = (self.train_data, self.valid_data, self.test_data, self.side_data, self.vocab)
            self._save_to_one(data)
        
        logger.info(f"Dataset loaded: {len(self.train_data)} train, {len(self.valid_data)} valid, {len(self.test_data)} test")
    
    def _verify_special_tokens(self):
        """Verify that special tokens exist in provided tokenizer"""
        required_tokens = ['<hg_start>', '<hg_end>', '<hg_patch>']
        
        for token in required_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                raise ValueError(
                    f"Special token '{token}' not found in tokenizer! "
                    f"Please ensure the tokenizer is initialized via model.initialize_hypergraph_tokenizer() first."
                )
        
        # Store special token IDs
        self.hg_start_id = self.tokenizer.convert_tokens_to_ids('<hg_start>')
        self.hg_end_id = self.tokenizer.convert_tokens_to_ids('<hg_end>')
        self.hg_patch_id = self.tokenizer.convert_tokens_to_ids('<hg_patch>')
        
        logger.debug(f"✓ Special tokens verified - hg_start: {self.hg_start_id}, hg_end: {self.hg_end_id}, hg_patch: {self.hg_patch_id}")
    
    def _add_special_tokens(self):
        """Add hypergraph special tokens to tokenizer"""
        special_tokens_dict = {
            'additional_special_tokens': ['<hg_start>', '<hg_end>', '<hg_patch>']
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added} hypergraph special tokens to tokenizer")
        
        # Store special token IDs
        self.hg_start_id = self.tokenizer.convert_tokens_to_ids('<hg_start>')
        self.hg_end_id = self.tokenizer.convert_tokens_to_ids('<hg_end>')
        self.hg_patch_id = self.tokenizer.convert_tokens_to_ids('<hg_patch>')
        
        logger.debug(f"Special token IDs - hg_start: {self.hg_start_id}, hg_end: {self.hg_end_id}, hg_patch: {self.hg_patch_id}")
    
    def _load_data(self):
        """
        Override parent abstract method (not used, we use _load_raw_data instead)
        """
        raise NotImplementedError("Use _load_raw_data() instead")
    
    def _load_raw_data(self):
        """
        Load preprocessed data from notebook output
        
        Returns:
            train_data: List[Dict] with keys {context_text, response_text, item, conv_id, turn_id, role}
            valid_data: Same format
            test_data: Same format
            side_data: Dict with {item2id, id2item}
        """
        logger.info(f"Loading data from {self.dpath}")
        
        # Check if data directory exists
        if not os.path.exists(self.dpath):
            raise FileNotFoundError(
                f"Data directory not found: {self.dpath}\n"
                f"Please run the notebook to generate data first!"
            )
        
        # Load recommendation samples
        with open(os.path.join(self.dpath, 'train_rec_samples_text.pkl'), 'rb') as f:
            train_rec = pickle.load(f)
        with open(os.path.join(self.dpath, 'test_rec_samples_text.pkl'), 'rb') as f:
            test_rec = pickle.load(f)
        
        # Load conversation samples
        with open(os.path.join(self.dpath, 'train_conv_samples_text.pkl'), 'rb') as f:
            train_conv = pickle.load(f)
        with open(os.path.join(self.dpath, 'test_conv_samples_text.pkl'), 'rb') as f:
            test_conv = pickle.load(f)
        
        # Load side data
        with open(os.path.join(self.dpath, 'side_data.pkl'), 'rb') as f:
            side_data = pickle.load(f)
        
        logger.debug(f"Loaded {len(train_rec)} train rec, {len(test_rec)} test rec samples")
        logger.debug(f"Loaded {len(train_conv)} train conv, {len(test_conv)} test conv samples")
        logger.debug(f"Side data keys: {list(side_data.keys())}")
        
        # Combine rec and conv samples
        train_data = self._combine_samples(train_rec, train_conv)
        test_data = self._combine_samples(test_rec, test_conv)
        
        # Use portion of test as valid (or create proper split in notebook)
        valid_size = len(test_data) // 5
        valid_data = test_data[:valid_size]
        test_data = test_data[valid_size:]
        
        return train_data, valid_data, test_data, side_data
    
    def _combine_samples(self, rec_samples: List[Dict], conv_samples: List[Dict]) -> List[Dict]:
        """
        Combine rec and conv samples into unified format
        
        CRSLab expects each sample to have both 'items' (for rec) and 'response_text' (for conv)
        We merge by (conv_id, turn_id)
        
        Returns:
            List of unified samples with format:
            {
                'conv_id': int,
                'turn_id': int,
                'context_text': str,
                'response_text': str (optional),
                'items': List[int],
                'role': str (optional)
            }
        """
        # Index conv samples by (conv_id, turn_id)
        conv_dict = {}
        for sample in conv_samples:
            key = (sample['conv_id'], sample['turn_id'])
            conv_dict[key] = sample
        
        # Merge with rec samples
        combined = []
        processed_keys = set()
        
        for rec_sample in rec_samples:
            key = (rec_sample['conv_id'], rec_sample['turn_id'])
            processed_keys.add(key)
            
            # Create unified sample
            unified_sample = {
                'conv_id': rec_sample['conv_id'],
                'turn_id': rec_sample['turn_id'],
                'context_text': rec_sample['context_text'],
                'items': [rec_sample['item']],  # List format for compatibility
            }
            
            # Add conv data if available
            if key in conv_dict:
                unified_sample['response_text'] = conv_dict[key]['response_text']
                unified_sample['role'] = conv_dict[key]['role']
            else:
                unified_sample['response_text'] = ""  # Empty response
                unified_sample['role'] = "Recommender"
            
            combined.append(unified_sample)
        
        # Add conv samples that don't have rec data
        for key, conv_sample in conv_dict.items():
            if key not in processed_keys:
                unified_sample = {
                    'conv_id': conv_sample['conv_id'],
                    'turn_id': conv_sample['turn_id'],
                    'context_text': conv_sample['context_text'],
                    'response_text': conv_sample['response_text'],
                    'role': conv_sample['role'],
                    'items': [],  # No item mentioned
                }
                combined.append(unified_sample)
        
        logger.debug(f"Combined {len(rec_samples)} rec + {len(conv_samples)} conv -> {len(combined)} unified samples")
        
        return combined
    
    def _data_preprocess(self, train_data, valid_data, test_data, side_data):
        """
        Process loaded data
        
        For HypergraphLlava, we keep text format and tokenize on-the-fly.
        This method mainly validates and adds metadata.
        
        Returns:
            Tuple of (processed_train, processed_valid, processed_test, processed_side_data)
        """
        # Add n_items to side_data
        side_data['n_items'] = len(side_data['item2id'])
        
        # Validate samples
        for split_name, split_data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
            logger.debug(f"Validating {split_name} data...")
            for i, sample in enumerate(split_data):
                # Check required fields
                assert 'context_text' in sample, f"Sample {i} missing context_text"
                assert 'items' in sample, f"Sample {i} missing items"
                
                # Ensure items is a list
                if not isinstance(sample['items'], list):
                    sample['items'] = [sample['items']] if sample['items'] else []
        
        logger.info(f"Preprocessing complete - {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
        
        return train_data, valid_data, test_data, side_data
    
    def _build_vocab(self) -> Dict:
        """
        Build vocab dict for CRSLab compatibility
        
        Unlike original _redial.py, we don't have tok2ind/ind2tok.
        Instead, we provide tokenizer info and special token IDs.
        
        Returns:
            Dict with tokenizer info and special tokens
        """
        vocab = {
            # Tokenizer info
            'vocab_size': len(self.tokenizer),
            'pad': self.tokenizer.pad_token_id,
            'eos': self.tokenizer.eos_token_id,
            'bos': self.tokenizer.bos_token_id,
            'unk': self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0,
            
            # Hypergraph special tokens
            'hg_start': self.hg_start_id,
            'hg_end': self.hg_end_id,
            'hg_patch': self.hg_patch_id,
            
            # Item info (for rec task)
            'n_entity': self.side_data.get('n_items', 0),
            'n_items': self.side_data.get('n_items', 0),
            
            # Add tokenizer reference (for dataloader to use)
            # WARNING: This makes vocab non-serializable, remove before saving
            '_tokenizer': self.tokenizer,
        }
        
        logger.debug(f"Built vocab: vocab_size={vocab['vocab_size']}, n_items={vocab['n_items']}")
        
        return vocab
    
    def get_tokenizer(self):
        """Get the tokenizer instance (useful for dataloader)"""
        return self.tokenizer
    
    def tokenize_context(self, context_text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Tokenize context text using LLaVA tokenizer
        
        Args:
            context_text: Raw context string
            max_length: Max sequence length
            
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        encoded = self.tokenizer(
            context_text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def tokenize_response(self, response_text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Tokenize response text using LLaVA tokenizer
        
        Args:
            response_text: Raw response string
            max_length: Max sequence length
            
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        encoded = self.tokenizer(
            response_text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def _save_to_one(self, data, file_name="all_data.pkl"):
        """
        Override parent method to handle non-serializable tokenizer
        
        Remove tokenizer from vocab before saving
        """
        train_data, valid_data, test_data, side_data, vocab = data
        
        # Remove tokenizer reference (not serializable)
        vocab_to_save = {k: v for k, v in vocab.items() if k != '_tokenizer'}
        
        data_to_save = (train_data, valid_data, test_data, side_data, vocab_to_save)
        
        # Call parent's save method
        super()._save_to_one(data_to_save, file_name)
    
    def _load_from_restore(self, file_name="all_data.pkl"):
        """
        Override parent method to restore tokenizer reference after loading
        """
        # Load data using parent method
        train_data, valid_data, test_data, side_data, vocab = super()._load_from_restore(file_name)
        
        # Restore tokenizer reference
        vocab['_tokenizer'] = self.tokenizer
        
        return train_data, valid_data, test_data, side_data, vocab
    
    def __len__(self):
        """Return size of training set (for compatibility)"""
        return len(self.train_data)
    
    def __getitem__(self, idx):
        """Get a single sample (for torch DataLoader compatibility)"""
        return self.train_data[idx]


# Alias for backward compatibility
HypergraphLlavaReDialDataset = ReDialDataset
