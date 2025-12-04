import torch
import torch.nn as nn
from crslab.data import get_dataset
from crslab.model.crs.hollarec.HypergraphLlava import MMHypergraphLlavaModel, MMHypergraphLlavaConfig
from crslab.config import PRETRAIN_PATH
import os.path as osp
from loguru import logger

def main(data_config):
    config = MMHypergraphLlavaConfig()
    attr_dict = {
        'hidden_size': 4096,
        # hgraph tower config
        'hgraph_tower': 'HGNN',
        'pretrained_mm_hgraph_tower_path': osp.join(PRETRAIN_PATH, 'graph-instruction-pretrain', 'hgnn_clip.pt'),
        'txt_dim': 1024,
        'img_dim': 512,
        'vdo_dim': 2048,
        'ado_dim': 1024,
        'hg_hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.1,
        # hgraph projector config
        'train_mm_hgraph_proj': True,
        'pretrained_mm_hgraph_proj_path' : None,
        # vocab
        'vocab_path': osp.join(PRETRAIN_PATH, 'graph-instruction-pretrain', 'token2ind.json'),
    }
    for k, v in attr_dict.items():
        setattr(config, k, v)
    
    # redial_dataset = ReDialDataset(config)
    # redial_dataset = get_dataset(data_config, 'llava', restore=False, save=False)
    # vocab = redial_dataset.vocab
    model = MMHypergraphLlavaModel.from_pretrained('D:\.Workspace\.MODEL\HF-Model-Backup\llava-1.5-7b-hf')
    logger.info("MMHypergraphLlavaModel loaded from pretrained LlavaModel.")
    # model.initialize_hgraph_modules(vocab)
    