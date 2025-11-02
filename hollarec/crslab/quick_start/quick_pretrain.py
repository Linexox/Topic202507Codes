from crslab.config import Config
from crslab.data import get_dataset
from loguru import logger
from crslab.model.crs.hollarec.HypergraphLlava.finetune import pretrain_modality_adaptor
import torch

def pretrain_step_1(config):
    dataset = get_dataset(config, config['tokenize'], restore=False, save=False)
    
    pretrain_modality_adaptor(config, dataset, device='cuda' if torch.cuda.is_available() else 'cpu')
    