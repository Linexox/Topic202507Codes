"""
Training Script for HypergraphLlava CRS Model

Usage:
    python train_hypergraph_llava.py --config config/crs/hypergraph_llava.yaml

Features:
    - Two-stage training (Recommendation -> Conversation)
    - Hypergraph multimodal integration
    - Automatic mixed precision training
    - Distributed training support
"""

import argparse
import os
import sys
import torch
from loguru import logger

# Add crslab to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crslab.config import Config
from crslab.data import get_dataloader
from crslab.evaluator import get_evaluator
from crslab.model.crs.hypergraph_llava import HypergraphLlavaCRSModel
from crslab.system.hypergraph_llava import HypergraphLlavaSystem


def parse_args():
    parser = argparse.ArgumentParser(description='Train HypergraphLlava CRS Model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save models')
    parser.add_argument('--gpu', type=str, default=None,
                       help='GPU ids (e.g., "0,1,2")')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable tensorboard logging')
    parser.add_argument('--restore', type=str, default=None,
                       help='Path to checkpoint to restore from')
    
    # Training options
    parser.add_argument('--skip_rec', action='store_true',
                       help='Skip recommendation training')
    parser.add_argument('--skip_conv', action='store_true',
                       help='Skip conversation training')
    parser.add_argument('--rec_only', action='store_true',
                       help='Train only recommendation')
    parser.add_argument('--conv_only', action='store_true',
                       help='Train only conversation')
    
    return parser.parse_args()


def setup_environment(args, config):
    """Setup training environment"""
    # Set GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        config['gpu'] = [int(i) for i in args.gpu.split(',')]
    
    # Set save directory
    if args.save_dir:
        config['save_dir'] = args.save_dir
    
    # Set training flags
    if args.skip_rec or args.conv_only:
        config['train_recommendation'] = False
    if args.skip_conv or args.rec_only:
        config['train_conversation'] = False
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(config['save_dir'], 'train.log')
    logger.add(log_file, rotation='500 MB', level='INFO')
    
    logger.info(f"Configuration:\n{config}")
    
    return config


def load_data(config):
    """Load datasets and create dataloaders"""
    logger.info('[Loading data]')
    
    # Get dataloaders
    train_dataloader, valid_dataloader, test_dataloader, vocab, side_data = get_dataloader(
        opt=config,
        dataset=config['dataset']
    )
    
    logger.info(f'Train samples: {len(train_dataloader.dataset)}')
    logger.info(f'Valid samples: {len(valid_dataloader.dataset)}')
    logger.info(f'Test samples: {len(test_dataloader.dataset)}')
    logger.info(f'Vocabulary size: {len(vocab)}')
    logger.info(f'Number of items: {side_data["n_entity"]}')
    
    return train_dataloader, valid_dataloader, test_dataloader, vocab, side_data


def create_model(config, vocab, side_data, device):
    """Create HypergraphLlava CRS model"""
    logger.info('[Creating model]')
    
    model = HypergraphLlavaCRSModel(
        opt=config,
        device=device,
        vocab=vocab,
        side_data=side_data
    )
    
    # Move to device
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    logger.info(f'Frozen parameters: {total_params - trainable_params:,}')
    
    return model


def create_system(config, model, train_dataloader, valid_dataloader, test_dataloader,
                 vocab, side_data, args):
    """Create training system"""
    logger.info('[Creating training system]')
    
    system = HypergraphLlavaSystem(
        opt=config,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        vocab=vocab,
        side_data=side_data,
        restore_system=args.restore is not None,
        debug=args.debug,
        tensorboard=args.tensorboard
    )
    
    # Load checkpoint if specified
    if args.restore:
        logger.info(f'[Restoring from {args.restore}]')
        system.load_model(args.restore)
    
    return system


def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = Config(config_file_list=[args.config], model='HypergraphLlavaCRS')
    
    # Setup environment
    config = setup_environment(args, config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load data
    train_dataloader, valid_dataloader, test_dataloader, vocab, side_data = load_data(config)
    
    # Create model
    model = create_model(config, vocab, side_data, device)
    
    # Create system
    system = create_system(
        config, model, train_dataloader, valid_dataloader, test_dataloader,
        vocab, side_data, args
    )
    
    # Start training
    logger.info('[Start training]')
    try:
        system.fit()
        logger.info('[Training completed successfully]')
    except KeyboardInterrupt:
        logger.warning('[Training interrupted by user]')
        system.save_model()
    except Exception as e:
        logger.error(f'[Training failed with error: {e}]')
        raise
    
    logger.info('[Done]')


if __name__ == '__main__':
    main()
