"""
ModalityAdaptor 预训练示例
使用方法：python examples/pretrain_modality_adaptor_example.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crslab.config import Config
from crslab.data import get_dataset
from crslab.model.crs.hollarec.HypergraphLlava.finetune import pretrain_modality_adaptor


def main():
    # 加载配置
    config = Config("config/crs/hollarec/redial.yaml")
    print(f"✓ Loaded config")
    
    # 加载数据集
    print("\nLoading dataset...")
    dataset = get_dataset(config, config['tokenize'], restore=False, save=False)
    print(f"✓ Dataset loaded")
    
    # 检查嵌入
    if not hasattr(dataset, 'embeddings'):
        print("✗ Dataset does not have embeddings!")
        print("Please ensure 'load_saved_embeddings: true' in config")
        return
    
    print(f"✓ Dataset has embeddings: {len(dataset.embeddings)} movies")
    
    # 运行预训练
    print("\n" + "="*60)
    print("Starting ModalityAdaptor Pre-training")
    print("="*60)
    
    model = pretrain_modality_adaptor(config, dataset)
    
    print("\n✓ Pre-training completed!")
    print(f"Model saved to: {config.get('pretrain_save_dir', 'save/modality_adaptor')}")


if __name__ == "__main__":
    main()
