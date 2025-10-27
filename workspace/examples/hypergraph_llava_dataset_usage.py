"""
HypergraphLlava Dataset Usage Example
=====================================

This example demonstrates the CORRECT way to initialize dataset and ensure
tokenizer consistency across model and dataset.

Key Points:
1. Model initializes tokenizer and adds special tokens
2. Dataset receives the initialized tokenizer
3. All components use the SAME tokenizer instance
"""

import torch
from transformers import AutoTokenizer
from crslab.config import Config
from crslab.data.dataset.redial import ReDialDataset

# Method 1: Recommended - Initialize model first, then dataset
# ===========================================================

def correct_usage_with_model():
    """CORRECT: Model initializes tokenizer, dataset uses it"""
    
    # Step 1: Initialize model and tokenizer
    from crslab.model.crs.hypergraph_llava import HypergraphLlavaCRSModel
    
    config = Config()
    config['llava_model_path'] = 'liuhaotian/llava-v1.5-7b'
    
    # Create model (this will load tokenizer)
    model = HypergraphLlavaCRSModel(config, None, None)  # vocab and side_data can be None initially
    
    # Step 2: Initialize hypergraph tokenizer (adds special tokens)
    # This is typically done in model initialization
    from crslab.model.hypergraphLlava.HypergraphLlava import (
        DEFAULT_HG_START_TOKEN, 
        DEFAULT_HG_END_TOKEN, 
        DEFAULT_HYPERGRAPH_PATCH_TOKEN
    )
    
    # Get model's tokenizer
    tokenizer = model.tokenizer
    
    # Add special tokens (this modifies tokenizer and resizes embeddings)
    model.llava_model.initialize_hypergraph_tokenizer(
        use_hg_start_end=True,
        tokenizer=tokenizer,
        device='cuda',
        tune_graph_mlp_adapter=False,
        pretrain_graph_mlp_adapter=None
    )
    
    print(f"✓ Model tokenizer initialized with {len(tokenizer)} tokens")
    print(f"  - <hg_start>: {tokenizer.convert_tokens_to_ids('<hg_start>')}")
    print(f"  - <hg_end>: {tokenizer.convert_tokens_to_ids('<hg_end>')}")
    print(f"  - <hg_patch>: {tokenizer.convert_tokens_to_ids('<hg_patch>')}")
    
    # Step 3: Create dataset with the SAME tokenizer
    dataset = ReDialDataset(
        opt=config,
        tokenize='llava',
        restore=False,
        save=False,
        tokenizer=tokenizer  # ← PASS THE SAME TOKENIZER!
    )
    
    print(f"\n✓ Dataset initialized with {len(dataset.tokenizer)} tokens")
    print(f"  - Same tokenizer instance: {dataset.tokenizer is tokenizer}")
    
    # Step 4: Verify consistency
    assert dataset.tokenizer is tokenizer, "Tokenizers should be the same instance!"
    assert dataset.hg_start_id == tokenizer.convert_tokens_to_ids('<hg_start>')
    
    print("\n✅ SUCCESS: Model and dataset use the same tokenizer!")
    
    return model, dataset, tokenizer


# Method 2: Standalone dataset (NOT recommended for production)
# =============================================================

def standalone_dataset_usage():
    """
    NOT RECOMMENDED: Dataset adds tokens independently
    
    This can work for testing but may cause inconsistency with model
    """
    
    config = Config()
    config['llava_model_path'] = 'liuhaotian/llava-v1.5-7b'
    
    # Dataset will load its own tokenizer and add tokens
    dataset = ReDialDataset(
        opt=config,
        tokenize='llava',
        restore=False,
        save=False,
        tokenizer=None  # Will load fresh tokenizer
    )
    
    print(f"⚠️  Standalone dataset created with {len(dataset.tokenizer)} tokens")
    print(f"   This may NOT match the model's tokenizer!")
    
    return dataset


# Method 3: Best Practice for CRSLab Integration
# ==============================================

def crslab_integration():
    """
    Best practice: Initialize in the correct order
    
    Order:
    1. Load config
    2. Initialize model (loads tokenizer, adds special tokens)
    3. Get tokenizer from model
    4. Create dataset with that tokenizer
    5. Create dataloader using dataset's tokenizer
    """
    
    from crslab.config import Config
    from crslab.data import get_dataloader
    
    # 1. Load config
    config = Config('config/crs/hypergraph_llava.yaml')
    
    # 2. Initialize model first
    from crslab.model.crs.hypergraph_llava import HypergraphLlavaCRSModel
    model = HypergraphLlavaCRSModel(config, None, None)
    
    # Initialize hypergraph tokenizer
    tokenizer = model.tokenizer
    model.llava_model.initialize_hypergraph_tokenizer(
        use_hg_start_end=True,
        tokenizer=tokenizer,
        device='cuda',
        tune_graph_mlp_adapter=False
    )
    
    # 3. Create dataset with model's tokenizer
    dataset = ReDialDataset(
        opt=config,
        tokenize='llava',
        tokenizer=tokenizer  # ← Use model's tokenizer
    )
    
    # 4. Build vocab from dataset (for CRSLab compatibility)
    vocab = dataset.vocab
    
    # 5. Now you can safely recreate the model with proper vocab
    model = HypergraphLlavaCRSModel(config, vocab, dataset.side_data)
    
    # 6. Create dataloader
    train_dataloader = get_dataloader(config, dataset.train_data, vocab)
    
    print("✅ Complete CRSLab integration setup!")
    print(f"   Model vocab size: {len(tokenizer)}")
    print(f"   Dataset vocab size: {vocab['vocab_size']}")
    print(f"   Consistency check: {len(tokenizer) == vocab['vocab_size']}")
    
    return model, dataset, train_dataloader


# Example: How tokenizer flows through the system
# ===============================================

def tokenizer_flow_diagram():
    """
    Visualize how tokenizer flows through the system
    """
    
    print("=" * 60)
    print("TOKENIZER FLOW IN HYPERGRAPHLLAVA + CRSLab")
    print("=" * 60)
    
    print("""
    
    [1] Model Initialization
    ┌─────────────────────────────────┐
    │ HypergraphLlavaCRSModel.__init__│
    │  ↓                              │
    │ Load LLaVA tokenizer            │
    │ self.tokenizer = AutoTokenizer  │
    └─────────────────────────────────┘
              ↓
    [2] Add Special Tokens
    ┌─────────────────────────────────┐
    │ initialize_hypergraph_tokenizer │
    │  ↓                              │
    │ tokenizer.add_tokens([          │
    │   '<hg_start>',                 │
    │   '<hg_end>',                   │
    │   '<hg_patch>'                  │
    │ ])                              │
    │  ↓                              │
    │ model.resize_token_embeddings() │
    └─────────────────────────────────┘
              ↓
    [3] Pass to Dataset
    ┌─────────────────────────────────┐
    │ ReDialDataset.__init__(         │
    │   tokenizer=model.tokenizer  ←──│ SAME instance!
    │ )                               │
    │  ↓                              │
    │ self.tokenizer = tokenizer      │
    │ _verify_special_tokens()        │
    └─────────────────────────────────┘
              ↓
    [4] Use in Dataloader
    ┌─────────────────────────────────┐
    │ Dataloader collate_fn           │
    │  ↓                              │
    │ encoded = dataset.tokenizer(    │
    │   context_text,                 │
    │   padding=True,                 │
    │   ...                           │
    │ )                               │
    └─────────────────────────────────┘
              ↓
    [5] Model Forward
    ┌─────────────────────────────────┐
    │ HypergraphLlavaForCausalLM      │
    │  ↓                              │
    │ Process input_ids with special  │
    │ tokens <hg_start>, <hg_end>     │
    │  ↓                              │
    │ Insert hypergraph features      │
    └─────────────────────────────────┘
    
    KEY PRINCIPLE: ONE tokenizer instance shared across all components!
    """)


# Error Example: What NOT to do
# =============================

def wrong_usage_example():
    """
    WRONG: Creating separate tokenizers
    """
    
    print("\n" + "=" * 60)
    print("❌ WRONG USAGE EXAMPLE (DO NOT DO THIS!)")
    print("=" * 60)
    
    from transformers import AutoTokenizer
    
    # WRONG: Model loads tokenizer
    model_tokenizer = AutoTokenizer.from_pretrained('liuhaotian/llava-v1.5-7b')
    model_tokenizer.add_tokens(['<hg_start>', '<hg_end>', '<hg_patch>'])
    
    # WRONG: Dataset loads its own tokenizer
    dataset_tokenizer = AutoTokenizer.from_pretrained('liuhaotian/llava-v1.5-7b')
    dataset_tokenizer.add_tokens(['<hg_start>', '<hg_end>', '<hg_patch>'])
    
    # Check if they are the same
    print(f"Are they the same instance? {model_tokenizer is dataset_tokenizer}")
    print(f"Do they have same vocab size? {len(model_tokenizer) == len(dataset_tokenizer)}")
    print(f"Same special token IDs? {model_tokenizer.convert_tokens_to_ids('<hg_start>') == dataset_tokenizer.convert_tokens_to_ids('<hg_start>')}")
    
    print("\n⚠️  Even though vocab sizes match, they are DIFFERENT instances!")
    print("   This can cause subtle bugs and wasted memory.")
    print("   ALWAYS pass the same tokenizer instance!")


if __name__ == '__main__':
    print("HypergraphLlava Dataset & Tokenizer Usage Examples\n")
    
    # Show the flow diagram
    tokenizer_flow_diagram()
    
    # Show wrong usage
    wrong_usage_example()
    
    print("\n" + "=" * 60)
    print("Run the correct usage example:")
    print("=" * 60)
    
    # Uncomment to run:
    # model, dataset, tokenizer = correct_usage_with_model()
