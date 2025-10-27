"""
HypergraphLlava Usage Example

This file demonstrates how to use the HypergraphLlava model for multi-modal understanding
including image, video, and audio hypergraphs.
"""

import torch
from transformers import AutoTokenizer
from torch_geometric.data import Data
from HypergraphLlava import HypergraphLlavaConfig, HypergraphLlavaForCausalLM

# ============================================================================
# 1. Configuration Setup
# ============================================================================

config = HypergraphLlavaConfig(
    # Base LLM configuration
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    
    # Hypergraph-specific configuration
    graph_tower='HGNN',  # Hypergraph neural network type
    hg_hidden_channels=768,  # Hypergraph feature dimension
    hg_num_layers=2,  # Number of HGNN layers
    hg_dropout=0.1,  # Dropout rate
    use_graph_proj=True,  # Use projection layer
    
    # Other settings
    use_cache=True,
    output_attentions=False,
    output_hidden_states=False,
)

# ============================================================================
# 2. Model and Tokenizer Initialization
# ============================================================================

# Initialize model
model = HypergraphLlavaForCausalLM(config)

# Initialize tokenizer (use LLaVA or Llama tokenizer)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Initialize hypergraph-related tokens
model.initialize_hypergraph_tokenizer(
    use_hg_start_end=True,  # Use start/end tokens to wrap hypergraph features
    tokenizer=tokenizer,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    tune_graph_mlp_adapter=False,
    pretrain_graph_mlp_adapter=None
)

# Move model to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# ============================================================================
# 3. Prepare Hypergraph Data
# ============================================================================

def create_sample_hypergraph(num_nodes=10, num_edges=5, feature_dim=768):
    """Create a sample hypergraph for demonstration"""
    # Node features
    x = torch.randn(num_nodes, feature_dim)
    
    # Hyperedge incidence matrix (simplified representation)
    # In practice, you would construct this from your actual hypergraph structure
    edge_index = torch.randint(0, num_nodes, (2, num_edges * 3))
    
    return Data(x=x, edge_index=edge_index)

# Example 1: Single modality (image hypergraph)
image_hypergraph = create_sample_hypergraph(num_nodes=20, feature_dim=768)

# Example 2: Multi-modality (image + audio + video)
multi_modal_hypergraph = {
    'image': create_sample_hypergraph(num_nodes=15, feature_dim=768),
    'audio': create_sample_hypergraph(num_nodes=10, feature_dim=768),
    'video': create_sample_hypergraph(num_nodes=25, feature_dim=768),
}

# ============================================================================
# 4. Prepare Input Text with Hypergraph Placeholders
# ============================================================================

# For single modality
text_single = "USER: <hg_start><hg_path><hg_path><hg_path><hg_end> What can you tell me about this image? ASSISTANT:"

# For multi-modality (all modalities concatenated)
text_multi = "USER: <hg_start>" + "<hg_path>" * 50 + "<hg_end> Describe the content across image, audio, and video. ASSISTANT:"

# Tokenize
input_ids_single = tokenizer(text_single, return_tensors='pt').input_ids.to(device)
input_ids_multi = tokenizer(text_multi, return_tensors='pt').input_ids.to(device)

# ============================================================================
# 5. Forward Pass
# ============================================================================

# Single modality inference
with torch.no_grad():
    outputs_single = model(
        input_ids=input_ids_single,
        graph_data=[image_hypergraph],  # List of hypergraphs
        return_dict=True
    )
    
    print("Single Modality Output:")
    print(f"  Loss: {outputs_single['loss']}")
    print(f"  Logits shape: {outputs_single['logits'].shape}")

# Multi-modality inference
with torch.no_grad():
    outputs_multi = model(
        input_ids=input_ids_multi,
        graph_data=[multi_modal_hypergraph],  # Dict with multiple modalities
        return_dict=True
    )
    
    print("\nMulti-Modality Output:")
    print(f"  Loss: {outputs_multi['loss']}")
    print(f"  Logits shape: {outputs_multi['logits'].shape}")

# ============================================================================
# 6. Training Example
# ============================================================================

# Prepare training data
train_text = "USER: <hg_start>" + "<hg_path>" * 20 + "<hg_end> What objects are in the image? ASSISTANT: There are cats and dogs."
train_inputs = tokenizer(train_text, return_tensors='pt')
input_ids = train_inputs.input_ids.to(device)
labels = input_ids.clone()

# Create hypergraph data
train_hypergraph = create_sample_hypergraph(num_nodes=20, feature_dim=768)

# Training step
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

optimizer.zero_grad()
outputs = model(
    input_ids=input_ids,
    labels=labels,
    graph_data=[train_hypergraph],
    return_dict=True
)

loss = outputs['loss']
loss.backward()
optimizer.step()

print(f"\nTraining Loss: {loss.item()}")

# ============================================================================
# 7. Generation Example
# ============================================================================

model.eval()

generation_text = "USER: <hg_start>" + "<hg_path>" * 15 + "<hg_end> Describe this image in detail. ASSISTANT:"
generation_inputs = tokenizer(generation_text, return_tensors='pt').input_ids.to(device)
generation_hypergraph = create_sample_hypergraph(num_nodes=15, feature_dim=768)

# Note: For actual generation, you'd need to implement a generate() method
# or use transformers' GenerationMixin
print("\nGeneration example prepared. Implement generate() for actual text generation.")

# ============================================================================
# 8. Batch Processing Example
# ============================================================================

batch_texts = [
    "USER: <hg_start>" + "<hg_path>" * 10 + "<hg_end> What is in this image? ASSISTANT:",
    "USER: <hg_start>" + "<hg_path>" * 12 + "<hg_end> Describe the audio content. ASSISTANT:",
]

batch_hypergraphs = [
    create_sample_hypergraph(num_nodes=10, feature_dim=768),
    create_sample_hypergraph(num_nodes=12, feature_dim=768),
]

# Tokenize with padding
batch_inputs = tokenizer(batch_texts, return_tensors='pt', padding=True).to(device)

with torch.no_grad():
    batch_outputs = model(
        input_ids=batch_inputs.input_ids,
        attention_mask=batch_inputs.attention_mask,
        graph_data=batch_hypergraphs,
        return_dict=True
    )

print(f"\nBatch Processing:")
print(f"  Batch size: {batch_inputs.input_ids.shape[0]}")
print(f"  Output logits shape: {batch_outputs['logits'].shape}")

print("\n" + "="*80)
print("Usage example completed successfully!")
print("="*80)
