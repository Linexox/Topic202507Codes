from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, LlavaConfig, LlavaModel

from transformers.modeling_outputs import BaseModelOutputWithPast

from .hypergraph_layers import HGNN
from torch_geometric.data import Data
import json
import os.path as osp

DEFAULT_HYPERGRAPH_TOKEN = "<hgraph>"
DEFAULT_HYPERGRAPH_PATCH_TOKEN = "<hg_patch>"  # Fixed typo: was "<hg_path>"
DEFAULT_HG_START_TOKEN = "<hg_start>"
DEFAULT_HG_END_TOKEN = "<hg_end>"

class HypergraphLlavaConfig(LlavaConfig):
    model_type = "HypergraphLlava"

class HypergraphPretrainConfig(AutoConfig):
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

class HypergraphLlavaModel(LlavaModel):
    config_class = HypergraphLlavaConfig
    """
    config contents:
    graph_tower: str, type of hypergraph tower, e.g., 'HGNN'
    hg_hiddens_size: int, hidden size of hypergraph tower
    hg_num_layers: int, number of layers in hypergraph tower
    hg_dropout: float, dropout rate in hypergraph tower
    use_graph_proj: bool, whether to use graph projector
    hidden_size: int, hidden size of LLM
    hg_patch_token: int, token id for hypergraph patch token
    use_hg_start_end: bool, whether to use start/end tokens for hypergraph patches
    hg_start_token: int, token id for hypergraph start token
    hg_end_token: int, token id for hypergraph end token

    """

    def __init__(self, config: HypergraphLlavaConfig):
        super().__init__(config)
        if config.graph_tower == 'HGNN':
            self.graph_tower = HGNN(
                in_channels=config.hg_hiddens_size,
                hidden_channels=config.hg_hiddens_size*2,
                out_channels=config.hg_hiddens_size,
                num_layers=getattr(config, 'hg_num_layers', 2),
                dropout=getattr(config, 'hg_dropout', 0.1)
            )
        else:
            raise ValueError(f"Unsupported graph_tower: {config.graph_tower}")
        
        if hasattr(config, 'use_graph_proj'):
            self.graph_projector = nn.Linear(config.hg_hiddens_size, config.hidden_size)
        
    def get_graph_tower(self):
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower
    
    def initialize_graph_modules(self, graph_tower, pretrain_graph_mlp_adapter=None, fsdp=None):
        """Initialize hypergraph modules"""
        self.config.graph_tower = graph_tower
        
        if not hasattr(self, 'graph_tower'):
            if self.config.graph_tower == 'HGNN':
                graph_tower = HGNN(
                    in_channels=self.config.hg_hiddens_size,
                    hidden_channels=self.config.hg_hiddens_size * 2,
                    out_channels=self.config.hg_hiddens_size,
                    num_layers=getattr(self.config, 'hg_num_layers', 2),
                    dropout=getattr(self.config, 'hg_dropout', 0.1)
                )
            else:
                raise ValueError(f"Unsupported graph_tower: {self.config.graph_tower}")
        else:
            graph_tower = self.graph_tower
        
        graph_tower.requires_grad_(False)
        
        if fsdp is not None and len(fsdp) > 0:
            self.graph_tower = [graph_tower]
        else:
            self.graph_tower = graph_tower
        
        self.config.use_graph_proj = True
        
        if not hasattr(self, 'graph_projector'):
            self.graph_projector = nn.Linear(self.config.hg_hiddens_size, self.config.hidden_size)
        
        if pretrain_graph_mlp_adapter is not None:
            graph_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
            self.graph_projector.load_state_dict({k.split('.')[-1]: v for k, v in graph_projector_weights.items()})
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph_data: Optional[Union[List[Data], List[dict]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass for HypergraphLlavaModel
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values: Cached key-value pairs for fast decoding
            inputs_embeds: Pre-computed input embeddings
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            graph_data: List of hypergraph data, each element can be:
                - Data object: single modality hypergraph
                - dict: multi-modality hypergraphs like {"image": Data, "audio": Data, "video": Data}
            return_dict: Whether to return dict or tuple
        
        Returns:
            BaseModelOutputWithPast or tuple
        Workflow:
            [1] Get input embeddings (if not provided)
            [2] Process hypergraph data through graph tower to get node features
            [3] Project hypergraph features and merge into input embeddings
            [4] Call parent LlavaModel forward

        """
        
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        
        # Get input embeddings if not provided
        if inputs_embeds is None:
            # Use the standard method to get embeddings (compatible with all models)
            # LlavaModel doesn't have embed_tokens directly, it's in language_model
            embedding_layer = self.get_input_embeddings()
            inputs_embeds = embedding_layer(input_ids)
        
        graph_tower = self.get_graph_tower()
        
        # Process hypergraph data if available
        if graph_tower is not None and (input_ids.shape[1] != 1 or self.training) and graph_data is not None:
            # TODO Forward through hypergraph tower to get node features
            with torch.no_grad():
                if type(graph_data) is list:
                    hypergraph_node_features = []
                    
                    for g in graph_data:
                        if isinstance(g, Data):
                            # Single modality hypergraph
                            node_forward_out = graph_tower(g)
                            hypergraph_node_features.append(node_forward_out)
                        elif isinstance(g, dict): 
                            # Multi-modality hypergraphs
                            # Process each modality and concatenate or handle separately
                            modality_features = {}
                            for modality_name, modality_graph in g.items():
                                node_forward_out = graph_tower(modality_graph)
                                modality_features[modality_name] = node_forward_out
                            # For simplicity, concatenate all modality features
                            # You can modify this to handle them separately
                            combined_features = torch.cat(list(modality_features.values()), dim=0)
                            hypergraph_node_features.append(combined_features)
                        else:
                            raise ValueError(f'Unexpected graph_data element type: {type(g)}')
                else:
                    raise ValueError(f'graph_data is expected to be a list but got {type(graph_data)}')
            
            # TODO Project hypergraph features to LLM hidden size
            if type(graph_data) is list:
                hypergraph_node_features = [self.graph_projector(node_feature) for node_feature in hypergraph_node_features]
            else:
                raise ValueError(f'graph_data is expected to be a list but got {type(graph_data)}')
            
            # Create dummy features for gradient flow
            dummy_graph_features = torch.zeros(256, self.config.hg_hiddens_size, 
                                               device=inputs_embeds.device, 
                                               dtype=inputs_embeds.dtype)
            dummy_graph_features = self.graph_projector(dummy_graph_features)
            
            new_input_embeds = []
            cur_graph_idx = 0
            
            # TODO Merge hypergraph features into input embeddings
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                # Check if current sample has hypergraph patch tokens
                if (cur_input_ids == graph_tower.config.hg_patch_token).sum() == 0:
                    # No hypergraph in this sample, keep text embeddings only
                    cur_input_embeds = cur_input_embeds + (0. * dummy_graph_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_graph_idx += 1
                    continue
                
                if graph_tower.config.use_hg_start_end:
                    # Use start/end tokens to wrap hypergraph features
                    cur_graph_features = hypergraph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    
                    # Verify start/end tokens match
                    if (cur_input_ids == graph_tower.config.hg_start_token).sum() != \
                       (cur_input_ids == graph_tower.config.hg_end_token).sum():
                        raise ValueError("The number of hypergraph start tokens and end tokens should be the same.")
                    
                    hg_start_tokens = torch.where(cur_input_ids == graph_tower.config.hg_start_token)[0]
                    
                    for hg_start_token_pos in hg_start_tokens:
                        cur_graph_features = hypergraph_node_features[cur_graph_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_graph_features.shape[0]
                        
                        # Verify end token follows after patches
                        if cur_input_ids[hg_start_token_pos + num_patches + 1] != graph_tower.config.hg_end_token:
                            raise ValueError("The hypergraph end token should follow the hypergraph start token.")
                        
                        # Insert hypergraph features between start and end tokens
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((
                                cur_input_embeds[:hg_start_token_pos].detach(), # token before start token
                                cur_input_embeds[hg_start_token_pos:hg_start_token_pos+1], # start token
                                cur_graph_features, # hypergraph features
                                cur_input_embeds[hg_start_token_pos + num_patches + 1:hg_start_token_pos + num_patches + 2], # end token
                                cur_input_embeds[hg_start_token_pos + num_patches + 2:].detach() # tokens after end token
                            ), dim=0)
                        else: # 允许更新
                            cur_new_input_embeds = torch.cat((
                                cur_input_embeds[:hg_start_token_pos+1], # token before start token + start token
                                cur_graph_features, # hypergraph features
                                cur_input_embeds[hg_start_token_pos + num_patches + 1:] # tokens after end token
                            ), dim=0)
                        
                        cur_graph_idx += 1
                    
                    new_input_embeds.append(cur_new_input_embeds)
                else: # if dont use start/end tokens
                    # Use patch tokens to mark hypergraph positions
                    cur_graph_features = hypergraph_node_features[cur_graph_idx]
                    num_patches = cur_graph_features.shape[0]
                    
                    # Verify patch token count matches
                    if (cur_input_ids == graph_tower.config.hg_patch_token).sum() != num_patches:
                        raise ValueError("The number of hypergraph patch tokens should be the same as the number of hypergraph patches.")
                    
                    masked_indices = torch.where(cur_input_ids == graph_tower.config.hg_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    
                    # Verify patch tokens are consecutive 保证位置连续
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches, 
                                                       device=masked_indices.device, 
                                                       dtype=masked_indices.dtype)).any():
                        raise ValueError("The hypergraph patch tokens should be consecutive.")
                    
                    # Replace patch tokens with hypergraph features
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:mask_index_start].detach(),
                            cur_graph_features,
                            cur_input_embeds[mask_index_start + num_patches:].detach()
                        ), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:mask_index_start],
                            cur_graph_features,
                            cur_input_embeds[mask_index_start + num_patches:]
                        ), dim=0)
                    
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_graph_idx += 1
            
            # Verify all hypergraphs are processed
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
        
        # Call parent LlavaModel forward
        return super(HypergraphLlavaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class HypergraphLlavaForCausalLM(nn.Module):
    """HypergraphLlava model for causal language modeling with hypergraph understanding"""
    config_class = HypergraphLlavaConfig
    
    def __init__(self, config):
        super().__init__()
        self.model = HypergraphLlavaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def post_init(self):
        """Initialize weights and apply final processing"""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def get_model(self):
        return self.model
    
    def get_graph_tower(self):
        return self.get_model().get_graph_tower()
      
    def get_input_embeddings(self):
        # return self.model.embed_tokens
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        # self.model.embed_tokens = value
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings and lm_head to new_num_tokens
        [1] Create new embedding layer
        [2] copy old weights
        [3] Create new lm_head
        [4] copy old weights
        """
        old_embeddings = self.get_input_embeddings()
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        new_embeddings.weight.data[:old_embeddings.num_embeddings] = old_embeddings.weight.data
        self.set_input_embeddings(new_embeddings)
        
        old_lm_head = self.get_output_embeddings()
        new_lm_head = nn.Linear(old_lm_head.in_features, new_num_tokens, bias=False)
        new_lm_head.weight.data[:old_lm_head.out_features] = old_lm_head.weight.data
        self.set_output_embeddings(new_lm_head)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph_data: Optional[Union[List[Data], List[dict]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        """
        Forward pass for causal language modeling with hypergraph
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Cached key-value pairs
            inputs_embeds: Pre-computed input embeddings
            labels: Labels for language modeling loss
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            graph_data: List of hypergraph data
            return_dict: Whether to return dict
        
        Returns:
            Loss and logits (and other outputs if requested)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions, # whether to output attention weights
            output_hidden_states=output_hidden_states, # whether to output hidden states
            graph_data=graph_data,
            return_dict=return_dict,
        )
        
        # Extract last hidden state (compatible with both return_dict formats)
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # ? why contiguous
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for generation"""
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph_data": kwargs.get("graph_data", None),
            }
        )
        return model_inputs
    
    def initialize_hypergraph_tokenizer(
        self, 
        use_hg_start_end, 
        tokenizer, 
        device,
        tune_graph_mlp_adapter=False, 
        pretrain_graph_mlp_adapter=None
    ):
        """
        Initialize hypergraph-related tokens in tokenizer
        
        Args:
            use_hg_start_end: Whether to use start/end tokens
            tokenizer: The tokenizer to modify
            device: Device to place tensors
            tune_graph_mlp_adapter: Whether to tune the MLP adapter
            pretrain_graph_mlp_adapter: Path to pretrained MLP adapter
        """
        hypergraph_config = self.get_graph_tower().config
        hypergraph_config.use_hg_start_end = use_hg_start_end
        
        # TODO: Add patch token
        tokenizer.add_tokens([DEFAULT_HYPERGRAPH_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        
        if use_hg_start_end:
            # TODO: Add start and end tokens
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_HG_START_TOKEN, DEFAULT_HG_END_TOKEN], 
                special_tokens=True
            )
            # 拓展embedding层以及lm_head
            self.resize_token_embeddings(len(tokenizer))
            
            hypergraph_config.hg_start_token, hypergraph_config.hg_end_token = \
                tokenizer.convert_tokens_to_ids([DEFAULT_HG_START_TOKEN, DEFAULT_HG_END_TOKEN])
            
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data
                
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
            
            if tune_graph_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)
                ]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            
            if pretrain_graph_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. "
                        f"Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. "
                        f"Number of new tokens: {num_new_tokens}."
                    )
        
        hypergraph_config.hg_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_HYPERGRAPH_PATCH_TOKEN]
        )[0]


# Register the model
AutoConfig.register("HypergraphLlava", HypergraphLlavaConfig)
