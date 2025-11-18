from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, LlavaConfig, LlavaModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from .hypergraph_layers import HGNN
from torch_geometric.data import Data

import os.path as osp
import json
import glob

# Hyperparameters
DUMMY_GRAPH_NODE_NUM = 256

# Speical tokens
TXT_HYPERGRAPH_TOKEN = "<txt_hgraph>"
TXT_HG_PATCH_TOKEN = "<txt_hg_patch>"
TXT_HG_START_TOKEN = "<txt_hg_start>"
TXT_HG_END_TOKEN = "<txt_hg_end>"

IMG_HYPERGRAPH_TOKEN = "<img_hgraph>"
IMG_HG_PATCH_TOKEN = "<img_hg_patch>"
IMG_HG_START_TOKEN = "<img_hg_start>"
IMG_HG_END_TOKEN = "<img_hg_end>"

VDO_HYPERGRAPH_TOKEN = "<vdo_hgraph>"
VDO_HG_PATCH_TOKEN = "<vdo_hg_patch>"
VDO_HG_START_TOKEN = "<vdo_hg_start>"
VDO_HG_END_TOKEN = "<vdo_hg_end>"

ADO_HYPERGRAPH_TOKEN = "<ado_hgraph>"
ADO_HG_PATCH_TOKEN = "<ado_hg_patch>"
ADO_HG_START_TOKEN = "<ado_hg_start>"
ADO_HG_END_TOKEN = "<ado_hg_end>"

class MMHypergraphLlavaConfig(LlavaConfig):
    model_type = "MMHypergraphLlava"
    """
    Config Attributes:
        graph_tower: str, name of the hypergraph neural network tower, e.g., "HGNN"
        hg_hidden_size: int, hidden size for hypergraph neural network layers
        use_mm_hgraph_proj: bool, whether to use the hypergraph projector
        hidden_size: int, hidden size for the main model
        pretrained_hgnn_path: str, path to the pretrained HGNN models for each modality
        use_hg_start_end: bool, whether to use start and end tokens for hypergraph patches
    """

class MMHypergraphPreTrainedModel:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def load_from_pretrained(model_name, pretrain_model_path):
    assert osp.exists(pretrain_model_path), f"Pretrained model path {pretrain_model_path} does not exist."

    with open(osp.join(pretrain_model_path, "config.json"), 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    args = MMHypergraphPreTrainedModel(config_dict)
    model = model_name(args)
    pkl_files = glob.glob(osp.join(pretrain_model_path, "*.pkl"))
    state_dict = torch.load(pkl_files[0], map_location='cpu')
    model.load_state_dict(state_dict)
    return model, args

class MMHypergraphLlavaModel(LlavaModel):
    config_class = MMHypergraphLlavaConfig

    def __init__(self, config: MMHypergraphLlavaConfig, vocab: Optional[Dict]=None):
        super().__init__(config)
        self.config = config
        self.vocab = vocab
        self._load_special_tokens()
        self._build_mm_hgraph_tower(config.graph_tower)
        config.use_mm_hgraph_proj = True
        self._build_mm_hgraph_projector(config.use_mm_hgraph_proj)
    
    def _load_special_tokens(self):
        self.hgraph_token_id = {
            "txt": self.vocab['tok2ind'][TXT_HYPERGRAPH_TOKEN],
            "img": self.vocab['tok2ind'][IMG_HYPERGRAPH_TOKEN],
            "vdo": self.vocab['tok2ind'][VDO_HYPERGRAPH_TOKEN],
            "ado": self.vocab['tok2ind'][ADO_HYPERGRAPH_TOKEN]
        }
        self.hg_patch_token_id = {
            "txt": self.vocab['tok2ind'][TXT_HG_PATCH_TOKEN],
            "img": self.vocab['tok2ind'][IMG_HG_PATCH_TOKEN],
            "vdo": self.vocab['tok2ind'][VDO_HG_PATCH_TOKEN],
            "ado": self.vocab['tok2ind'][ADO_HG_PATCH_TOKEN]
            }
        self.hg_start_token_id = {
            "txt": self.vocab['tok2ind'][TXT_HG_START_TOKEN],
            "img": self.vocab['tok2ind'][IMG_HG_START_TOKEN],
            "vdo": self.vocab['tok2ind'][VDO_HG_START_TOKEN],
            "ado": self.vocab['tok2ind'][ADO_HG_START_TOKEN]
        }
        self.hg_end_token_id = {
            "txt": self.vocab['tok2ind'][TXT_HG_END_TOKEN],
            "img": self.vocab['tok2ind'][IMG_HG_END_TOKEN],
            "vdo": self.vocab['tok2ind'][VDO_HG_END_TOKEN],
            "ado": self.vocab['tok2ind'][ADO_HG_END_TOKEN]
        }

    def _build_mm_hgraph_tower(self, graph_tower_name: str):
        self.mm_hgraph_tower = nn.ModuleDict()
        if graph_tower_name == "HGNN":
            for modality in ["txt", "img", "vdo", "ado"]:
                self.mm_hgraph_tower[modality], _ = load_from_pretrained(
                    HGNN,
                    pretrain_model_path=osp.join(
                        self.config.pretrained_hgnn_path,
                        f"{modality}_hgraph_tower"
                    )
                )
                self.mm_hgraph_tower[modality].requires_grad_(False)
        else:
            raise ValueError(f"Unsupported graph tower: {graph_tower_name}")
    
    def _build_mm_hgraph_projector(self, use_mm_hgraph_proj: bool):
        if not use_mm_hgraph_proj:
            return
        self.mm_hgraph_projector = nn.ModuleDict()
        for modality in ["txt", "img", "vdo", "ado"]:
            self.mm_hgraph_projector[modality] = nn.Linear(
                self.config.hg_hidden_size,
                self.config.hidden_size
            )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hgraph_data: Optional[List[Dict[str, Data]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            embedding_layer = self.get_input_embeddings()
            inputs_embeds = embedding_layer(input_ids)

        mm_hgraph_tower = self.mm_hgraph_tower

        if mm_hgraph_tower is not None and (input_ids.shape[1]!=1 or self.training) and hgraph_data is not None:
            with torch.no_grad():
                if type(hgraph_data) is list:
                    # shape hgraph_data: List[Dict[str, Data]], len=hgraph_batch_size
                    hgraph_node_features = []
                    for g in hgraph_data:
                        node_feature_dict = {}
                        for m in g.keys():
                            # 注意device一致性
                            node_feature_dict[m] = mm_hgraph_tower[m](g[m])
                        hgraph_node_features.append(node_feature_dict)
                else:
                    raise ValueError("hgraph_data should be a list of dicts.")
                
            if type(hgraph_data) is list:
                hgraph_node_features = [
                    {
                        m: self.mm_hgraph_projector[m](hgraph_dict[m])
                        for m in hgraph_dict.keys()
                    }
                    for hgraph_dict in hgraph_node_features
                ]
            else:
                raise ValueError("hgraph_data should be a list of dicts.")
            
            dummy_hgraph_features = torch.zeros(
                (DUMMY_GRAPH_NODE_NUM, self.config.hidden_size),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype
            )
            # dummy_hgraph_features = self.mm_hgraph_projector['txt'](dummy_hgraph_features)
            mm_dummy_hgraph_features = {
                m: self.mm_hgraph_projector[m](dummy_hgraph_features)
                for m in mm_hgraph_tower.keys()
            }

            new_inputs_embeds = []
            cur_hgraph_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                has_any_hgraph = False
                for m in ["txt", "img", "vdo", "ado"]:
                    if (cur_input_ids == self.hg_patch_token_id[m]).sum() > 0:
                        has_any_hgraph = True
                        break
                if not has_any_hgraph:
                    for m in ["txt", "img", "vdo", "ado"]:
                        cur_input_embeds = cur_input_embeds + (0. * mm_dummy_hgraph_features[m]).sum()
                    new_inputs_embeds.append(cur_input_embeds)
                    cur_hgraph_idx += 1
                    continue

                # 获取当前样本的超图特征字典
                # if cur_hgraph_idx >= len(hgraph_node_features):
                #     raise ValueError(f"Hypergraph index {cur_hgraph_idx} out of range {len(hgraph_node_features)}")
                cur_sample_hgraphs = hgraph_node_features[cur_hgraph_idx]
                
                for m in ["txt", "img", "vdo", "ado"]:
                    # 检查当前模态超图是否存在于输入中
                    if (cur_input_ids == self.hg_patch_token_id[m]).sum() == 0:
                        cur_input_embeds = cur_input_embeds + (0. * mm_dummy_hgraph_features[m]).sum()
                        continue

                    # 检查超图数据中是否包含该模态
                    if m not in cur_sample_hgraphs:
                        raise ValueError(f"Modality '{m}' expected in input but not found in hgraph_data[{cur_hgraph_idx}]")

                    if self.config.use_hg_start_end:
                        cur_hgraph_features = cur_sample_hgraphs[m].to(device=cur_input_embeds.device)
                        num_patches = cur_hgraph_features.shape[0]

                        num_start = (cur_input_ids == self.hg_start_token_id[m]).sum()
                        num_end = (cur_input_ids == self.hg_end_token_id[m]).sum()
                        if num_start != num_end:
                            raise ValueError(f"Modality '{m}': start tokens ({num_start}) != end tokens ({num_end})")
                        
                        hg_start_tokens = torch.where(cur_input_ids == self.hg_start_token_id[m])[0]
                        
                        for hg_start_token_pos in hg_start_tokens:
                            expected_end_pos = hg_start_token_pos + num_patches + 1
                            if expected_end_pos >= len(cur_input_ids) or \
                               cur_input_ids[expected_end_pos] != self.hg_end_token_id[m]:
                                raise ValueError(
                                    f"Modality '{m}': end token not at expected position {expected_end_pos}"
                                )
                            
                            if orig_embeds_params is not None:
                                cur_new_input_embeds = torch.cat((
                                    cur_input_embeds[:hg_start_token_pos].detach(),
                                    cur_input_embeds[hg_start_token_pos:hg_start_token_pos + 1],
                                    cur_hgraph_features,
                                    cur_input_embeds[hg_start_token_pos + num_patches + 1:hg_start_token_pos + num_patches + 2],
                                    cur_input_embeds[hg_start_token_pos + num_patches + 2:].detach()
                                ), dim=0)
                            else:
                                cur_new_input_embeds = torch.cat((
                                    cur_input_embeds[:hg_start_token_pos+1],
                                    cur_hgraph_features,
                                    cur_input_embeds[hg_start_token_pos + num_patches + 1:]
                                ), dim=0)
                            cur_input_embeds = cur_new_input_embeds
                    else:
                        raise NotImplementedError("Only support use_hg_start_end=True currently.")
                
                cur_hgraph_idx += 1
                new_inputs_embeds.append(cur_input_embeds)
            
            inputs_embeds = torch.stack(new_inputs_embeds, dim=0)
            return super(MMHypergraphLlavaModel, self).forward(
                input_ids = None,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )