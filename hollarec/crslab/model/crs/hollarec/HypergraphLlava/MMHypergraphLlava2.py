from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import AutoConfig, AutoModel,LlavaConfig, LlavaModel

# from .hypergraph_layers import HGNN
from hypergraph_layers import HGNN
# from crslab.model.crs.hollarec.HypergraphLlava.hypergraph_layers import HGNN
# from crslab.config import PRETRAIN_PATH, DATA_PATH

import os
import os.path as osp
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from tqdm import tqdm
from loguru import logger
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

    def __init__(
        self,
        config: LlavaConfig,
        vocab: Optional[Dict[str, Dict]] = None,
    ):
        logger.info("Initializing MMHypergraphLlavaModel...")
        super().__init__(config)
        logger.info("Superclass LlavaModel initialized.")
        self.config = config
        self.vocab = vocab
        if self.vocab is not None:
            self._load_special_tokens()
        self._build_mm_hgraph_tower(self.config.hgraph_tower)
        self._build_mm_hgraph_projector()
        
    
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
                self.mm_hgraph_tower[modality] = HGNN(
                    in_channels=getattr(self.config, f'{modality}_dim'),
                    hidden_channels=self.config.hg_hidden_size * 2,
                    out_channels=self.config.hg_hidden_size,
                    num_layers=getattr(self.config, 'hg_num_layers', 2),
                    dropout=getattr(self.config, 'hg_dropout', 0.1),
                )
                self.mm_hgraph_tower[modality].requires_grad_(False)
                logger.info(f"Initialized HGNN for modality '{modality}'")
        else:
            raise ValueError(f"Unsupported graph tower: {graph_tower_name}")
    
    def _build_mm_hgraph_projector(self):
        self.mm_hgraph_projector = nn.ModuleDict()
        for modality in ["txt", "img", "vdo", "ado"]:
            self.mm_hgraph_projector[modality] = nn.Linear(
                self.config.hg_hidden_size,
                self.config.hidden_size
            )
            self.mm_hgraph_projector[modality].requires_grad_(False)
            logger.info(f"Initialized hypergraph projector for modality '{modality}'")
                
    def load_pretrained(self, pretrained_dir: str):
        for file in glob.glob(osp.join(pretrained_dir, "*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as f:
                for weight_name in tqdm(f.keys(), desc='loading from pretrained'):
                    weight_tensor = f.get_tensor(weight_name)
                    param = self.get_parameter(weight_name)
                    param.data.copy_(weight_tensor)

    def load_pretrained_llava(self, pretrained_dir: str):
        """从safetensors加载LLaVA预训练权重"""
        from safetensors.torch import load_file
        
        logger.info(f"Loading from {pretrained_dir}")
        
        # 合并所有safetensors文件
        state_dict = {}
        for file in glob.glob(osp.join(pretrained_dir, "*.safetensors")):
            state_dict.update(load_file(file))
            logger.info(f"Loaded {osp.basename(file)}")

        new_state_dict = {}
        for key, value in state_dict.items():
            # 将 'language_model.model.xxx' 转换为 'language_model.xxx'
            if key.startswith('language_model.model.'):
                new_key = key.replace('language_model.model.', 'language_model.')
                new_state_dict[new_key] = value
                # logger.debug(f"Remapped: {key} → {new_key}")
            else:
                new_state_dict[key] = value
        
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        
        logger.info(f"Loaded {len(new_state_dict)} parameters")
        logger.warning(f"Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        
        if missing_keys:
            logger.warning(f"missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"unexpected keys {unexpected_keys}")
        return missing_keys, unexpected_keys
        

    def _get_mm_hgraph_tower(self) -> nn.ModuleDict:
        return self.mm_hgraph_tower

    def _get_mm_hgraph_projector(self) -> nn.ModuleDict:
        return self.mm_hgraph_projector

    def save_2_safetensors(self, save_dir, max_bytes=5*1024**3):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = self.state_dict()

        def compute_tensor_bytes(t: torch.Tensor):
            return t.numel() * t.element_size()
        
        shards = []
        cur_shard = {}
        cur_bytes = 0

        for k, t in state_dict.items():
            tensor_bytes = compute_tensor_bytes(t)
            if cur_bytes + tensor_bytes > max_bytes and cur_shard:
                shards.append(cur_shard)
                cur_shard = {}
                cur_bytes = 0
            cur_shard[k] = t
            cur_bytes += tensor_bytes
        if cur_shard:
            shards.append(cur_shard)
            for i, shard in tqdm(enumerate(shards), desc='saving shards'):
                shard_path = os.path.join(save_dir, f"MMHypergraphLlava-{i+1:05}-of-{len(shards):05}.safetensors")
            save_file(shard, shard_path)
            # logger.info(f"Saved shard {i+1}")

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
            dummy_hgraph_features = self.mm_hgraph_projector['txt'](dummy_hgraph_features)
            
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
                        # 该模态超图不存在，使用dummy特征占位
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
                                    cur_hgraph_features, # （num_patches, hidden_size）
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
        
# AutoConfig.register("MMHypergraphLlava", MMHypergraphLlavaConfig)
# AutoModel.register(MMHypergraphLlavaConfig, MMHypergraphLlavaModel)

if __name__ == '__main__':
    config_path = 'D:\.Workspace\.MODEL\HF-Model-Backup\llava-1.5-7b-hf'
    config = MMHypergraphLlavaConfig.from_pretrained(config_path)
    attr_dict = {
        'hidden_size': 4096,
        # hgraph tower config
        'hgraph_tower': 'HGNN',
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
    }
    for k, v in attr_dict.items():
        setattr(config, k, v)
    model = MMHypergraphLlavaModel(config)
    # model.load_pretrained(config_path)
    for i, (name, _) in enumerate(model.named_parameters()):
        print(name, end='   ')
        if i>50:
            break
    model.load_pretrained_llava(config_path)
    save_path = 'D:\.Workspace\Topic202507Codes\hollarec\pretrain\mmhgllv4g-t-grounding'
    model.save_2_safetensors(save_path)
    config.save_pretrained(save_path)