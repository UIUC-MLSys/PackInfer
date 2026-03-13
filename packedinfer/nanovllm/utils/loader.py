from einops import pack
import os
from glob import glob
import re
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if "experts" in weight_name: 
                    match = re.search(r"(.*\.mlp\.experts)\.(\d+)\.([a-z_]+)\.weight", weight_name)
                    if match:
                        module_path, expert_id, proj_name = match.groups()
                        moe_shard_mapping = {
                            "gate_proj": "w1", 
                            "down_proj": "w2", 
                            "up_proj": "w3"
                        }
                        shard_id = moe_shard_mapping.get(proj_name)
                        
                        try:
                            moe_module = model.get_submodule(module_path)
                            moe_module.weight_loader(
                                loaded_weight=f.get_tensor(weight_name),
                                expert_id=int(expert_id),
                                shard_id=shard_id
                            )
                        except AttributeError:
                            print(f"Error: FusedMoE module not found at {module_path} for weight {weight_name}")
                    else:
                        raise ValueError(f"Unexpected weight name format: {weight_name}")
                    continue

                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
