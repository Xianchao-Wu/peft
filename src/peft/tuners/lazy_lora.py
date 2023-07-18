# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union #, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from .prompt_tuning import PromptTuningConfig
from .prefix_tuning import PrefixTuningConfig

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LAZY_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PromptLearningConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)


if is_bnb_available():
    import bitsandbytes as bnb
    import bitsandbytes.functional as Fbnb


@dataclass
class LazyLoraConfig(PromptLearningConfig): #PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LazyLoraModel`].

    Args:
        r (`int`): lazy Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lazy Lora to.
        lazy_lora_alpha (`float`): The alpha parameter for Lazy Lora scaling.
        lazy_pre_lora_alpha (`float`): The alpha parameter for Pre-Lazy Lora (LLaMA adapter) scaling.
        lazy_lora_dropout (`float`): The dropout probability for Lazy Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        is_r_by_svd (`bool`): Set this to True if use singular value of pretrained weight matrices to dynamically determine rank r, where averaged rank budget is still determined by the given r.
        rank_file (`str`): exist dynamic rank file for direct loading and time saving.
        r_by_module_dict (`Dict{str:int}`): Dict of module-to-rank for lazy lora.
        is_r_reuse (`bool`): Set this to True if we reuse the ranks given in r_by_module_dict.
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lazy Lora. Can be 'none', 'all' or 'lora_only'
        lazy_pre_adapter_type (`str`): LLaMA adapter for lazy lora. Can be 'linear', 'conv1d', or 'none'.
        modules_to_save (`List[str]`): List of modules apart from Lazy LoRA layers to be set as trainable
            and saved in the final checkpoint.
        init_lazy_lora_weights (`bool`): Set this to True if initialize the weights of lazy lora layers. Default is True.

        prompt_tuning_config (`PromptTuningConfig`): The config for prompt tuning
        prefix_tuning_config (`PrefixTuningConfig`): The config for prefix tuning

    """

    r: int = field(default=8, metadata={"help": "Lazy Lora attention dimension, can be fixed or dynamically determined by singular values of weight matrices"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lazy Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lazy_lora_alpha: int = field(default=None, metadata={"help": "lazy Lora alpha"})
    lazy_pre_lora_alpha: float = field(default=None, metadata={"help": "lazy Pre-Lora adapter alpha, default=0.1"})
    lazy_lora_dropout: float = field(default=None, metadata={"help": "Lazy Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    is_r_by_svd: bool = field(
        default=False,
        metadata={"help": "if True, then use singular value to determine rank r and averaged rank budget is still determined by the given r"},
    )
    rank_file: str = field(
        default='',
        metadata={"help": "Existing rank file in .json format, for rank value reusage and time saving."},
    )
    r_by_module_dict : Optional[dict] = field(
        default=None,
        metadata={
            'help': "dictionary of module.key to its rank, where the rank is determined by the module's weight matrix's singular value."
        },
    )
    is_r_reuse: bool = field(
        default=True,
        metadata={"help": "if True, then reuse the ranks stored in r_by_module_dict"},
    )
    bias: str = field(
        default="none", 
        metadata={"help": "Bias type for lazy Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    lazy_pre_adapter_type: str = field(
        default="linear", 
        metadata={"help": "LLaMa Adapter type for lazy Lora. Can be 'none', 'linear' or 'conv1d'"}
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from lazy LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lazy_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the lazy Lora layers."},
    )

    #--- for prompt learning
    prompt_tuning_config : PromptTuningConfig = field(
        default=None,
        metadata={'help': 'config for prompt tuning in lazy lora'}
    )

    #--- for prefix learning
    prefix_tuning_config : PrefixTuningConfig = field(
        default=None,
        metadata={'help': 'config for prefix tuning in lazy lora'}
    )

    def __post_init__(self):
        self.peft_type = PeftType.LAZY_LORA


class LazyLoraModel(torch.nn.Module):
    """
    Creates Lazy Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LazyLoraConfig`]): The configuration of the lazy Lora model.

    Returns:
        `torch.nn.Module`: The lazy Lora model.

    Example:
        ```py
        >>> import os
        >>> from transformers AutoModelForCausalLM, BitsAndBytesConfig
        >>> model_name_or_path = 'bigscience/bloomz-560m'
        >>> cache_dir = os.getcwd() # change this

        >>> bnb_config = BitsAndBytesConfig(
        ...     load_in_4bit=True,
        ...     bnb_4bit_use_double_quant=True,
        ...     bnb_4bit_quant_type='nf4',
        ...     bnb_4bit_compute_dtype=torch.bfloat16
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
        ...     quantization_config=bnb_config, # 4-bit, change this
        ...     device_map={"":0}, # gpu0, change this
        ...     cache_dir=cache_dir) 

        >>> from peft import LazyLoraConfig, get_peft_model, PrefixTuningConfig, TaskType, PeftType, PromptTuningConfig, PromptTuningInit

        >>> peft_config_prompt_tuning = PromptTuningConfig(
        ...     task_type=TaskType.CAUSAL_LM,
        ...     prompt_tuning_init=PromptTuningInit.TEXT,
        ...     num_virtual_tokens=8,
        ...     prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
        ...     tokenizer_name_or_path=model_name_or_path,
        ... ) 

        >>> peft_config_prefix_tuning = PrefixTuningConfig(
        ...     task_type=TaskType.CAUSAL_LM,
        ...     num_virtual_tokens=30
        ... )

        >>> config_lazy_lora = LazyLoraConfig(
        ...     r=8,
        ...     is_r_by_svd=True, 
        ...     lazy_lora_alpha=32,
        ...     lazy_pre_lora_alpha=0.1,
        ...     lazy_pre_adapter_type='linear', #'linear', 'conv1d', 'none'
        ...     target_modules=['query_key_value', 'dense_h_to_4h', 'dense_4h_to_h'],
        ...     lazy_lora_dropout=0.05,
        ...     bias='none',
        ...     task_type=TaskType.CAUSAL_LM,
        ...     prompt_tuning_config=peft_config_prompt_tuning,
        ...     prefix_tuning_config=peft_config_prefix_tuning,
        ... )
        >>> model = get_peft_model(model, config_lazy_lora)

        ```

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import LazyLoraModel, LazyLoraConfig

        >>> config = LazyLoraConfig(
        ...     peft_type="LAZY_LORA",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     is_r_by_svd=True,
        ...     lazy_lora_alpha=32,
        ...     lazy_pre_lora_alpha=0.1,
        ...     lazy_pre_adapter_type='linear', 
        ...     target_modules=["q", "v"],
        ...     lazy_lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lazy_lora_model = LazyLoraModel(config, model)
        ```

        ```py
        >>> import transformers
        >>> from peft import LazyLoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LazyLoraConfig(
        ...     r=4, lazy_lora_alpha=16, lazy_pre_lora_alpha=0.1, lazy_pre_adapter_type='linear', target_modules=target_modules, lazy_lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> lazy_lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LazyLoraConfig`]): The configuration of the lazy Lora model.
    """

    def __init__(self, model, config, adapter_name):
        #import ipdb; ipdb.set_trace()
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config 
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None: 
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lazy_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name) # NOTE
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LazyLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lazy_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _svd_s_sum(self, weight):
        _, si, _ = torch.svd(weight.to(torch.float32))
        return sum(si).item()

    def _find_rank_by_svd(self, key_list, lazy_lora_config, loaded_in_4bit, loaded_in_8bit):
        #import ipdb; ipdb.set_trace()
        if lazy_lora_config.r_by_module_dict is not None and lazy_lora_config.is_r_reuse:
            return lazy_lora_config.r_by_module_dict

        key_to_rank_dict = dict()
        for key in key_list:
            if 'embed' in key or 'head' in key:
                continue # skip 'word_embeddings' and 'lm_head'
            if isinstance(lazy_lora_config.target_modules, str):
                target_module_found = re.fullmatch(lazy_lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lazy_lora_config.target_modules)
            if target_module_found:
                #import ipdb; ipdb.set_trace()
                parent, target, target_name = _get_submodules(self.model, key)
                if loaded_in_4bit:
                    if self.model.config.quantization_config.bnb_4bit_quant_type == 'fp4':
                        weight = Fbnb.dequantize_fp4(target.weight, target.weight.quant_state)
                        s_value_sum = self._svd_s_sum(weight)
                        key_to_rank_dict[key] = s_value_sum
                    elif self.model.config.quantization_config.bnb_4bit_quant_type == 'nf4':
                        weight = Fbnb.dequantize_nf4(target.weight, target.weight.quant_state)
                        s_value_sum = self._svd_s_sum(weight)
                        key_to_rank_dict[key] = s_value_sum
                    else:
                        raise ValueError('only {fp4, nf4} are supported for bnb_4bit_quant_type, given={}'.format(self.model.config.quantization_config.bnb_4bit_quant_type))

                elif loaded_in_8bit:
                    # TODO need to confirm if this is okay?
                    weight = target.weight
                    s_value_sum = self._svd_s_sum(weight)
                    key_to_rank_dict[key] = s_value_sum
                else:
                    weight = target.weight
                    s_value_sum = self._svd_s_sum(weight)
                    key_to_rank_dict[key] = s_value_sum

        #import ipdb; ipdb.set_trace()
        #print(key_to_rank_dict)
        budget = lazy_lora_config.r * lazy_lora_config.num_layers
        #if isinstance(lazy_lora_config.target_modules, list):
        #    budget = budget * len(lazy_lora_config.target_modules)
        target_keys = lazy_lora_config.target_modules
        if isinstance(target_keys, str):
            target_keys = [target_keys]
        for target_key in target_keys:
            a_key_list = [akey for akey in key_to_rank_dict.keys() if akey.endswith(target_key)]
            s_value_total = sum([key_to_rank_dict[akey] for akey in a_key_list])

            for akey in a_key_list:
                key_to_rank_dict[akey] = round(budget * key_to_rank_dict[akey]/s_value_total)
        #import ipdb; ipdb.set_trace()
        print(key_to_rank_dict)
        lazy_lora_config.r_by_module_dict = key_to_rank_dict
        return key_to_rank_dict

    def _find_and_replace(self, adapter_name):
        #import ipdb; ipdb.set_trace()
        lazy_lora_config = self.peft_config[adapter_name]
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available(): # NOTE not in
            raise ImportError(
                "To use Lazy Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lazy_lora_config.r, # 8
            "lazy_lora_alpha": lazy_lora_config.lazy_lora_alpha, # 32
            "lazy_pre_lora_alpha": lazy_lora_config.lazy_pre_lora_alpha, # 0.1
            "lazy_lora_dropout": lazy_lora_config.lazy_lora_dropout, # 0.05
            "fan_in_fan_out": lazy_lora_config.fan_in_fan_out, # False
            "init_lazy_lora_weights": lazy_lora_config.init_lazy_lora_weights, # True
            "lazy_pre_adapter_type": lazy_lora_config.lazy_pre_adapter_type # 'linear', 'conv1d', or 'none'
        }
        key_list = [key for key, _ in self.model.named_modules()]
        key_to_rank_dict = None
        if lazy_lora_config.is_r_by_svd:
            key_to_rank_dict = self._find_rank_by_svd(
                key_list, 
                lazy_lora_config,
                loaded_in_4bit,
                loaded_in_8bit,
            )
        for key in key_list:
            if isinstance(lazy_lora_config.target_modules, str):
                target_module_found = re.fullmatch(lazy_lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lazy_lora_config.target_modules)
            if target_module_found:
                #import ipdb; ipdb.set_trace()
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key) # NOTE e.g., key='transformer.h.0.self_attention.query_key_value'
                if hasattr(target, "bias"):
                    bias = target.bias is not None # bias=True

                if isinstance(target, LazyLoraLayer):
                    target.update_layer(
                        adapter_name,
                        lazy_lora_config.r if (key_to_rank_dict == None or key not in key_to_rank_dict) else key_to_rank_dict[key],
                        lazy_lora_config.lazy_lora_alpha,
                        lazy_lora_config.lazy_lora_dropout,
                        lazy_lora_config.init_lazy_lora_weights,
                        lazy_lora_config.lazy_pre_lora_alpha,
                        lazy_lora_config.lazy_pre_adapter_type,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt): # NOTE
                        kwargs['r'] = lazy_lora_config.r if (key_to_rank_dict == None or key not in key_to_rank_dict) else key_to_rank_dict[key]
                        eightbit_kwargs = kwargs.copy()
                        eightbit_kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = Linear8bitLt(
                            adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
                        ) # NOTE case 1
                    elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
                        kwargs['r'] = lazy_lora_config.r if (key_to_rank_dict == None or key not in key_to_rank_dict) else key_to_rank_dict[key]
                        fourbit_kwargs = kwargs.copy()
                        fourbit_kwargs.update(
                            {
                                "compute_dtype": target.compute_dtype,
                                "compress_statistics": target.weight.compress_statistics,
                                "quant_type": target.weight.quant_type,
                            }
                        )
                        new_module = Linear4bit(
                            adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs
                        ) # NOTE case 2
                    elif isinstance(target, torch.nn.Embedding):
                        #import ipdb; ipdb.set_trace()
                        embedding_kwargs = kwargs.copy()
                        embedding_kwargs.pop("fan_in_fan_out", None)
                        in_features, out_features = target.num_embeddings, target.embedding_dim
                        new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
                    else:
                        kwargs['r'] = lazy_lora_config.r if (key_to_rank_dict == None or key not in key_to_rank_dict) else key_to_rank_dict[key]
                        if isinstance(target, torch.nn.Linear): # NOTE linear layer, qkv
                            in_features, out_features = target.in_features, target.out_features # e.g., 1024, 3072
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lazy_lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lazy_lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs) # NOTE case 3, adapter_name='default', in_features=1024, out_features=3072, bias=True, kwargs={'r': 8, 'lazy_lora_alpha': 32, 'lazy_lora_dropout': 0.05, 'fan_in_fan_out': False, 'init_lazy_lora_weights': True}

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lazy_lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight # [3072, 1024], 这是使用预训练模型中的权重
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias # in, 顺带也替换一下bias vector, okay

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device, dispatch=派遣,发送
        for name, module in new_module.named_modules():
            if "lazy_lora_" in name or "lazy_pre_lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LazyLoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LazyLoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LazyLoraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LazyLoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lazy_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LAZY_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LAZY_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config # NOTE basically nothing changed (training)

    def merge_and_unload(self):
        r"""
        This method merges the lazy LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        #import ipdb; ipdb.set_trace()
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging lazy LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge lazy LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LazyLoraLayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].lazy_lora_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_lazy_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LazyLoraLayer):
                if adapter_name in target.lazy_lora_A:
                    target.lazy_lora_A[adapter_name].weight.data = target.lazy_lora_A[adapter_name].weight.data * 0.0
                    target.lazy_lora_B[adapter_name].weight.data = target.lazy_lora_B[adapter_name].weight.data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lazy_lora_A:
                            continue
                        target.lazy_lora_A[adapter_name].weight.data += (
                            target.lazy_lora_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.lazy_lora_B[adapter_name].weight.data += target.lazy_lora_B[adapter].weight.data * weight

                elif adapter_name in target.lazy_lora_embedding_A:
                    target.lazy_lora_embedding_A[adapter_name].data = target.lazy_lora_embedding_A[adapter_name].data * 0.0
                    target.lazy_lora_embedding_B[adapter_name].data = target.lazy_lora_embedding_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lazy_lora_embedding_A:
                            continue
                        target.lazy_lora_embedding_A[adapter_name].data += (
                            target.lazy_lora_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lazy_lora_embedding_B[adapter_name].data += target.lazy_lora_embedding_B[adapter].data * weight


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lazy_lora_only` to work
def mark_only_lazy_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        #if "lazy_lora_" not in n and "lazy_pre_lora_" not in n:
        if "lora_" not in n: # and "lazy_pre_lora_" not in n:
            p.requires_grad = False # 把lazy_lora_之外的，都设置为不需要梯度
    if bias == "none": # bias='none', so return
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lazy_lora_only":
        for m in model.modules():
            if isinstance(m, LazyLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LazyLoraLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        self.r = {}
        self.lazy_lora_alpha = {}
        self.lazy_pre_lora_alpha = {}
        self.scaling = {}
        self.pre_scaling = {}
        self.lazy_lora_dropout = nn.ModuleDict({}) # NOTE
        self.lazy_lora_A = nn.ModuleDict({}) # NOTE
        self.lazy_lora_B = nn.ModuleDict({}) # NOTE
        self.lazy_pre_lora_dropout = nn.ModuleDict({}) # NOTE
        self.lazy_pre_lora_A = nn.ModuleDict({}) # NOTE
        self.lazy_pre_lora_B = nn.ModuleDict({}) # NOTE
        # For Embedding layer
        self.lazy_lora_embedding_A = nn.ParameterDict({}) # NOTE
        self.lazy_lora_embedding_B = nn.ParameterDict({}) # NOTE
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features # 1024
        self.out_features = out_features # 3072

    def update_layer(
        self, 
        adapter_name, 
        r, 
        lazy_lora_alpha, 
        lazy_lora_dropout, 
        init_lazy_lora_weights,
        lazy_pre_lora_alpha,
        lazy_pre_adapter_type,
    ):
        self.r[adapter_name] = r
        self.lazy_lora_alpha[adapter_name] = lazy_lora_alpha
        self.lazy_pre_lora_alpha[adapter_name] = lazy_pre_lora_alpha
        self.lazy_pre_adapter_type = lazy_pre_adapter_type
        if lazy_lora_dropout > 0.0:
            lazy_lora_dropout_layer = nn.Dropout(p=lazy_lora_dropout)
            lazy_pre_lora_dropout_layer = nn.Dropout(p=lazy_lora_dropout)
        else:
            lazy_lora_dropout_layer = nn.Identity()
            lazy_pre_lora_dropout_layer = nn.Identity()

        self.lazy_lora_dropout.update(nn.ModuleDict({adapter_name: lazy_lora_dropout_layer})) #     (default): Dropout(p=0.05, inplace=False)
        self.lazy_pre_lora_dropout.update(nn.ModuleDict({adapter_name: lazy_pre_lora_dropout_layer})) #     (default): Dropout(p=0.05, inplace=False)
        # Actual trainable parameters
        if r > 0:
            self.lazy_lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)})) #     (default): Linear(in_features=1024, out_features=8, bias=False)
            self.lazy_lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)})) #     (default): Linear(in_features=8, out_features=3072, bias=False)
            self.scaling[adapter_name] = lazy_lora_alpha / r # 32/8=4
            self.pre_scaling[adapter_name] = lazy_pre_lora_alpha # 0.1 NOTE
           
            if self.lazy_pre_adapter_type == 'conv1d':
                self.lazy_pre_lora_A.update(
                    nn.ModuleDict(
                        {adapter_name: nn.Conv1d(self.in_features, r, 1, groups=1, bias=False)}
                    )
                ) # (default): Conv1d(in_features=1024, out_features=8, kernel_size=(1,), stride=(1,),bias=False) # NOTE 
                self.lazy_pre_lora_B.update(
                    nn.ModuleDict(
                        {adapter_name: nn.Conv1d(r, self.in_features, 1, groups=2, bias=False)}
                    )
                ) # (default): Conv1d(in_features=8, out_features=1024, kernel_size=(1,), stride=(1,),bias=False) # NOTE
            elif self.lazy_pre_adapter_type == 'linear':
                self.lazy_pre_lora_A.update(
                    nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)})
                ) 
                self.lazy_pre_lora_B.update(
                    nn.ModuleDict({adapter_name: nn.Linear(r, self.in_features, bias=False)})
                ) 
            else:
                print('no llama adapter will be used, only support linear/conv1d, lazy_pre_adapter_type={}'.format(lazy_pre_adapter_type))


        if init_lazy_lora_weights: # True
            self.reset_lazy_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(
        self, 
        adapter_name, 
        r, 
        lazy_lora_alpha, 
        lazy_lora_dropout, 
        init_lazy_lora_weights,
        lazy_pre_lora_alpha,
        lazy_pre_adapter_type
    ):
        #import ipdb; ipdb.set_trace()
        self.r[adapter_name] = r
        self.lazy_lora_alpha[adapter_name] = lazy_lora_alpha
        self.lazy_pre_lora_alpha[adapter_name] = lazy_pre_lora_alpha
        self.lazy_pre_adapter_type = lazy_pre_adapter_type
        if lazy_lora_dropout > 0.0:
            lazy_lora_dropout_layer = nn.Dropout(p=lazy_lora_dropout)
            lazy_pre_lora_dropout_layer = nn.Dropout(p=lazy_lora_dropout) # TODO separate?
        else:
            lazy_lora_dropout_layer = nn.Identity()
            lazy_pre_lora_dropout_layer = nn.Identity()

        self.lazy_lora_dropout.update(nn.ModuleDict({adapter_name: lazy_lora_dropout_layer}))
        self.lazy_pre_lora_dropout.update(nn.ModuleDict({adapter_name: lazy_pre_lora_dropout_layer}))

        # Actual trainable parameters
        if r > 0:
            self.lazy_lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((r, self.in_features)))})
            )
            self.lazy_lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros((self.out_features, r)))})
            )
            self.scaling[adapter_name] = lazy_lora_alpha / r
            self.pre_scaling[adapter_name] = lazy_pre_lora_alpha
        if init_lazy_lora_weights:
            self.reset_lazy_lora_parameters(adapter_name)
        #import ipdb; ipdb.set_trace()
        self.to(self.weight.device) # TODO why 'cpu'?

    def reset_lazy_lora_parameters(self, adapter_name):
        if adapter_name in self.lazy_lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lazy_lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lazy_lora_B[adapter_name].weight)
        
        if adapter_name in self.lazy_pre_lora_A.keys():
            if self.lazy_pre_adapter_type == 'conv1d':
                nn.init.xavier_uniform_(self.lazy_pre_lora_A[adapter_name].weight)
                nn.init.zeros_(self.lazy_pre_lora_B[adapter_name].weight)
            elif self.lazy_pre_adapter_type == 'linear':
                nn.init.kaiming_uniform_(self.lazy_pre_lora_A[adapter_name].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lazy_pre_lora_B[adapter_name].weight)
            else:
                print('skip reset pre llama adapter')

        if adapter_name in self.lazy_lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lazy_lora_embedding_A[adapter_name])
            nn.init.normal_(self.lazy_lora_embedding_B[adapter_name])


class Linear(nn.Linear, LazyLoraLayer):
    # lazy Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str, # 'default'
        in_features: int, # 1024
        out_features: int, # 3072
        r: int = 0, # r=8
        lazy_lora_alpha: int = 1, # 32
        lazy_pre_lora_alpha : float = 0.1, # 0.1
        lazy_lora_dropout: float = 0.0, # 0.05
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out), False
        lazy_pre_adapter_type: str = 'linear', # 'none', 'linear', or 'conv1d'
        **kwargs, # {'bias': True, 'init_lazy_lora_weights': True}
    ):
        init_lazy_lora_weights = kwargs.pop("init_lazy_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs) # kwargs={'bias': True}, 这是原来的线性层 -> Linear(in_features=1024, out_features=3072, bias=True) NOTE
        LazyLoraLayer.__init__(self, in_features=in_features, out_features=out_features) # NOTE
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False # 原来的线性层的weight，冻结

        self.fan_in_fan_out = fan_in_fan_out # False
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self) # NOTE for what? 重新设置self.weight的值... 没啥用啊... 后续还需要从checkpoint中读取... TODO
        self.update_layer(adapter_name, r, lazy_lora_alpha, lazy_lora_dropout, init_lazy_lora_weights, lazy_pre_lora_alpha, lazy_pre_adapter_type) # NOTE
        self.active_adapter = adapter_name # 'default'

    def merge(self):
        #import ipdb; ipdb.set_trace()
        if self.active_adapter not in self.lazy_lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lazy_lora_B[self.active_adapter].weight @ self.lazy_lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        #import ipdb; ipdb.set_trace()
        if self.active_adapter not in self.lazy_lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lazy_lora_B[self.active_adapter].weight @ self.lazy_lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def forward(self, x: torch.Tensor): # x.shape=[8, 64, 1024]
        #import ipdb; ipdb.set_trace()
        previous_dtype = x.dtype
        if self.active_adapter not in self.lazy_lora_A.keys(): # self.active_adapter='default'
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            #import ipdb; ipdb.set_trace()
            debug_conv1d = (self.lazy_pre_adapter_type == 'conv1d') 
            if debug_conv1d and self.active_adapter in self.lazy_pre_lora_A:
                previous_dtype_1 = x.dtype
                x = x.to(self.lazy_pre_lora_A[self.active_adapter].weight.dtype)
                x = x.transpose(1, 2)
                x = self.pre_scaling[self.active_adapter] * self.lazy_pre_lora_B[self.active_adapter](
                        self.lazy_pre_lora_dropout[self.active_adapter](
                            self.lazy_pre_lora_A[self.active_adapter](x)
                            ) # A -> dropout -> B
                        ) + x  
                x = x.transpose(1, 2).contiguous()
                x = x.to(previous_dtype_1)
            
            debug_linear = (self.lazy_pre_adapter_type == 'linear')
            if debug_linear and self.active_adapter in self.lazy_pre_lora_A:
                previous_dtype_2 = x.dtype
                x = x.to(self.lazy_pre_lora_A[self.active_adapter].weight.dtype)
                x = self.pre_scaling[self.active_adapter] * self.lazy_pre_lora_B[self.active_adapter](
                        self.lazy_pre_lora_A[self.active_adapter](
                            self.lazy_pre_lora_dropout[self.active_adapter](x)
                            ) # dropout -> A -> B
                        ) + x  
                x = x.to(previous_dtype_2)

            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias) # NOTE here
            # 这是调用原来的预训练里面的weight/bias, result.shape=[8, 64, 3072]
            x = x.to(self.lazy_lora_A[self.active_adapter].weight.dtype)

            result += (
                self.lazy_lora_B[self.active_adapter](
                    self.lazy_lora_A[self.active_adapter](
                        self.lazy_lora_dropout[self.active_adapter](x)
                    )
                )
                * self.scaling[self.active_adapter]
            ) # residual add; x + 4.0 * (dropout -> A=1024-to-8 -> B=8-to-3072)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


class Embedding(nn.Embedding, LazyLoraLayer):
    # lazy LoRA implemented in a Embedding layer
    def __init__(
        self,
        adapter_name: str,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lazy_lora_alpha: int = 1,
        lazy_pre_lora_alpha: float = 0.1,
        lazy_lora_dropout: float = 0.0,
        lazy_pre_adapter_type: str = 'linear', # 'none', 'linear', or 'conv1d'
        **kwargs,
    ):
        init_lazy_lora_weights = kwargs.pop("init_lazy_lora_weights", True)

        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LazyLoraLayer.__init__(self, in_features=num_embeddings, out_features=embedding_dim)

        self.weight.requires_grad = False # self.weight 是重新初始化的? 还是从pretrained checkpoint中load 过来的? NOTE TODO -> okay 后续调用了_replace_module()，是按照pretrained ckpt，对这里的self.weight的取值，以及gpu都赋值了, okay.

        nn.Embedding.reset_parameters(self)
        self.update_layer_embedding(adapter_name, r, lazy_lora_alpha, lazy_lora_dropout, init_lazy_lora_weights, lazy_pre_lora_alpha, lazy_pre_adapter_type)
        self.active_adapter = adapter_name

    def unmerge(self, mode: bool = True):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lazy_lora_embedding_B[self.active_adapter] @ self.lazy_lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def merge(self):
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lazy_lora_embedding_B[self.active_adapter] @ self.lazy_lora_embedding_A[self.active_adapter], True
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def forward(self, x: torch.Tensor):
        #import ipdb; ipdb.set_trace()
        if self.disable_adapters:
            if self.r[self.active.adapter] > 0 and self.merged:
                self.weight.data -= (
                    transpose(
                        self.lazy_lora_embedding_B[self.active_adapter].weight
                        @ self.lazy_lora_embedding_A[self.active_adapter].weight,
                        True,
                    )
                    * self.scaling[self.active_adapter]
                )
                self.merged = False
            return nn.Embedding.forward(self, x.to(self.weight.device))

        elif self.r[self.active_adapter] > 0 and not self.merged:
            if x.device != self.weight.device:
                x = x.to(self.weight.device)
            result = nn.Embedding.forward(self, x)
            if self.r[self.active_adapter] > 0:
                after_A = F.embedding(
                    x,
                    self.lazy_lora_embedding_A[self.active_adapter].T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lazy_lora_embedding_B[self.active_adapter].T) * self.scaling[self.active_adapter]
            return result
        else:
            return nn.Embedding.forward(self, x)


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LazyLoraLayer):
        # lazy Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name, # 'default'
            in_features, # 4096
            out_features, # 4096
            r: int = 0, # 8
            lazy_lora_alpha: int = 1, # 16
            lazy_pre_lora_alpha: float = 0.1,
            lazy_lora_dropout: float = 0.0, # 0.05
            lazy_pre_adapter_type: str = 'linear', # 'none', 'linear', or, 'conv1d'
            **kwargs, # {'bias': False, 'fan_in_fan_out': False, 'init_lazy_lora_weights': True, 'has_fp16_weights': False, 'memory_efficient_backward': False, 'threshold': 6.0, 'index': None}
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LazyLoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_lazy_lora_weights = kwargs.pop("init_lazy_lora_weights", True)
            self.update_layer(adapter_name, r, lazy_lora_alpha, lazy_lora_dropout, init_lazy_lora_weights, lazy_pre_lora_alpha, lazy_pre_adapter_type) # NOTE
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            #import ipdb; ipdb.set_trace()
            #debug = True
            if self.r[self.active_adapter] > 0:
                debug_conv1d = (self.lazy_pre_adapter_type == 'conv1d')
                if debug_conv1d and (self.active_adapter in self.lazy_pre_lora_A):
                    x = x.transpose(1, 2)
                    x = self.pre_scaling[self.active_adapter] * self.lazy_pre_lora_B[self.active_adapter](
                        self.lazy_pre_lora_dropout[self.active_adapter](
                            self.lazy_pre_lora_A[self.active_adapter](x)
                        ) # A -> dropout -> B
                    ) + x
                    x = x.transpose(1, 2).contiguous()

                debug_linear = (self.lazy_pre_adapter_type == 'linear')
                if debug_linear and (self.active_adapter in self.lazy_pre_lora_A):
                    previous_dtype = x.dtype
                    x = x.to(self.lazy_pre_lora_A[self.active_adapter].weight.dtype)
                    x = self.pre_scaling[self.active_adapter] * self.lazy_pre_lora_B[self.active_adapter](
                        self.lazy_pre_lora_A[self.active_adapter](
                            self.lazy_pre_lora_dropout[self.active_adapter](x)
                        ) # dropout -> A -> B
                    ) + x
                    x = x.to(previous_dtype)
            #import ipdb; ipdb.set_trace()
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lazy_lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lazy_lora_B[self.active_adapter](
                            self.lazy_lora_A[self.active_adapter](self.lazy_lora_dropout[self.active_adapter](x))
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                        self.lazy_lora_B[self.active_adapter](
                            self.lazy_lora_A[self.active_adapter](self.lazy_lora_dropout[self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                    )
                result += output
            return result

    if is_bnb_4bit_available():

        class Linear4bit(bnb.nn.Linear4bit, LazyLoraLayer):
            # lazy Lora implemented in a dense layer
            def __init__(
                self,
                adapter_name,
                in_features,
                out_features,
                r: int = 0,
                lazy_lora_alpha: int = 1,
                lazy_pre_lora_alpha : float = 0.1, 
                lazy_lora_dropout: float = 0.0,
                lazy_pre_adapter_type: str = 'linear', # 'none', 'linear', or 'conv1d'
                **kwargs,
            ):
                bnb.nn.Linear4bit.__init__(
                    self,
                    in_features,
                    out_features,
                    bias=kwargs.get("bias", True),
                    compute_dtype=kwargs.get("compute_dtype", torch.float32),
                    compress_statistics=kwargs.get("compress_statistics", True),
                    quant_type=kwargs.get("quant_type", "nf4"),
                )
                LazyLoraLayer.__init__(self, in_features=in_features, out_features=out_features)

                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                init_lazy_lora_weights = kwargs.pop("init_lazy_lora_weights", True)
                self.update_layer(adapter_name, r, lazy_lora_alpha, lazy_lora_dropout, init_lazy_lora_weights, lazy_pre_lora_alpha, lazy_pre_adapter_type)
                self.active_adapter = adapter_name

            def forward(self, x: torch.Tensor):
                #import ipdb; ipdb.set_trace()
                # --- llama adapter, pre ---
                if self.r[self.active_adapter] > 0:
                    debug_conv1d = (self.lazy_pre_adapter_type == 'conv1d')
                    if debug_conv1d and (self.active_adapter in self.lazy_pre_lora_A):
                        x = x.transpose(1,2)
                        x = self.pre_scaling[self.active_adapter] * self.lazy_pre_lora_B[self.active_adapter](
                            self.lazy_pre_lora_dropout[self.active_adapter](
                                self.lazy_pre_lora_A[self.active_adapter](x)
                            ) # A -> dropout -> B
                        ) + x
                        x = x.transpose(1, 2).contiguous()

                    debug_linear = (self.lazy_pre_adapter_type == 'linear')
                    if debug_linear and (self.active_adapter in self.lazy_pre_lora_A):
                        #import ipdb; ipdb.set_trace() # NOTE 
                        previous_dtype_2 = x.dtype
                        x = x.to(self.lazy_pre_lora_A[self.active_adapter].weight.dtype)
                        x = self.pre_scaling[self.active_adapter] * self.lazy_pre_lora_B[self.active_adapter](
                            self.lazy_pre_lora_A[self.active_adapter](
                                self.lazy_pre_lora_dropout[self.active_adapter](x)
                            ) # dropout -> A -> B
                        ) + x
                        x = x.to(previous_dtype_2)
                #import ipdb; ipdb.set_trace()
                result = super().forward(x) # why? > /usr/local/lib/python3.8/dist-packages/bitsandbytes/nn/modules.py(207)forward()

                if self.disable_adapters or self.active_adapter not in self.lazy_lora_A.keys():
                    return result
                elif self.r[self.active_adapter] > 0:
                    result = result.clone()
                    if not torch.is_autocast_enabled():
                        expected_dtype = result.dtype
                        x = x.to(self.lazy_lora_A[self.active_adapter].weight.dtype)
                        output = (
                            self.lazy_lora_B[self.active_adapter](
                                self.lazy_lora_A[self.active_adapter](
                                    self.lazy_lora_dropout[self.active_adapter](x)
                                )
                            ).to(expected_dtype)
                            * self.scaling[self.active_adapter]
                        )
                    else:
                        output = (
                            self.lazy_lora_B[self.active_adapter](
                                self.lazy_lora_A[self.active_adapter](
                                    self.lazy_lora_dropout[self.active_adapter](x)
                                )
                            )
                            * self.scaling[self.active_adapter]
                        )
                    result += output
                return result

