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
import enum
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin

from .other import CONFIG_NAME


class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    LAZY_LORA = "LAZY_LORA"
    ADALORA = "ADALORA"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"


class TaskType(str, enum.Enum):
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"


@dataclass
class PeftConfigMixin(PushToHubMixin):
    r"""
    This is the base configuration class for PEFT adapter models. It contains all the methods that are common to all
    PEFT adapter models. This class inherits from [`~transformers.utils.PushToHubMixin`] which contains the methods to
    push your model to the Hub. The method `save_pretrained` will save the configuration of your adapter model in a
    directory. The method `from_pretrained` will load the configuration of your adapter model from a directory.

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
    """
    peft_type: Optional[PeftType] = field(default=None, metadata={"help": "The type of PEFT model."})

    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This method saves the configuration of your adapter model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the [`~transformers.utils.PushToHubMixin.push_to_hub`]
                method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        output_dict = self.__dict__
        output_path = os.path.join(save_directory, CONFIG_NAME)

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, subfolder=None, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the Hub repository id where the configuration is saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the child class initialization.
        """
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )
        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME) # 'bigscience/bloomz-560m_PREFIX_TUNING_CAUSAL_LM_epoch200/adapter_config.json' NOTE
        else:
            try:
                config_file = hf_hub_download(pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder)
            except Exception:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file) # {'base_model_name_or_path': 'bigscience/bloomz-560m', 'encoder_hidden_size': 1024, 'inference_mode': True, 'num_attention_heads': 16, 'num_layers': 24, 'num_transformer_submodules': 1, 'num_virtual_tokens': 30, 'peft_type': 'PREFIX_TUNING', 'prefix_projection': False, 'task_type': 'CAUSAL_LM', 'token_dim': 1024}

        config = cls(**kwargs) # <class 'peft.utils.config.PeftConfig'>

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                if key == 'prompt_tuning_config':
                    # for lazy lora only
                    from ..tuners.prompt_tuning import PromptTuningConfig
                    temp = PromptTuningConfig()
                    for k, v in value.items():
                        if hasattr(temp, k):
                            setattr(temp, k, v)
                    value = temp
                setattr(config, key, value) # NOTE TODO 很多内容，没有导入进来啊... why?

        return config # PeftConfig(peft_type='PREFIX_TUNING', base_model_name_or_path='bigscience/bloomz-560m', task_type='CAUSAL_LM', inference_mode=True)

    @classmethod
    def from_json_file(cls, path_json_file, **kwargs):
        r"""
        Loads a configuration file from a json file.

        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object


@dataclass
class PeftConfig(PeftConfigMixin):
    """
    This is the base configuration class to store the configuration of a [`PeftModel`].

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
        task_type (Union[[`~peft.utils.config.TaskType`], `str`]): The type of task to perform.
        inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """

    base_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    peft_type: Union[str, PeftType] = field(default=None, metadata={"help": "Peft type"})
    task_type: Union[str, TaskType] = field(default=None, metadata={"help": "Task type"})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})


@dataclass
class PromptLearningConfig(PeftConfig):
    """
    This is the base configuration class to store the configuration of [`PrefixTuning`], [`PromptEncoder`], or
    [`PromptTuning`].

    Args:
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
    """

    num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})
    token_dim: int = field(
        default=None, metadata={"help": "The hidden embedding dimension of the base transformer model"}
    )
    num_transformer_submodules: Optional[int] = field(
        default=None, metadata={"help": "Number of transformer submodules"}
    )
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    num_layers: Optional[int] = field(default=None, metadata={"help": "Number of transformer layers"})
