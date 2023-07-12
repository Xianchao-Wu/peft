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
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from ..utils import PeftType, PromptLearningConfig


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


@dataclass
class PromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        prompt_tuning_init (Union[[`PromptTuningInit`], `str`]): The initialization of the prompt embedding.
        prompt_tuning_init_text (`str`, *optional*):
            The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_name_or_path (`str`, *optional*):
            The name or path of the tokenizer. Only used if `prompt_tuning_init` is `TEXT`.
    """

    prompt_tuning_init: Union[PromptTuningInit, str] = field(
        default=PromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    prompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_TUNING


class PromptEmbedding(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example:

    ```py
    >>> from peft import PromptEmbedding, PromptTuningConfig

    >>> config = PromptTuningConfig(
    ...     peft_type="PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     prompt_tuning_init="TEXT",
    ...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ... )

    >>> # t5_model.shared is the word embeddings of the base model
    >>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
    ```

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config, word_embeddings):
        super().__init__()

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules # 8*1
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim) # Embedding(8, 1024)
        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            from transformers import AutoTokenizer # NOTE in
            import os
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, cache_dir=os.getcwd()) # 'bigscience/bloomz-560m' -> BloomTokenizerFast(name_or_path='bigscience/bloomz-560m', vocab_size=250680, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=False)
            init_text = config.prompt_tuning_init_text # 'Classify if the tweet is a complaint or not:'
            init_token_ids = tokenizer(init_text)["input_ids"] # len=11
            # [8456, 6552, 1320, 368, 98254, 632, 267, 106863, 791, 1130, 29]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens -> 这个有问题... 反正就是一种初始化的方法而已，不必过于纠结... 中间在训练的时候，这个都会被更新，重新根据data来学习的.
            num_text_tokens = len(init_token_ids) # 11
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens] # 只取前8个
                # [8456, 6552, 1320, 368, 98254, 632, 267, 106863]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens] # NOTE is this necessary?
            # NOTE 这里还是有bug的，因为事先设定的prompt 'Classify if the tweet is a complaint or not:'
            # 经过tokenizer之后，得到的长度为11，但是目前配置文件中，设置的是8.
            # 这种对提示prompt的截取，是有问题的. TODO
            #import ipdb; ipdb.set_trace() # check device of virtual tokens and word_embeddings TODO
            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids)).detach().clone() # [8, 1024]
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights) # 这个相当于用word_embedding_weights来初始化self.embedding.weight了 NOTE 仍然是可训练的

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings
