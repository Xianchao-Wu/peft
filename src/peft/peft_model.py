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

import inspect
import os
import warnings
from contextlib import contextmanager

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin

from .tuners import (
    AdaLoraModel,
    AdaptionPromptModel,
    LoraModel,
    LazyLoraModel,
    LazyLoraConfig,
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
)
from .utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftConfig,
    PeftType,
    PromptLearningConfig,
    TaskType,
    _set_adapter,
    _set_trainable,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    shift_tokens_right,
)


PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.LAZY_LORA: LazyLoraModel,
    PeftType.PROMPT_TUNING: PromptEmbedding,
    PeftType.P_TUNING: PromptEncoder,
    PeftType.PREFIX_TUNING: PrefixEncoder,
    PeftType.ADALORA: AdaLoraModel,
    PeftType.ADAPTION_PROMPT: AdaptionPromptModel,
}


class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.


    **Attributes**:
        - **base_model** ([`~transformers.PreTrainedModel`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
        saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
        using [`PromptLearningConfig`].
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
        using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
        backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
        in the base model if using [`PromptLearningConfig`].
    """
    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        #import ipdb; ipdb.set_trace()
        super().__init__()
        self.base_model = model
        self.config = self.base_model.config
        self.modules_to_save = None
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        self.base_model_torch_dtype = getattr(model, "dtype", None) # torch.float32
        #import ipdb; ipdb.set_trace()
        if isinstance(peft_config, LazyLoraConfig):
            self.peft_config[adapter_name] = peft_config
            self.base_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type](
                self.base_model, self.peft_config, adapter_name
            ) # Lazy LoRA
            self.set_additional_trainable_modules(peft_config, adapter_name)
            #import ipdb; ipdb.set_trace() # add prompt tuning for lazy lora
            self.add_adapter(adapter_name, peft_config)
        elif not isinstance(peft_config, PromptLearningConfig):
            self.peft_config[adapter_name] = peft_config
            self.base_model = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type](
                self.base_model, self.peft_config, adapter_name
            ) # LoRA, AdaLoRA
            self.set_additional_trainable_modules(peft_config, adapter_name)
        else:
            #import ipdb; ipdb.set_trace() # prefix-tuning, prompt-tuning, p-tuning
            self.add_adapter(adapter_name, peft_config)

    def save_pretrained(self, save_directory, **kwargs):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)

        for adapter_name, peft_config in self.peft_config.items():
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if isinstance(peft_config, PromptLearningConfig) # TODO
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model, model_id, adapter_name="default", is_trainable=False, **kwargs):
        r"""
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
        """
        from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING
        #import ipdb; ipdb.set_trace()
        # load the config
        config = PEFT_TYPE_TO_CONFIG_MAPPING[
            PeftConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None)).peft_type
        ].from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        #import ipdb; ipdb.set_trace()
        if not isinstance(config, LazyLoraConfig) and isinstance(config, PromptLearningConfig) and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config, adapter_name)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name) # NOTE 构造peft model
        model.load_adapter(model_id, adapter_name, **kwargs) # NOTE, model_id='bigscience/bloomz-560m_PREFIX_TUNING_CAUSAL_LM_epoch200', load trained parameter weights
        #import ipdb; ipdb.set_trace()
        return model

    def _setup_prompt_encoder(self, adapter_name):
        #import ipdb; ipdb.set_trace()
        config = self.peft_config[adapter_name]
        self.prompt_encoder = torch.nn.ModuleDict({}) # NOTE, network
        self.prompt_tokens = {} # NOTE virtual tokens
        transformer_backbone = None
        for name, module in self.base_model.named_children(): # 'transformer' and 'lm_head'
            if not config.peft_type == PeftType.LAZY_LORA: # lazy lora requires its lora adapter to be trainable, so we need to skip lazy lora here
                for param in module.parameters():
                    param.requires_grad = False
            if isinstance(module, PreTrainedModel): # <class 'transformers.modeling_utils.PreTrainedModel'>
                # Make sure to freeze Tranformers model, yes, in NOTE
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name # 'transformer'

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1
            if config.peft_type == PeftType.LAZY_LORA:
                if config.prompt_tuning_config is not None:
                    config.prompt_tuning_config.num_transformer_submodules = config.num_transformer_submodules
                    config.prompt_tuning_config.token_dim = config.token_dim
                    #config.num_virtual_tokens = config.prompt_tuning_config.num_virtual_tokens
                if config.prefix_tuning_config is not None:
                    config.prefix_tuning_config.num_transformer_submodules = config.num_transformer_submodules
                    config.prefix_tuning_config.token_dim = config.token_dim
                    config.prefix_tuning_config.num_layers = config.num_layers
                    config.prefix_tuning_config.num_attention_heads = config.num_attention_heads

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size: # [250880, 1024] for word embedding matrix, name='word_embeddings.weight'
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", "")) #   (word_embeddings): Embedding(250880, 1024) -> Embedding(250880, 1024)
                break
        #import ipdb; ipdb.set_trace()
        if config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings) 
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(config)
        elif config.peft_type == PeftType.LAZY_LORA:
            prompt_encoder = torch.nn.ModuleDict()
            if (config.prompt_tuning_config is not None) and (config.prompt_tuning_config.peft_type == PeftType.PROMPT_TUNING):
                prompt_embedding = PromptEmbedding(
                    config.prompt_tuning_config, 
                    self.word_embeddings
                ) # NOTE
                name1 = adapter_name + '_prompt_tuning'
                prompt_encoder[name1] = prompt_embedding

                self.prompt_tokens[name1] = torch.arange(
                    config.prompt_tuning_config.num_virtual_tokens * config.num_transformer_submodules # 30 * 1 = 30
                ).long() 
            if (config.prefix_tuning_config is not None) and (config.prefix_tuning_config.peft_type == PeftType.PREFIX_TUNING):
                prefix_encoder = PrefixEncoder(
                    config.prefix_tuning_config
                ) # NOTE
                name2 = adapter_name + '_prefix_tuning'
                prompt_encoder[name2] = prefix_encoder

                self.prompt_tokens[name2] = torch.arange(
                    config.prefix_tuning_config.num_virtual_tokens * config.num_transformer_submodules # 30 * 1 = 30
                ).long() 

            self.prompt_encoder.update(prompt_encoder) 
            # for lazy lora which uses both prompt tuning and prefix encoding

            return
        else:
            raise ValueError("Not supported")

        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder})) 
        # NOTE 这是把刚才初始化好的prompt_encoder，放入self.prompt_encoder里面去！

        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules # 30 * 1 = 30
        ).long() 
        # {'default': tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 
        # 10, 11, 12, 13, 14, 15, 16, 17, ..., 29])} 
        # 重要！这是30个新增加的虚拟的tokens for prefix-tuning; ||| 
        # {'default': tensor([0, 1, 2, 3, 4, 5, 6, 7])} for prompt fine-tuning

    def get_prompt_embedding_to_save(self, adapter_name):
        """
        Returns the prompt embedding to save when saving the model. Only applicable when `peft_config.peft_type !=
        PeftType.LORA`.
        """
        if adapter_name not in self.prompt_encoder:
            return None

        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (
            self.prompt_tokens[adapter_name].unsqueeze(0).expand(1, -1).to(prompt_encoder.embedding.weight.device)
        )
        if adapter_name in self.peft_config and self.peft_config[adapter_name].peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config[adapter_name].num_virtual_tokens]
        prompt_embeddings = prompt_encoder(prompt_tokens) # NOTE 保存prompt_encoder和prompt_embeddings啥区别？目前prompt tuning上，一样的啊... TODO
        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size, in_dtype=torch.float32, device=None):
        """
        Returns the virtual prompts to use for Peft. Only applicable when `peft_config.peft_type != PeftType.LORA`.
        """
        #import ipdb; ipdb.set_trace()
        peft_config = self.active_peft_config
        active_adapter = self.active_adapter
        if peft_config.peft_type == PeftType.LAZY_LORA:
            active_adapter = self.active_adapter + '_prompt_tuning'
        prompt_encoder = self.prompt_encoder[active_adapter] if active_adapter in self.prompt_encoder else None 
        # 'default' -> PrefixEncoder( (embedding): Embedding(30, 49152) ) NOTE prefix tuning
        # ||| [8, 1024] for prompt tuning 
        # ||| [20,1024] for virtual token embedding and then 3 linear layers of 1024-to-1024 (separated by 2 RELUs) for p-tuning
        prompt_tokens = (
            self.prompt_tokens[active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embedding.weight.device)
        ) if prompt_encoder is not None else None 
        # prefix-tuning: [B=8, L=30], 30个虚拟tokens, 0 to 29 
        # ||| [8,8], 8个虚拟的tokens, 0 to 7
        if peft_config.peft_type == PeftType.LAZY_LORA:
            # --- 1 prompt tuning part ---
            prompts = None # NOTE first return
            if prompt_encoder is not None:
                if peft_config.inference_mode:
                    prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
                else:
                    prompts = prompt_encoder(prompt_tokens) 
                    # NOTE, [8, 8] -> embedding(8, 1024) -> [8, 8, 1024], already in cuda:0 
                    # ||| p-tuning, [8, 20] -> [8, 20, 1024], embeding + 3 linear layers NOTE

            # --- 2 prefix tuning part ---
            active_adapter = self.active_adapter + '_prefix_tuning'
            prefix_encoder = self.prompt_encoder[active_adapter] if active_adapter in self.prompt_encoder else None
            if prefix_encoder is not None and device is not None:
                prefix_encoder.to(device)
            prefix_tokens = (
                self.prompt_tokens[active_adapter]
                .unsqueeze(0)
                .expand(batch_size, -1)
                .to(prefix_encoder.embedding.weight.device)
            ) if prefix_encoder is not None else None 

            past_key_values = None # NOTE second return
            if prefix_encoder is not None:
                #prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
                if peft_config.inference_mode:
                    past_key_values = prefix_encoder.embedding.weight.repeat(batch_size, 1, 1) 
                    # torch.Size([30, 49152]) 正是学习到的30个virtual tokens的对应的词嵌入向量
                else:
                    past_key_values = prefix_encoder(prefix_tokens) 
                    # NOTE, here, [8, 30] -> torch.Size([8, 30, 49152=24*2*1024])
                prefix_config = peft_config.prefix_tuning_config
                past_key_values = past_key_values.view(
                    batch_size, # 8
                    prefix_config.num_virtual_tokens, # 30
                    prefix_config.num_layers * 2, # 48
                    prefix_config.num_attention_heads, # 16
                    prefix_config.token_dim // prefix_config.num_attention_heads, # 1024/16=64
                ).to(in_dtype) 
                # [8, 30, 49152] -> [8, 30, 48, 16, 64]

                if peft_config.num_transformer_submodules == 2:
                    past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
                past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                    peft_config.num_transformer_submodules * 2
                ) 
                # [48, 8, 16, 30, 64], 2*24 for k,v; 
                # where, 8=batch-size; 16=head-num; 30=virtual-token; 64=head-dim NOTE -> 
                # 24 elements in the tuple, each is 
                # [2=k/v, 8=batch.size, 16=head.num, 30=len.virtual.token.num, 64=dim.1head]

                if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                    post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                    past_key_values = post_process_fn(past_key_values)
            #import ipdb; ipdb.set_trace()
            return prompts, past_key_values # for bloom, 24 * ( [128, 64, 30], [128, 30, 64] )
        elif peft_config.peft_type == PeftType.PREFIX_TUNING: # NOTE prefix tuning
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1) 
                # torch.Size([30, 49152]) 正是学习到的30个virtual tokens的对应的词嵌入向量
            else:
                past_key_values = prompt_encoder(prompt_tokens) 
                # NOTE, here, [8, 30] -> torch.Size([8, 30, 49152=24*2*1024])
            past_key_values = past_key_values.view(
                batch_size, # 8
                peft_config.num_virtual_tokens, # 30
                peft_config.num_layers * 2, # 48
                peft_config.num_attention_heads, # 16
                peft_config.token_dim // peft_config.num_attention_heads, # 1024/16=64
            ) 
            # [8, 30, 49152] -> [8, 30, 48, 16, 64]

            if peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                peft_config.num_transformer_submodules * 2
            ) 
            # [48, 8, 16, 30, 64], 2*24 for k,v; 
            # where, 8=batch-size; 16=head-num; 30=virtual-token; 64=head-dim NOTE -> 
            # 24 elements in the tuple, each is 
            # [2=k/v, 8=batch.size, 16=head.num, 30=len.virtual.token.num, 64=dim.1head]

            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            return past_key_values # for bloom, 24 * ( [128, 64, 30], [128, 30, 64] )
        else: # NOTE prompt tuning
            if peft_config.inference_mode:
                prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = prompt_encoder(prompt_tokens) 
                # NOTE, [8, 8] -> embedding(8, 1024) -> [8, 8, 1024], already in cuda:0 
                # ||| p-tuning, [8, 20] -> [8, 20, 1024], embeding + 3 linear layers NOTE
            return prompts

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        return self.get_base_model()(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        """
        Disables the adapter module.
        """
        try:
            if isinstance(self.peft_config, PromptLearningConfig): # TODO 
                old_forward = self.forward
                self.forward = self.base_model.forward
            else:
                self.base_model.disable_adapter_layers()
            yield
        finally:
            if isinstance(self.peft_config, PromptLearningConfig):
                self.forward = old_forward
            else:
                self.base_model.enable_adapter_layers()

    def get_base_model(self):
        """
        Returns the base model.
        """
        return self.base_model if isinstance(self.active_peft_config, PromptLearningConfig) else self.base_model.model

    def add_adapter(self, adapter_name, peft_config):
        #import ipdb; ipdb.set_trace()
        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )
        self.peft_config[adapter_name] = peft_config
        #import ipdb; ipdb.set_trace()
        if isinstance(peft_config, PromptLearningConfig):
            self._setup_prompt_encoder(adapter_name) # NOTE, for prompt-learning, prefix-learning, p-tuning, lazy-lora
        else:
            self.base_model.add_adapter(adapter_name, peft_config) # for lora, adalora...

        self.set_additional_trainable_modules(peft_config, adapter_name) # useless for prefix-tuning/prompt-tuning/p-tuning/lazylora

    def set_additional_trainable_modules(self, peft_config, adapter_name):
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)

    def load_adapter(self, model_id, adapter_name, is_trainable=False, **kwargs):
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING
        #import ipdb; ipdb.set_trace()
        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None)).peft_type
            ].from_pretrained(model_id, subfolder=kwargs.get("subfolder", None))
            if isinstance(peft_config, PromptLearningConfig) and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)

        # load weights if any
        path = os.path.join(model_id, kwargs["subfolder"]) if kwargs.get("subfolder", None) is not None else model_id

        if os.path.exists(os.path.join(path, WEIGHTS_NAME)):
            filename = os.path.join(path, WEIGHTS_NAME) # 'bigscience/bloomz-560m_PREFIX_TUNING_CAUSAL_LM_epoch200/adapter_model.bin' NOTE
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, subfolder=kwargs.get("subfolder", None))
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # load the weights into the model
        set_peft_model_state_dict(self, adapters_weights, adapter_name=adapter_name) # NOTE TODO
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto") # not in currently for prefix tuning TODO
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )
            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )
            hook = AlignDevicesHook(io_same_device=True)
            if isinstance(self.peft_config[adapter_name], PromptLearningConfig): # TODO
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        # Set model in evaluation mode to deactivate Dropout modules by default
        self.eval()

    def set_adapter(self, adapter_name):
        """
        Sets the active adapter.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        #import ipdb; ipdb.set_trace()
        if not isinstance(self.peft_config[adapter_name], PromptLearningConfig):
            self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]


class PeftModelForSequenceClassification(PeftModel):
    """
    Peft model for sequence classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForSequenceClassification
        >>> from peft import PeftModelForSequenceClassification, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "SEQ_CLS",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 768,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 12,
        ...     "num_layers": 12,
        ...     "encoder_hidden_size": 768,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForSequenceClassification(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
        ```
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {"classifier", "score"}
        else:
            self.modules_to_save.update({"classifier", "score"})

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        peft_config = self.active_peft_config
        #import ipdb; ipdb.set_trace()
        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size, in_dtype=inputs_embeds.dtype)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            pooled_output = outputs[1] if len(outputs) > 1 else outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                pooled_output = self.base_model.dropout(pooled_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(pooled_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.base_model.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.base_model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.base_model.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.base_model.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


class PeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.


    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModelForCausalLM, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "CAUSAL_LM",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 1280,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 20,
        ...     "num_layers": 36,
        ...     "encoder_hidden_size": 1280,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> peft_model = PeftModelForCausalLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
        ```
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation # method

    def forward(
        self,
        input_ids=None, # [8, 64]
        attention_mask=None, # [8, 64]
        inputs_embeds=None,
        labels=None, # [8, 64]
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        #import ipdb; ipdb.set_trace()
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(  # NOTE for LoRA, AdaLoRA, call base_model's forward func directly
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs, # {}
            ) # e.g., > /opt/conda/lib/python3.8/site-packages/transformers/models/bloom/modeling_bloom.py(876)forward()

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            num_virtual_tokens = peft_config.num_virtual_tokens
            if peft_config.peft_type == PeftType.LAZY_LORA:
                num_virtual_tokens = 0
                if peft_config.prompt_tuning_config is not None:
                    num_virtual_tokens = peft_config.prompt_tuning_config.num_virtual_tokens
                    peft_config.num_virtual_tokens = num_virtual_tokens # for input/output seq len extending
                if peft_config.prefix_tuning_config is not None:
                    num_virtual_tokens += peft_config.prefix_tuning_config.num_virtual_tokens # for middle layers only, not for input/output layers
            if num_virtual_tokens is not None:
                prefix_attention_mask = torch.ones(
                    batch_size, num_virtual_tokens
                ).to(attention_mask.device) 
                # prefix tuning: all 1, shape=torch.Size([8, 30]), 
                # 每个序列前面增加30个虚拟tokens NOTE 
                # ||| prompt tuning, [8, 8] 是在prompt前面增加8个tokens NOTE 
                # ||| p-tuning, [8, 20] all 1
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1) 
                # [8, 30] + [8, 64] -> [8, 94=64+30] ||| 
                # [8, 8] + [8, 64] -> [8, 72] ||| p-tuning [8, 20] + [8, 64] -> [8, 84]

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask, # torch.Size([8, 94]) ||| [8, 72] ||| [8, 84]
                "labels": labels, # torch.Size([8, 64]) ||| [8, 64] ||| [8, 64]
                "output_attentions": output_attentions, # None
                "output_hidden_states": output_hidden_states, # None
                "return_dict": return_dict, # None
            }
        )
        #import ipdb; ipdb.set_trace()
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            #import ipdb; ipdb.set_trace() # NOTE
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, past_key_values=past_key_values, **kwargs
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids) 
                # NOTE ||| [8, 64] -> [8, 64, 1024]
            # concat prompt labels
            if labels is not None and peft_config.num_virtual_tokens is not None and peft_config.num_virtual_tokens > 0:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device) 
                #prefix_labels = torch.full((batch_size, num_virtual_tokens), -100).to(labels.device) 
                # ||| [8,8] 个 -100 ||| [8,20]个-100
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1) 
                # ||| [8, 72] prefix + labels ||| [8, 84]
            prompts = self.get_prompt(
                batch_size=batch_size, 
                in_dtype=inputs_embeds.dtype,
                device=inputs_embeds.device
            ) # ||| [8, 8, 1024] 8个虚拟的tokens, 0 to 7, 然后用prompt embedding (8, 1024)给embed了一下，就得到最后的张量[8, 8, 1024]
            if isinstance(prompts, tuple):
                #import ipdb; ipdb.set_trace()
                # for lazy lora with (1) prompt tuning and (2) prefix tuning
                prompts_in, past_key_values = prompts
                if prompts_in is not None:
                    prompts_in = prompts_in.to(inputs_embeds.dtype)
                    inputs_embeds = torch.cat((prompts_in, inputs_embeds), dim=1) 
                return self.base_model(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    **kwargs
                )
            else:
                prompts = prompts.to(inputs_embeds.dtype)
                inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1) 
                # ||| [8, 8, 1024] + [8, 64, 1024] -> [8, 72, 1024] 
                # ||| [8, 20, 1024] + [8, 64, 1024] -> [8, 84, 1024]
                return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, **kwargs): # NOTE, dict_keys(['input_ids'=[8,64], 'attention_mask'=[8,64], 'max_new_tokens'=10, 'eos_token_id'=3]) ||| input_ids=[1,43], attention_mask=[1,43], max_new_tokens=10, eos_token_id=3
        #import ipdb; ipdb.set_trace()
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation # NOTE important!
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
            if peft_config.peft_type in [PeftType.LAZY_LORA]:
                self.base_model.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation # NOTE important! to ensure lazy lora uses prompt-tuning's prompt-embedding 
        else:
            self.base_model.generation_config = self.generation_config
        try:
            #import ipdb; ipdb.set_trace()
            if not isinstance(peft_config, PromptLearningConfig):
                outputs = self.base_model.generate(**kwargs)
            else: # include: lazy lora 
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Peft model generation")
                # For gpt2 models, we construct postion_ids on the fly by using attention mask, and position ids need to match input_shape.
                # for prefix tuning, input shape is determined using `input_ids`. Thus we should not expand 'attention_mask' here
                # for prompt tuning input_ids is not passed but a concatenated input_embeds is passed. Thus attention_mask needs to be of same size of num_virtual_tokens + input_ids
                if kwargs.get("attention_mask", None) is not None and peft_config.peft_type in [
                    PeftType.PROMPT_TUNING,
                    PeftType.P_TUNING, # TODO why not include LAZY_LORA?
                ]:
                    # concat prompt attention mask
                    prefix_attention_mask = torch.ones(
                        kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                    ).to(kwargs["input_ids"].device)
                    kwargs["attention_mask"] = torch.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1) # ||| [1,8]+[1,43]=[1,51] 

                if kwargs.get("position_ids", None) is not None:
                    warnings.warn(
                        "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                    )
                    kwargs["position_ids"] = None
                if kwargs.get("token_type_ids", None) is not None:
                    warnings.warn(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None
                #import ipdb; ipdb.set_trace() # NOTE
                outputs = self.base_model.generate(**kwargs) # > /usr/local/lib/python3.8/dist-packages/transformers/generation/utils.py(1145)generate() NOTE, input_ids=[1,43], attention_mask=[1,51],max_new_tokens=10,eos_token_id=3 --> outputs.shape=[1,53] TODO why 53 ???
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        #import ipdb; ipdb.set_trace() # NOTE transformers/generation/utils.py里面的greedy_search，会回调这个方法的 TODO
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        #import ipdb; ipdb.set_trace()
        if isinstance(peft_config, PromptLearningConfig):
            if peft_config.peft_type == PeftType.PREFIX_TUNING:
                prefix_attention_mask = torch.ones(
                    model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                ).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )
            elif peft_config.peft_type == PeftType.LAZY_LORA:
                num_virtual_tokens = 0
                # --- the prompt tuning part for Lazy Lora ---
                if peft_config.prompt_tuning_config is not None:
                    num_virtual_tokens = peft_config.prompt_tuning_config.num_virtual_tokens
                    if num_virtual_tokens > 0:
                        if model_kwargs.get('position_ids', None) is not None:
                            warnings.warn(
                                'Position ids are not supported for PEFT lazy lora with prompt tuning. Ignoring position ids.'
                            )
                            model_kwargs['position_ids'] = None # TODO 
                        if model_kwargs.get('token_type_ids', None) is not None:
                            warnings.warn(
                                'Token type ids are not supported for PEFT lazy lora with prompt tuning. Ignoring token type ids.'
                            )
                            model_kwargs['token_type_ids'] = None
                # --- the prefix tuning part for Lazy Lora ---
                if peft_config.prefix_tuning_config is not None:
                    num_virtual_tokens += peft_config.prefix_tuning_config.num_virtual_tokens
                prefix_attention_mask = torch.ones(
                    model_kwargs["input_ids"].shape[0], num_virtual_tokens
                ).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])

                if self.base_model_torch_dtype is not None:
                    # handle the case for Bloom where it outputs tuple of tuples
                    if isinstance(past_key_values[0], tuple):
                        past_key_values = tuple(
                            tuple(
                                past_key_value.to(self.base_model_torch_dtype)
                                for past_key_value in past_key_value_tuple
                            )
                            for past_key_value_tuple in past_key_values
                        )
                    else:
                        past_key_values = tuple(
                            past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_values
                        )

                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None: # NOTE for prompt tuning
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"]) # torch.Size([1, 43, 1024])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], in_dtype=inputs_embeds.dtype) # NOTE, torch.Size([1, 8, 1024])
                    if isinstance(prompts, tuple):
                        prompts_in, past_key_values = prompts
                        prompts_in = prompts_in.to(inputs_embeds.dtype)
                        model_kwargs["inputs_embeds"] = torch.cat((prompts_in, inputs_embeds), dim=1) 
                        model_kwargs["input_ids"] = None
                        model_kwargs["past_key_values"] = past_key_values
                    else:
                        prompts = prompts.to(inputs_embeds.dtype) 
                        model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1) # [1,8,1024] + [1,43,1024] -> torch.Size([1, 51, 1024])
                        model_kwargs["input_ids"] = None
        #import ipdb; ipdb.set_trace()
        return model_kwargs


class PeftModelForSeq2SeqLM(PeftModel):
    """
    Peft model for sequence-to-sequence language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.


    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import PeftModelForSeq2SeqLM, get_peft_config

        >>> config = {
        ...     "peft_type": "LORA",
        ...     "task_type": "SEQ_2_SEQ_LM",
        ...     "inference_mode": False,
        ...     "r": 8,
        ...     "target_modules": ["q", "v"],
        ...     "lora_alpha": 32,
        ...     "lora_dropout": 0.1,
        ...     "merge_weights": False,
        ...     "fan_in_fan_out": False,
        ...     "enable_lora": None,
        ...     "bias": "none",
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> peft_model = PeftModelForSeq2SeqLM(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
        ```
    """

    def __init__(self, model, peft_config: PeftConfig, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        #import ipdb; ipdb.set_trace()
        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if decoder_attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                decoder_attention_mask.device
            )
            decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, decoder_input_ids=decoder_input_ids, past_key_values=past_key_values, **kwargs
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            if decoder_inputs_embeds is None and decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                    attention_mask.device
                )
                kwargs["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # concat prompt labels
            if labels is not None:
                if peft_config.num_transformer_submodules == 1:
                    kwargs["labels"] = labels
                elif peft_config.num_transformer_submodules == 2:
                    prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                    kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
            if peft_config.num_transformer_submodules == 1:
                return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
            elif peft_config.num_transformer_submodules == 2:
                decoder_inputs_embeds = torch.cat(
                    (prompts[:, peft_config.num_virtual_tokens :], decoder_inputs_embeds), dim=1
                )
                return self.base_model(
                    inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, **kwargs
                )

    def generate(self, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self._prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            #import ipdb; ipdb.set_trace()
            if not isinstance(peft_config, PromptLearningConfig):
                outputs = self.base_model.generate(**kwargs)
            else:
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Peft model generation")
                if kwargs.get("position_ids", None) is not None:
                    warnings.warn(
                        "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                    )
                    kwargs["position_ids"] = None
                if kwargs.get("token_type_ids", None) is not None:
                    warnings.warn(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None

                if peft_config.peft_type == PeftType.PREFIX_TUNING:
                    outputs = self.base_model.generate(**kwargs)
                else:
                    raise NotImplementedError
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        #import ipdb; ipdb.set_trace()
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
            batch_size = model_kwargs["decoder_input_ids"].shape[0]
            past_key_values = self.get_prompt(batch_size)
            if self.base_model_torch_dtype is not None:
                # handle the case for Bloom where it outputs tuple of tuples
                if isinstance(past_key_values[0], tuple):
                    past_key_values = tuple(
                        tuple(
                            past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_value_tuple
                        )
                        for past_key_value_tuple in past_key_values
                    )
                else:
                    past_key_values = tuple(
                        past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_values
                    )
            model_kwargs["past_key_values"] = past_key_values

        return model_kwargs


class PeftModelForTokenClassification(PeftModel):
    """
    Peft model for token classification tasks.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    **Attributes**:
        - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.
        - **cls_layer_name** (`str`) -- The name of the classification layer.

    Example:

        ```py
        >>> from transformers import AutoModelForSequenceClassification
        >>> from peft import PeftModelForTokenClassification, get_peft_config

        >>> config = {
        ...     "peft_type": "PREFIX_TUNING",
        ...     "task_type": "TOKEN_CLS",
        ...     "inference_mode": False,
        ...     "num_virtual_tokens": 20,
        ...     "token_dim": 768,
        ...     "num_transformer_submodules": 1,
        ...     "num_attention_heads": 12,
        ...     "num_layers": 12,
        ...     "encoder_hidden_size": 768,
        ...     "prefix_projection": False,
        ...     "postprocess_past_key_value_function": None,
        ... }

        >>> peft_config = get_peft_config(config)
        >>> model = AutoModelForTokenClassification.from_pretrained("bert-base-cased")
        >>> peft_model = PeftModelForTokenClassification(model, peft_config)
        >>> peft_model.print_trainable_parameters()
        trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
        ```
    """

    def __init__(self, model, peft_config: PeftConfig = None, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {"classifier", "score"}
        else:
            self.modules_to_save.update({"classifier", "score"})

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #import ipdb; ipdb.set_trace()
        if not isinstance(peft_config, PromptLearningConfig):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            return self._prefix_tuning_forward(input_ids=input_ids, **kwargs)
        else:
            if kwargs.get("token_type_ids", None) is not None:
                kwargs["token_type_ids"] = torch.cat(
                    (
                        torch.zeros(batch_size, peft_config.num_virtual_tokens).to(self.word_embeddings.weight.device),
                        kwargs["token_type_ids"],
                    ),
                    dim=1,
                ).long()
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            prompts = self.get_prompt(batch_size=batch_size, in_dtype=inputs_embeds.dtype)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def _prefix_tuning_forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size)
        fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
        kwargs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "inputs_embeds": inputs_embeds,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
            }
        )
        if "past_key_values" in fwd_params:
            return self.base_model(labels=labels, **kwargs)
        else:
            transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
            fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
            if "past_key_values" not in fwd_params:
                raise ValueError("Model does not support past key values which are required for prefix tuning.")
            outputs = transformer_backbone_name(**kwargs)
            sequence_output = outputs[0]
            if "dropout" in [name for name, _ in list(self.base_model.named_children())]:
                sequence_output = self.base_model.dropout(sequence_output)
            logits = self.base_model.get_submodule(self.cls_layer_name)(sequence_output)

            loss = None
            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
