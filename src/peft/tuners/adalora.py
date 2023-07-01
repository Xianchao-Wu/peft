import importlib
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import (
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
)


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class AdaLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.AdaLora`].

    Args:
        target_r (`int`): The target average rank of incremental matrix.
        init_r (`int`): The initial rank for each incremental matrix.
        tinit (`int`): The steps of initial fine-tuning warmup.
        tfinal (`int`): The step of final fine-tuning.
        deltaT (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        orth_reg_weight (`float`): The coefficient of orthogonal regularization.
        total_step (`int`): The total training steps that should be specified before training.
        rank_pattern (`list`): The allocated rank for each weight matrix by RankAllocator.
    """

    target_r: int = field(default=8, metadata={"help": "Target Lora matrix dimension."})
    init_r: int = field(default=12, metadata={"help": "Intial Lora matrix dimension."})
    tinit: int = field(default=0, metadata={"help": "The steps of initial warmup."})
    tfinal: int = field(default=0, metadata={"help": "The steps of final warmup."})
    deltaT: int = field(default=1, metadata={"help": "Step interval of rank allocation."})
    beta1: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    beta2: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    orth_reg_weight: float = field(default=0.5, metadata={"help": "The orthogonal regularization coefficient."})
    total_step: Optional[int] = field(default=None, metadata={"help": "The total training steps."})
    rank_pattern: Optional[dict] = field(default=None, metadata={"help": "The saved rank pattern."})

    def __post_init__(self):
        self.peft_type = PeftType.ADALORA


class AdaLoraModel(LoraModel):
    """
    Creates AdaLoRA (Adaptive LoRA) model from a pretrained transformers model. Paper:
    https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.

    Returns:
        `torch.nn.Module`: The AdaLora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import AdaLoraModel, AdaLoraConfig
        >>> config = AdaLoraConfig(
                peft_type="ADALORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.01,
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> model = AdaLoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`AdaLoraConfig`]): The configuration of the AdaLora model.
    """

    def __init__(self, model, config, adapter_name): # {'default': AdaLoraConfig(peft_type=<PeftType.ADALORA: 'ADALORA'>, base_model_name_or_path='bigscience/bloomz-560m', task_type='CAUSAL_LM', inference_mode=False, r=8, target_modules=['query_key_value'], lora_alpha=32, lora_dropout=0.01, fan_in_fan_out=False, bias='none', modules_to_save=None, init_lora_weights=True, target_r=8, init_r=12, tinit=0, tfinal=0, deltaT=1, beta1=0.85, beta2=0.85, orth_reg_weight=0.5, total_step=None, rank_pattern=None)} = config; 'default' = adapter_name
        nn.Module.__init__(self)
        self.model = model
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name]) # NOTE

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_adalora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name) # NOTE
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "AdaLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "AdaLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name
            self.rankallocator = RankAllocator(self.model, self.peft_config[adapter_name], self.trainable_adapter_name) # NOTE for what? 重新为每一层分类budget of rank (r), <peft.tuners.adalora.RankAllocator object at 0x7f6902308940>

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False) # False
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.init_r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        } # {'r': 12, 'lora_alpha': 32, 'lora_dropout': 0.01, 'fan_in_fan_out': False, 'init_lora_weights': True}
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key) # parent=transformer.h.0.self_attention; target='transformer.h.0.self_attention.query_key_value'; target_name=
                bias = target.bias is not None # True
                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.init_r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = SVDLinear8bitLt(
                            adapter_name, target.in_features, target.out_features, bias=bias, **kwargs
                        )
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features # NOTE here, 1024, 3072
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = SVDLinear(adapter_name, in_features, out_features, bias=bias, **kwargs) # NOTE

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs) # odict_keys(['loss', 'logits', 'past_key_values'])

        # Calculate the orthogonal regularization
        orth_reg_weight = self.peft_config[self.trainable_adapter_name].orth_reg_weight # 0.5
        assert orth_reg_weight > 0

        if hasattr(outputs, "loss"):
            regu_loss = 0
            num_param = 0
            for n, p in self.model.named_parameters():
                if ("lora_A" in n or "lora_B" in n) and self.trainable_adapter_name in n: # 'default' in a module NOTE
                    para_cov = p @ p.T if "lora_A" in n else p.T @ p # torch.Size([12, 12]), [12, 1024]*[1024, 12] -> [12, 12]
                    I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov)) # 单位矩阵，I, shape=12*12
                    I.requires_grad = False
                    num_param += 1
                    regu_loss += torch.norm(para_cov - I, p="fro") # NOTE important, regularization loss; alike MSE loss, L2-norm: ||p - I || ^ 2 = sqrt(\sum(xi^2))
            regu_loss = regu_loss / num_param
            outputs.loss += orth_reg_weight * regu_loss # NOTE 这是对outputs的loss进行了更近，然后呢??? TODO
        return outputs

    def resize_modules_by_rank_pattern(self, rank_pattern, adapter_name):
        #import ipdb; ipdb.set_trace()
        lora_config = self.peft_config[adapter_name]
        for name, rank_idx in rank_pattern.items():
            if isinstance(rank_idx, list):
                rank = sum(rank_idx)
            elif isinstance(rank_idx, torch.Tensor):
                rank_idx = rank_idx.view(-1)
                rank = rank_idx.sum().item()
            else:
                raise ValueError("Unexcepted type of rank_idx")
            key = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
            _, target, _ = _get_submodules(self.model, key)
            lora_E_weights = target.lora_E[adapter_name][rank_idx]
            lora_A_weights = target.lora_A[adapter_name][rank_idx]
            lora_B_weights = target.lora_B[adapter_name][:, rank_idx]
            ranknum = target.ranknum[adapter_name]
            target.update_layer(
                adapter_name,
                rank,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
            with torch.no_grad():
                if rank > 0:
                    target.lora_E[adapter_name].copy_(lora_E_weights)
                    target.lora_A[adapter_name].copy_(lora_A_weights)
                    target.lora_B[adapter_name].copy_(lora_B_weights)
                    # The scaling is exactly as the previous
                    target.ranknum[adapter_name].copy_(ranknum)

    def resize_state_dict_by_rank_pattern(self, rank_pattern, state_dict, adapter_name):
        #import ipdb; ipdb.set_trace()
        for name, rank_idx in rank_pattern.items():
            rank = sum(rank_idx) # e.g., rank_idx=[False, False, False, False, False, True, False, False, False, False, False, False]
            print('resize_state_by_rank_pattern: rank_idx={}, rank={}'.format(rank_idx, rank))
            prefix = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
            for layer in ["lora_E", "lora_A", "lora_B"]:
                key = f"base_model.model.{prefix}.{layer}.{adapter_name}"
                if layer != "lora_B":
                    state_dict[key] = (
                        state_dict[key][rank_idx] if rank != state_dict[key].shape[0] else state_dict[key]
                    ) # [rank_idx] -> 只保留那些没有被mask掉的位置
                else:
                    state_dict[key] = (
                        state_dict[key][:, rank_idx] if rank != state_dict[key].shape[1] else state_dict[key]
                    )
        return state_dict

    def update_and_allocate(self, global_step):
        #import ipdb; ipdb.set_trace()
        lora_config = self.peft_config[self.trainable_adapter_name]
        # Update the importance score and allocate the budget
        if global_step < lora_config.total_step - lora_config.tfinal:
            _, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step)
            if rank_pattern:
                lora_config.rank_pattern = rank_pattern
        # Finalize the budget allocation
        elif global_step == lora_config.total_step - lora_config.tfinal:
            _, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step, force_mask=True)
            # for some reason, this freezes the trainable parameters and nothing gets updates
            # self.resize_modules_by_rank_pattern(rank_pattern, self.trainable_adapter_name)
            lora_config.rank_pattern = rank_pattern
            self.rankallocator.reset_ipt()
        # Currently using inefficient way to mask the unimportant weights using the rank pattern
        #  due to problem mentioned above
        elif global_step > lora_config.total_step - lora_config.tfinal:
            self.rankallocator.mask_using_rank_pattern(self.model, lora_config.rank_pattern)
        # Pass the function and do forward propagation
        else:
            return None

    @staticmethod
    def _prepare_adalora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config # did nothing at training, AdaLoraConfig(peft_type=<PeftType.ADALORA: 'ADALORA'>, base_model_name_or_path='bigscience/bloomz-560m', task_type='CAUSAL_LM', inference_mode=False, r=8, target_modules=['query_key_value'], lora_alpha=32, lora_dropout=0.01, fan_in_fan_out=False, bias='none', modules_to_save=None, init_lora_weights=True, target_r=8, init_r=12, tinit=0, tfinal=0, deltaT=1, beta1=0.85, beta2=0.85, orth_reg_weight=0.5, total_step=None, rank_pattern=None)


class AdaLoraLayer(LoraLayer):
    def __init__(
        self,
        in_features: int, # 1024
        out_features: int, # 3072
    ):
        super().__init__(in_features, out_features)
        self.lora_E = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.ranknum = nn.ParameterDict({})

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:

            def lora_dropout_layer(x):
                return x

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        # Right singular vectors, e.g., ParameterDict(  (default): Parameter containing: [torch.FloatTensor of size 12x1024]) NOTE
        self.lora_A.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros(r, self.in_features))}))
        # Singular values, e.g., ParameterDict(  (default): Parameter containing: [torch.FloatTensor of size 12x1]) NOTE
        self.lora_E.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros(r, 1))}))
        # Left singular vectors, e.g., ParameterDict(  (default): Parameter containing: [torch.FloatTensor of size 3072x12]) NOTE
        self.lora_B.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros(self.out_features, r))}))
        # The current rank
        self.ranknum.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.zeros(1), requires_grad=False)}))
        self.ranknum[adapter_name].data.fill_(float(r))
        self.ranknum[adapter_name].requires_grad = False # ParameterDict(  (default): Parameter containing: [torch.FloatTensor of size 1])
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r) # {'default': 32}
        if init_lora_weights: # True
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device) # currently, 'cpu'

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.zeros_(self.lora_E[adapter_name])
            nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)


class SVDLinear(nn.Linear, AdaLoraLayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        adapter_name: str, # 'default'
        in_features: int, # 1024
        out_features: int, # 3072
        r: int = 0, # 12 NOTE, used init_r
        lora_alpha: int = 1, # 32
        lora_dropout: float = 0.0, # 0.01
        fan_in_fan_out: bool = False, # False
        **kwargs, # {'bias': True, 'init_lora_weights': True}
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs) # SVDLinear(in_features=1024, out_features=3072, bias=True)
        AdaLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights) # NOTE
        self.active_adapter = adapter_name # 'default'

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[self.active_adapter]
                    @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]),
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
                / (self.ranknum[self.active_adapter] + 1e-5)
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[self.active_adapter]
                    @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter])
                )
                * self.scaling[self.active_adapter]
                / (self.ranknum[self.active_adapter] + 1e-5)
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            result += (
                (
                    self.lora_dropout[self.active_adapter](x)
                    @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]).T
                    @ self.lora_B[self.active_adapter].T
                )
                * self.scaling[self.active_adapter]
                / (self.ranknum[self.active_adapter] + 1e-5)
            )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return result


if is_bnb_available():

    class SVDLinear8bitLt(bnb.nn.Linear8bitLt, AdaLoraLayer):
        # Low-rank matrix for SVD-based adaptation
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
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
            AdaLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        (
                            self.lora_dropout[self.active_adapter](x)
                            @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]).T
                            @ self.lora_B[self.active_adapter].T
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                        / (self.ranknum[self.active_adapter] + 1e-5)
                    )
                else:
                    output = (
                        (
                            self.lora_dropout[self.active_adapter](x)
                            @ (self.lora_A[self.active_adapter] * self.lora_E[self.active_adapter]).T
                            @ self.lora_B[self.active_adapter].T
                        )
                        * self.scaling[self.active_adapter]
                        / (self.ranknum[self.active_adapter] + 1e-5)
                    )
                result += output
            return result


class RankAllocator(object):
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1 # 0.85
        self.beta2 = peft_config.beta2 # 0.85
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.init_bgt = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                self.init_bgt += p.size(0)
                self.name_set.add(n.replace("lora_A", "%s")) # {'transformer.h.0.self_attention.query_key_value.%s.default'}
        self.name_set = sorted(self.name_set) # 24layers * 12 = 288 = self.init_bgt; ['transformer.h.0.self_attention.query_key_value.%s.default', 'transformer.h.1.self_attention.query_key_value.%s.default', 'transformer.h.10.self_attention.query_key_value.%s.default', 'transformer.h.11.self_attention.query_key_value.%s.default', 'transformer.h.12.self_attention.query_key_value.%s.default', 'transformer.h.13.self_attention.query_key_value.%s.default', 'transformer.h.14.self_attention.query_key_value.%s.default', 'transformer.h.15.self_attention.query_key_value.%s.default', 'transformer.h.16.self_attention.query_key_value.%s.default', 'transformer.h.17.self_attention.query_key_value.%s.default', 'transformer.h.18.self_attention.query_key_value.%s.default', 'transformer.h.19.self_attention.query_key_value.%s.default', 'transformer.h.2.self_attention.query_key_value.%s.default', 'transformer.h.20.self_attention.query_key_value.%s.default', 'transformer.h.21.self_attention.query_key_value.%s.default', 'transformer.h.22.self_attention.query_key_value.%s.default', 'transformer.h.23.self_attention.query_key_value.%s.default', 'transformer.h.3.self_attention.query_key_value.%s.default', 'transformer.h.4.self_attention.query_key_value.%s.default', 'transformer.h.5.self_attention.query_key_value.%s.default', 'transformer.h.6.self_attention.query_key_value.%s.default', 'transformer.h.7.self_attention.query_key_value.%s.default', 'transformer.h.8.self_attention.query_key_value.%s.default', 'transformer.h.9.self_attention.query_key_value.%s.default']
        # The total final rank budget
        self.target_bgt = self.peft_config.target_r * len(self.name_set) # 8*24=192 for the target budget

    def budget_schedule(self, step: int):
        #import ipdb; ipdb.set_trace()
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        # Initial warmup
        if step <= tinit:
            budget = self.init_bgt
            mask_ind = False
        # Final fine-tuning
        elif step > total_step - tfinal:
            budget = self.target_bgt
            mask_ind = True
        else:
            # Budget decreasing with a cubic scheduler
            mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
            budget = int((self.init_bgt - self.target_bgt) * (mul_coeff**3) + self.target_bgt)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return budget, mask_ind

    def update_ipt(self, model):
        #import ipdb; ipdb.set_trace()
        # Update the sensitivity and uncertainty for every weight
        for n, p in model.named_parameters():
            if "lora_" in n and self.adapter_name in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    self.ipt[n] = (p * p.grad).abs().detach() # Equation (8) in 2303.10512 paper adalora
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n] # Equation (9) in 2303.10512 paper, adalora
                    # Uncertainty quantification
                    self.exp_avg_unc[n] = (
                        self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    ) # NOTE Equation (10) in 2303.10512 paper, adalora

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n] # Equation (11) in 2303.10512 paper, adalora

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False) # [12, 2] -> [12, 1]
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1) # [12, 1], 就是对三者的简单求和
        return sum_ipt

    def mask_to_budget(self, model, budget):
        #import ipdb; ipdb.set_trace()
        value_ipt = {}
        vector_ipt = {}
        triplet_ipt = {}
        # Get the importance score for A, E, B
        for n, p in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True) # [12, 768] -> [12, 1]
                name_m = n.replace("lora_A", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_B.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1) # [768, 12] -> [12, 1]
                name_m = n.replace("lora_B", "%s") # [12, 1]
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"lora_E.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n) # [12, 1]
                name_m = n.replace("lora_E", "%s")
                value_ipt[name_m] = entry_ipt

        all_score = []
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_E = value_ipt[name_m] # [12, 1]
            ipt_AB = torch.cat(vector_ipt[name_m], dim=1) # [12, 2]
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB) # NOTE 
            name_E = name_m % "lora_E" # 'model.encoder.layers.0.self_attn.k_proj.%s.default', 'lora_E' -> 'model.encoder.layers.0.self_attn.k_proj.lora_E.default' NOTE
            triplet_ipt[name_E] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

        # Get the threshold by ranking ipt
        mask_threshold = torch.kthvalue( # NOTE  k th smallest element, sort automatically
            torch.cat(all_score),
            k=self.init_bgt - budget, # 1152-1147=5
        )[0].item() # e.g., 1.172337834448589e-12=mask_threshold

        rank_pattern = {}
        # Mask the unimportant triplets
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    p.masked_fill_(triplet_ipt[n] <= mask_threshold, 0.0) # 这是把E的<=遮掩阈值的地方，赋值为0.0，即该位置不再起作用了 NOTE
                    rank_pattern[n] = (~(triplet_ipt[n] <= mask_threshold)).view(-1).tolist()
        return rank_pattern # len=96, e.g.,  'model.decoder.layers.0.self_attn.k_proj.lora_E.default': [True, True, True, True, True, False, True, True, True, True, True, False]

    def update_and_allocate(self, model, global_step, force_mask=False):
        #import ipdb; ipdb.set_trace()
        # # Update the importance score and allocate the budget
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
        else:
            rank_pattern = None
        return budget, rank_pattern

    def mask_using_rank_pattern(self, model, rank_pattern):
        #import ipdb; ipdb.set_trace()
        # Mask the unimportant triplets
        is_adapter_name_truncated = False
        if self.adapter_name not in next(iter(rank_pattern.keys())):
            is_adapter_name_truncated = True

        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"lora_E.{self.adapter_name}" in n:
                    key = n if not is_adapter_name_truncated else n.replace(f".{self.adapter_name}", "")
                    mask = torch.Tensor(rank_pattern[key]).unsqueeze(-1).to(p.device)
                    print(n, 'keep.dim=', mask.sum().item())
                    p.masked_fill_(~mask.bool(), 0.0)
                    # 'False' -> 0.0, and 'True' -> keep no change NOTE
