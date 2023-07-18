import sys
sys.path.insert(1, '/workspace/asr/peft/src')
import os
import torch

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
model_name_or_path = 'bigscience/bloomz-560m'
cache_dir = os.getcwd() # change this

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
    quantization_config=bnb_config, # 4-bit, change this
    device_map={"":0}, # gpu0, change this
    cache_dir=cache_dir)

import ipdb; ipdb.set_trace()

from peft import (LazyLoraConfig, get_peft_model, 
    PrefixTuningConfig, TaskType, PeftType, 
    PromptTuningConfig, PromptTuningInit)

peft_config_prompt_tuning = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)

peft_config_prefix_tuning = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=30
)

config_lazy_lora = LazyLoraConfig(
    r=8,
    is_r_by_svd=True,
    lazy_lora_alpha=32,
    lazy_pre_lora_alpha=0.1,
    lazy_pre_adapter_type='linear', #'linear', 'conv1d', 'none'
    target_modules=['query_key_value', 'dense_h_to_4h', 'dense_4h_to_h'],
    lazy_lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_config=peft_config_prompt_tuning,
    prefix_tuning_config=peft_config_prefix_tuning,
)
model = get_peft_model(model, config_lazy_lora)
model.print_trainable_parameters()

