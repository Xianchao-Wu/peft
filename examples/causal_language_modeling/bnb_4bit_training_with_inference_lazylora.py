# -*- coding: utf-8 -*-
"""bnb-4bit-training-with-inference.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Vvju5kOyBsDr7RX_YAvp6ZsSOoSMjhKD

# `transformers` meets `bitsandbytes` for democratzing Large Language Models (LLMs) through 4bit quantization

<center>
<img src="https://github.com/huggingface/blog/blob/main/assets/96_hf_bitsandbytes_integration/Thumbnail_blue.png?raw=true" alt="drawing" width="700" class="center"/>
</center>

Welcome to this notebook that goes through the recent `bitsandbytes` integration that includes the work from XXX that introduces no performance degradation 4bit quantization techniques, for democratizing LLMs inference and training.

In this notebook, we will learn together how to load a large model in 4bit (`gpt-neo-x-20b`) and train it using Google Colab and PEFT library from Hugging Face 🤗.

[In the general usage notebook](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing), you can learn how to propely load a model in 4bit with all its variants. 

If you liked the previous work for integrating [*LLM.int8*](https://arxiv.org/abs/2208.07339), you can have a look at the [introduction blogpost](https://huggingface.co/blog/hf-bitsandbytes-integration) to lean more about that quantization method.
"""

#!pip install -q -U bitsandbytes
#!pip install -q -U git+https://github.com/huggingface/transformers.git 
#!pip install -q -U git+https://github.com/huggingface/peft.git
#!pip install -q -U git+https://github.com/huggingface/accelerate.git
#!pip install -q datasets

"""First let's load the model we are going to use - GPT-neo-x-20B!
Note that the model itself is around 40GB in half precision"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "EleutherAI/gpt-neox-20b"
#model_id = "EleutherAI/gpt-j-6b"
#import ipdb; ipdb.set_trace()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # NOTE, 4bit, bnb = bits and bytes
    bnb_4bit_use_double_quant=True, 
    # NOTE double quantization, in the paper of https://arxiv.org/abs/2305.14314
    bnb_4bit_quant_type="nf4", # NOTE
    bnb_4bit_compute_dtype=torch.bfloat16 # NOTE
) # 这是四个配置，放一起，都在working TODO

tokenizer = AutoTokenizer.from_pretrained(model_id, 
        cache_dir='/workspace/asr/Huatuo-Llama-Med-Chinese')

model = AutoModelForCausalLM.from_pretrained(model_id, 
        quantization_config=bnb_config, # NOTE 这个是最重要的部分，是使用4bit导入pretrained model 
        #device_map={"":0},
        device_map='auto',
        cache_dir='/workspace/asr/Huatuo-Llama-Med-Chinese')

"""Then we have to apply some preprocessing to the model to prepare it for training. 
For that use the `prepare_model_for_kbit_training` method from PEFT."""

import os
import sys
sys.path.append('/workspace/asr/peft/src')

from peft import prepare_model_for_kbit_training, TaskType, PeftType, PromptTuningConfig, PromptTuningInit, PrefixTuningConfig # NOTE 非常重要, 4bit, 8bit
#import ipdb; ipdb.set_trace()
#model.gradient_checkpointing_enable() # TODO this has a bug, TODO, layer_past was not used
model.gradient_checkpointing_disable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False) # NOTE

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LazyLoraConfig, LoraConfig, get_peft_model

#import ipdb; ipdb.set_trace()
config_lora = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], # NOTE for gpt-neox-20b 
    #target_modules=["q_proj", 'k_proj', 'v_proj'],  # NOTE for gpt-j-6b
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

peft_config_prompt_tuning = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_id,
)
peft_config_prefix_tuning = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=30,
)
config_lazylora = LazyLoraConfig(
    r=8,
    is_r_by_svd=True, 
    lazy_lora_alpha=32,
    lazy_pre_lora_alpha=0.1,
    lazy_pre_adapter_type='linear',
    target_modules=['query_key_value'],
    lazy_lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_config=peft_config_prompt_tuning,
    prefix_tuning_config=peft_config_prefix_tuning,
)
config = config_lazylora

#import ipdb; ipdb.set_trace()
model = get_peft_model(model, config)
print_trainable_parameters(model) 
# trainable params: 8650752 || all params: 10597552128 || trainable%: 0.08162971878329976

"""Let's load a common dataset, english quotes, to fine tune our model on famous quotes."""

from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

"""Run the cell below to run the training! For the sake of the demo, 
we just ran it for few steps just to showcase how to use this integration 
with existing tools on the HF ecosystem."""

import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

#import ipdb; ipdb.set_trace()
# NOTE
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        #optim="paged_adamw_8bit",
        #n_gpu=1,
        #model_parallel=False,
        optim="adamw_bnb_8bit" # NOTE
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# NOTE --- 进入peft fine-tuning的逻辑 ---
#import ipdb; ipdb.set_trace()
trainer.train()
#import ipdb; ipdb.set_trace()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  
# Take care of distributed/parallel training

# NOTE --- 这是调用peft model的保存方法，把adapter lora的相关参数，保存到outputs ---
model_to_save.save_pretrained("outputs")

# NOTE --- 从outputs中取出保存好的config和.bin checkpoint ---
lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)

text = "Elon Musk "
device = "cuda:0"
#import ipdb; ipdb.set_trace()

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

