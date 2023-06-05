#!/usr/bin/env python
# coding: utf-8

# ## Fine-tune large models using 🤗 `peft` adapters, `transformers` & `bitsandbytes`
# 
# In this tutorial we will cover how we can fine-tune large language models using the very recent `peft` library and `bitsandbytes` for loading large models in 8-bit.
# The fine-tuning method will rely on a recent method called "Low Rank Adapters" (LoRA), instead of fine-tuning the entire model you just have to fine-tune these adapters and load them properly inside the model. 
# After fine-tuning the model you can also share your adapters on the 🤗 Hub and load them very easily. Let's get started!

# ### Install requirements
# 
# First, run the cells below to install the requirements:

# In[1]:


get_ipython().system('pip install -q bitsandbytes datasets accelerate loralib')
get_ipython().system('pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git')


# ### Model loading
# 
# Here let's load the `opt-6.7b` model, its weights in half-precision (float16) are about 13GB on the Hub! If we load them in 8-bit we would require around 7GB of memory instead.

# In[2]:


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")


# ### Prepare model for training
# 
# Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_int8_training` that will: 
# - Casts all the non `int8` modules to full precision (`fp32`) for stability
# - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
# - Enable gradient checkpointing for more memory-efficient training

# In[3]:


from peft import prepare_model_for_int8_training

model = prepare_model_for_int8_training(model)


# ### Apply LoRA
# 
# Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.

# In[4]:


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


# In[5]:


from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


# ### Training

# 

# In[ ]:


import transformers
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


# ## Share adapters on the 🤗 Hub

# In[ ]:


from huggingface_hub import notebook_login

notebook_login()


# In[ ]:


model.push_to_hub("ybelkada/opt-6.7b-lora", use_auth_token=True)


# ## Load adapters from the Hub
# 
# You can also directly load adapters from the Hub using the commands below:

# In[ ]:


import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-6.7b-lora"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


# ## Inference
# 
# You can then directly use the trained model or the model that you have loaded from the 🤗 Hub for inference as you would do it usually in `transformers`.

# In[ ]:


batch = tokenizer("Two things are infinite: ", return_tensors="pt")

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))


# As you can see by fine-tuning for few steps we have almost recovered the quote from Albert Einstein that is present in the [training data](https://huggingface.co/datasets/Abirate/english_quotes).
