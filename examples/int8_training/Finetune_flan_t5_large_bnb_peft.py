#!/usr/bin/env python
# coding: utf-8

# # Fine-tune FLAN-T5 using `bitsandbytes`, `peft` & `transformers` 🤗 

# In this notebook we will see how to properly use `peft` , `transformers` & `bitsandbytes` to fine-tune `flan-t5-large` in a google colab!
# 
# We will finetune the model on [`financial_phrasebank`](https://huggingface.co/datasets/financial_phrasebank) dataset, that consists of pairs of text-labels to classify financial-related sentences, if they are either `positive`, `neutral` or `negative`.
# 
# Note that you could use the same notebook to fine-tune `flan-t5-xl` as well, but you would need to shard the models first to avoid CPU RAM issues on Google Colab, check [these weights](https://huggingface.co/ybelkada/flan-t5-xl-sharded-bf16).

# ## Install requirements

# In[ ]:


get_ipython().system('pip install -q bitsandbytes datasets accelerate loralib')
get_ipython().system('pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git@main')


# ## Import model and tokenizer

# In[ ]:


# Select CUDA device index
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-large"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


# ## Prepare model for training

# Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_int8_training` that will: 
# - Casts all the non `int8` modules to full precision (`fp32`) for stability
# - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
# - Enable gradient checkpointing for more memory-efficient training

# In[ ]:


from peft import prepare_model_for_int8_training

model = prepare_model_for_int8_training(model)


# ## Load your `PeftModel` 
# 
# Here we will use LoRA (Low-Rank Adaptators) to train our model

# In[ ]:


from peft import LoraConfig, get_peft_model, TaskType


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


lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
)


model = get_peft_model(model, lora_config)
print_trainable_parameters(model)


# As you can see, here we are only training 0.6% of the parameters of the model! This is a huge memory gain that will enable us to fine-tune the model without any memory issue.

# ## Load and process data
# 
# Here we will use [`financial_phrasebank`](https://huggingface.co/datasets/financial_phrasebank) dataset to fine-tune our model on sentiment classification on financial sentences. We will load the split `sentences_allagree`, which corresponds according to the model card to the split where there is a 100% annotator agreement.

# In[ ]:


# loading dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]
del dataset["test"]

classes = dataset["train"].features["label"].names
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)


# Let's also apply some pre-processing of the input data, the labels needs to be pre-processed, the tokens corresponding to `pad_token_id` needs to be set to `-100` so that the `CrossEntropy` loss associated with the model will correctly ignore these tokens.

# In[ ]:


# data preprocessing
text_column = "sentence"
label_column = "text_label"
max_length = 128


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]


# ## Train our model! 
# 
# Let's now train our model, run the cells below.
# Note that for T5 since some layers are kept in `float32` for stability purposes there is no need to call autocast on the trainer.

# In[ ]:


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    "temp",
    evaluation_strategy="epoch",
    learning_rate=1e-3,
    gradient_accumulation_steps=1,
    auto_find_batch_size=True,
    num_train_epochs=1,
    save_steps=100,
    save_total_limit=8,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


# In[ ]:


trainer.train()


# ## Qualitatively test our model

# Let's have a quick qualitative evaluation of the model, by taking a sample from the dataset that corresponds to a positive label. Run your generation similarly as you were running your model from `transformers`:

# In[ ]:


model.eval()
input_text = "In January-September 2009 , the Group 's net interest income increased to EUR 112.4 mn from EUR 74.3 mn in January-September 2008 ."
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)

print("input sentence: ", input_text)
print(" output prediction: ", tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))


# ## Share your adapters on 🤗 Hub

# Once you have trained your adapter, you can easily share it on the Hub using the method `push_to_hub` . Note that only the adapter weights and config will be pushed

# In[ ]:


from huggingface_hub import notebook_login

notebook_login()


# In[ ]:


model.push_to_hub("ybelkada/flan-t5-large-financial-phrasebank-lora", use_auth_token=True)


# ## Load your adapter from the Hub

# You can load the model together with the adapter with few lines of code! Check the snippet below to load the adapter from the Hub and run the example evaluation!

# In[ ]:


import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

peft_model_id = "ybelkada/flan-t5-large-financial-phrasebank-lora"
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)


# In[ ]:


model.eval()
input_text = "In January-September 2009 , the Group 's net interest income increased to EUR 112.4 mn from EUR 74.3 mn in January-September 2008 ."
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)

print("input sentence: ", input_text)
print(" output prediction: ", tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

