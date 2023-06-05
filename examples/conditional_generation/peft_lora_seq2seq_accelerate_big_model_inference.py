#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

dataset_name = "twitter_complaints"
text_column = "Tweet text"
label_column = "text_label"
batch_size = 8

peft_model_id = "smangrul/twitter_complaints_bigscience_T0_3B_LORA_SEQ_2_SEQ_LM"
config = PeftConfig.from_pretrained(peft_model_id)


# In[2]:


peft_model_id = "smangrul/twitter_complaints_bigscience_T0_3B_LORA_SEQ_2_SEQ_LM"
max_memory = {0: "6GIB", 1: "0GIB", 2: "0GIB", 3: "0GIB", 4: "0GIB", "cpu": "30GB"}
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map="auto", max_memory=max_memory)
model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto", max_memory=max_memory)


# In[ ]:


from datasets import load_dataset

dataset = load_dataset("ought/raft", dataset_name)

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
print(classes)
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
print(dataset)
dataset["train"][0]


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, truncation=True)
    labels = tokenizer(
        targets, max_length=target_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["train"]
test_dataset = processed_datasets["test"]


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)


# In[5]:


model.eval()
i = 15
inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
print(dataset["test"][i]["Tweet text"])
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))


# In[6]:


model.eval()
eval_preds = []
for _, batch in enumerate(tqdm(eval_dataloader)):
    batch = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=10)
    preds = outputs.detach().cpu().numpy()
    eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))


# In[7]:


correct = 0
total = 0
for pred, true in zip(eval_preds, dataset["train"][label_column]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy=}")
print(f"{eval_preds[:10]=}")
print(f"{dataset['train'][label_column][:10]=}")


# In[ ]:


model.eval()
test_preds = []

for _, batch in enumerate(tqdm(test_dataloader)):
    batch = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=10)
    preds = outputs.detach().cpu().numpy()
    test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
    if len(test_preds) > 100:
        break
test_preds

