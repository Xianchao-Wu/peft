#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

cache_dir=os.getcwd()

device = "cuda:2"
model_name_or_path = "bigscience/bloomz-560m"
tokenizer_name_or_path = "bigscience/bloomz-560m"
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)

dataset_name = "twitter_complaints"
#checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
#    "/", "_"
#)
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
lr = 3e-2
num_epochs = 500 #200
batch_size = 8


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
print(dataset["train"][0])


# In[ ]:


# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print(target_max_length)


def preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
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
eval_dataset = processed_datasets["train"]


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, 
        collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


# In[ ]:


def test_preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    model_inputs = tokenizer(inputs)
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
    return model_inputs


test_dataset = dataset["test"].map(
    test_preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

test_dataloader = DataLoader(test_dataset, 
        collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
print(next(iter(test_dataloader)))


# In[ ]:


print(next(iter(train_dataloader)))


# In[ ]:


len(test_dataloader)


# In[ ]:


print(next(iter(test_dataloader)))


# In[ ]:

import ipdb; ipdb.set_trace()
# creating model
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# trainable params: 8192 || all params: 559222784 || trainable%: 0.0014648902430985358
# NOTE

# In[9]:


# model
# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)


# In[10]:


# training and evaluation
model = model.to(device)
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_epoch{num_epochs}"

is_train=False #True
if is_train:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()} 
            # dict_keys(['input_ids', 'attention_mask', 'labels']), all in shape = [8, 64]
            #         print(batch)
            #         print(batch["input_ids"].shape)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), 
                    skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

    import ipdb; ipdb.set_trace()
    # saving model
    #peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_epoch{num_epochs}"
    model.save_pretrained(peft_model_id)

# In[29]:

import ipdb; ipdb.set_trace()
model.eval()
i = 33
inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
print(dataset["test"][i]["Tweet text"])
print(inputs)

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], 
        max_new_tokens=10, eos_token_id=3
    )
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))


# In[12]:

# In[13]:

ckpt = f"{peft_model_id}/adapter_model.bin"
#get_ipython().system('du -h $ckpt')
os.system('du -h {}'.format(ckpt))


# In[15]:


from peft import PeftModel, PeftConfig

#peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, cache_dir=cache_dir)
model = PeftModel.from_pretrained(model, peft_model_id)


# In[33]:


model.to(device)
model.eval()
i = 33
inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
print(dataset["test"][i]["Tweet text"])
print(inputs)

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
    )
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))


# In[ ]:

eval_preds = []

for _, batch in enumerate(tqdm(eval_dataloader)):
    batch = {k:v.to(device) for k, v in batch.items() if k != 'labels'}
    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=10, eos_token_id=3)
    #preds = outputs[:, max_length:].detach().cpu().numpy()
    preds = outputs.detach().cpu().numpy()
    import ipdb; ipdb.set_trace()
    temp = tokenizer.batch_decode(preds, skip_special_tokens=True)
    temp = [atemp.split('Label : ')[-1] for atemp in temp]
    temp2 = []
    print('before processing', temp)
    for atemp in temp:
        if atemp.startswith('no complaint'):
            temp2.append('no complaint')
        elif atemp.startswith('complaint'):
            temp2.append('complaint')
        else:
            temp2.append(atemp)
    eval_preds.extend(temp2)
    print(temp2)
    print('after processing', temp2)


correct = 0
total = 0
for pred, true in zip(eval_preds, dataset['train'][label_column]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1

accuracy = correct / float(total) * 100.0
print(f'{accuracy=}')
print(f'{eval_preds=}')
print(f"{dataset['train'][label_column]=}")



