#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import AutoModelForCausalLM
import os
import sys
sys.path.append(os.getcwd()+"/../../src")
from peft import get_peft_config, get_peft_model, PrefixTuningConfig, TaskType, PeftType, PromptTuningConfig, PromptTuningInit
import torch
torch.autograd.set_detect_anomaly(True)
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
#from datasets import load_dataset

#cache_dir=os.getcwd()
cache_dir="/workspace/asr/Huatuo-Llama-Med-Chinese"

device = "cuda:0"
#model_name_or_path = "bigscience/bloomz-560m" # NOTE change model size if required
#tokenizer_name_or_path = "bigscience/bloomz-560m"

model_name_or_path = "EleutherAI/gpt-neox-20b" # NOTE change model size if required
tokenizer_name_or_path = "EleutherAI/gpt-neox-20b"

#model_name_or_path = "bigscience/bloomz-7b1"
#tokenizer_name_or_path = "bigscience/bloomz-7b1"

#peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30) # PrefixTuningConfig(peft_type=<PeftType.PREFIX_TUNING: 'PREFIX_TUNING'>, base_model_name_or_path=None, task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, inference_mode=False, num_virtual_tokens=30, token_dim=None, num_transformer_submodules=None, num_attention_heads=None, num_layers=None, encoder_hidden_size=None, prefix_projection=False)

dataset_name = "twitter_complaints"
#checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
#    "/", "_"
#) # 'twitter_complaints_bigscience_bloomz-560m_PREFIX_TUNING_CAUSAL_LM_v1.pt' NOTE
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
lr = 1e-12
num_epochs = 300 # NOTE TODO, change this to 50 for the real peft
batch_size = 8 #16 


# In[3]:


#from datasets import load_dataset

dataset = load_dataset("ought/raft", dataset_name)

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names] 
# ['Unlabeled', 'complaint', 'no complaint']
print(classes)
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
print(dataset)
print(dataset["train"][0]) 
# {'Tweet text': '@HMRCcustomers No this is my first job', 'ID': 0, 'Label': 2, 'text_label': 'no complaint'}


# In[4]:


# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
        cache_dir=cache_dir) 
# BloomTokenizerFast(name_or_path='bigscience/bloomz-560m', vocab_size=250680, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes]) 
# 3 after tokenization NOTE
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
    train_dataset, 
    shuffle=True, 
    collate_fn=default_data_collator, 
    batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, 
        collate_fn=default_data_collator, 
        batch_size=batch_size, 
        pin_memory=True)


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
        collate_fn=default_data_collator, 
        batch_size=batch_size, 
        pin_memory=True)
print(next(iter(test_dataloader)))


# In[ ]:


print(next(iter(train_dataloader)))


# In[7]:


print(len(test_dataloader))


# In[ ]:


print(next(iter(test_dataloader)))


# In[9]:

#import ipdb; ipdb.set_trace()

# creating model, NOTE

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
        quantization_config=bnb_config,
        device_map={"":0},
        #device_map='auto', 
        cache_dir=cache_dir) # 559,214,592

from peft import LazyLoraConfig, get_peft_model

peft_config_prompt_tuning = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
) # PromptTuningConfig(peft_type=<PeftType.PROMPT_TUNING: 'PROMPT_TUNING'>, base_model_name_or_path=None, task_type=<Ta
#skType.CAUSAL_LM: 'CAUSAL_LM'>, inference_mode=False, num_virtual_tokens=8, token_dim=None, num_transformer_submodules=
#None, num_attention_heads=None, num_layers=None, prompt_tuning_init=<PromptTuningInit.TEXT: 'TEXT'>, prompt_tuning_init
#_text='Classify if the tweet is a complaint or not:', tokenizer_name_or_path='bigscience/bloomz-560m')

peft_config_prefix_tuning = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM, 
    num_virtual_tokens=30
)


#import ipdb; ipdb.set_trace()
config_lazy_lora = LazyLoraConfig(
    r=8,
    is_r_by_svd=False, #True, # use svd singular value to determine rank r, dynamically NOTE
    lazy_lora_alpha=32,
    lazy_pre_lora_alpha=0.1, 
    lazy_pre_adapter_type='linear', #'linear', 'conv1d', 'none'
    target_modules=['query_key_value', 'dense_h_to_4h', 'dense_4h_to_h'],
    lazy_lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    prompt_tuning_config=peft_config_prompt_tuning,
    prefix_tuning_config=peft_config_prefix_tuning,
) 
peft_config = config_lazy_lora

model = get_peft_model(model, config_lazy_lora)

#import ipdb; ipdb.set_trace()
#model = get_peft_model(model, peft_config) # 560,689,152
model.print_trainable_parameters()

# trainable params: 786432 || all params: 560001024 || trainable%: 0.14043402892063284, yes for LoRA
# trainable params: 1277952 || all params: 560492544 || trainable%: 0.228005173963563, yes for lazy lora, with two conv1d layers now! NOTE
# trainable params: 2668544 || all params: 410888192 || trainable%: 0.6494574562999367 # with prompt learning + lazy lora + r_by_svd NOTE
# In[10]:
#model.print_trainable_parameters()

# trainable params: 16825344 || all params: 425044992 || trainable%: 3.9584854113514645 # for 'query_key_value', 'dense_h_to_4h', 'dense_4h_to_h' -> three linear layers

# In[ ]:
#import ipdb; ipdb.set_trace()
print(model)

# In[12]:

print(model.peft_config) 

# In[13]:

# model
# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# In[14]:


# training and evaluation
model = model.to(device)

is_train = True # NOTE
if is_train:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()} 
            # dict_keys(['input_ids', 'attention_mask', 'labels']), 
            # batch.size=8; batch['input_ids'].shape=[8, 64]; 
            # batch['attention_mask']=[8, 64], batch['labels']=[8, 64] NOTE 
            #         print(batch)
            #         print(batch["input_ids"].shape)
            #import ipdb; ipdb.set_trace()
            outputs = model(**batch) 
            # TODO forward, need to check the forward algorithm details... NOTE
            loss = outputs.loss
            total_loss += loss.detach().float()
            #import ipdb; ipdb.set_trace()
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
                tokenizer.batch_decode(torch.argmax(outputs.logits, 
                    -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

    #import ipdb; ipdb.set_trace()
    # saving model
    peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_epoch{num_epochs}_4bit" # 'bigscience/bloomz-560m_LAZY_LORA_CAUSAL_LM_epoch500' NOTE
    model.save_pretrained(peft_model_id)
    ckpt = f"{peft_model_id}/adapter_model.bin"
    #get_ipython().system('du -h $ckpt')
    os.system('du -h {}'.format(ckpt))

# In[36]:

#import ipdb; ipdb.set_trace()
model.eval()

if is_train: 
    with torch.no_grad():
        #i = 16
        for i in range(len(dataset['test'])):
            inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', 
                    return_tensors="pt")
            print(dataset["test"][i]["Tweet text"])
            print(inputs)
            inputs = {k: v.to(device) for k, v in inputs.items()} 
            # k='input_ids' in cpu, v=tensor in gpu, 这个有意思
            outputs = model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                max_new_tokens=10, 
                eos_token_id=3
            ) # NOTE
            print(outputs)
            print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), 
                skip_special_tokens=True))

            break

# In[16]:

# In[18]:

from peft import PeftModel, PeftConfig

#import ipdb; ipdb.set_trace()
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_epoch{num_epochs}_4bit"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map={"":0},
        #device_map='auto', 
        cache_dir=cache_dir)
model = PeftModel.from_pretrained(model, peft_model_id)

#import ipdb; ipdb.set_trace()

# In[21]:

print('-'*30)
print('-'*30)

model.to(device)
model.eval()

if is_train:
    with torch.no_grad():
        for i in range(len(dataset['test'])):
            inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', 
                    return_tensors="pt")
            print(dataset["test"][i]["Tweet text"])
            print(inputs)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                max_new_tokens=10, 
                eos_token_id=3
            )
            print(outputs)
            print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), 
                skip_special_tokens=True))
            break

eval_preds = []
#import ipdb; ipdb.set_trace()
for _, batch in enumerate(tqdm(eval_dataloader)):
    batch = {k:v.to(device) for k, v in batch.items() if k != 'labels'}
    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=10, eos_token_id=3)
    #preds = outputs[:, max_length:].detach().cpu().numpy()
    preds = outputs.detach().cpu().numpy()
    #import ipdb; ipdb.set_trace()
    temp = tokenizer.batch_decode(preds, skip_special_tokens=True)
    temp = [atemp.split('Label : ')[-1] for atemp in temp]
    temp2 = []
    for atemp in temp:
        if atemp.startswith('complaint'):
            temp2.append('complaint')
        elif atemp.startswith('no complaint'):
            temp2.append('no complaint')
        else:
            temp2.append(atemp)
    eval_preds.extend(temp2)
    break

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

