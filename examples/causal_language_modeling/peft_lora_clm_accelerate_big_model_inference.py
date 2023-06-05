#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
#from datasets import load_dataset

device = "cuda"
model_name_or_path = "bigscience/bloomz-7b1" # 70亿参数的模型
tokenizer_name_or_path = "bigscience/bloomz-7b1"
dataset_name = "twitter_complaints"
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
lr = 1e-3
num_epochs = 50
batch_size = 8

cache_dir=os.getcwd() # '/workspace/asr/peft/examples/causal_language_modeling'


# In[ ]:


#from datasets import load_dataset

dataset = load_dataset("ought/raft", dataset_name) # The Real-world Annotated Few-shot Tasks (RAFT) dataset is an aggregation of English-language datasets found in the real world. Associated with each dataset is a binary or multiclass classification task, intended to improve our understanding of how language models perform on tasks that have concrete, real-world value. Only 50 labeled examples are provided in each dataset.

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
print(classes)
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
print(dataset)
print(dataset["train"][0]) # {'Tweet text': '@HMRCcustomers No this is my first job', 'ID': 0, 'Label': 2, 'text_label': 'no complaint'}


# In[3]:


# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir) # BloomTokenizerFast(name_or_path='bigscience/bloomz-7b1', vocab_size=250680, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=False), NOTE 25万个词条
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print(target_max_length) # ['Unlabeled', 'complaint', 'no complaint'], 经过tokenizer之后的序列的长度，最长为3


def preprocess_function(examples):
    batch_size = len(examples[text_column]) # 50
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]] # 'Tweet text : @HMRCcustomers No this is my first job Label : ' = inputs[0]
    targets = [str(x) for x in examples[label_column]] # 'no complaint' = targets[0]
    model_inputs = tokenizer(inputs)
    '''ipdb> len(model_inputs['input_ids'])
    50
    ipdb> model_inputs['input_ids'][0]
    [227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210]
    ipdb> model_inputs['attention_mask'][0]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ''' 

    labels = tokenizer(targets)
    # targets = ['no complaint', 'no complaint' ... ]
    '''
    ipdb> labels['input_ids'][0]
    [1936, 106863]
    ipdb> labels['attention_mask'][0]
    [1, 1]
    '''
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i] # [227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210] -> 17个token ids
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id] # [1936, 106863, 3] -> 3个token ids
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids # [227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210,  ||| NOTE ||| 1936, 106863, 3]
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids # [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1936, 106863, 3] # NOTE 需要注意的是，前面都是-100! 占据位置的而已, 20个位置！之后是1936, 106863, 3 这个标签
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # ipdb> model_inputs['attention_mask'][0]
        # len=20 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i] # [227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210, ||| NOTE ||| 1936, 106863, 3]
        label_input_ids = labels["input_ids"][i] # [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1936, 106863, 3]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids) # max_length=64
        ) + sample_input_ids # [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210, 1936, 106863, 3]
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i] # ipdb> model_inputs['attention_mask'][0]
        #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        #ipdb> len(model_inputs['attention_mask'][0])
        #64

        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        #ipdb> labels['input_ids'][0]
        #[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1936, 106863, 3]
        #ipdb> len(labels['input_ids'][0])
        #64

        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        #ipdb> model_inputs['input_ids'][0]
        '''tensor([     3,      3,      3,      3,      3,      3,      3,      3,      3,
                     3,      3,      3,      3,      3,      3,      3,      3,      3,
                     3,      3,      3,      3,      3,      3,      3,      3,      3,
                     3,      3,      3,      3,      3,      3,      3,      3,      3,
                     3,      3,      3,      3,      3,      3,      3,      3, 227985,
                  5484,    915,   2566, 169403,  15296,  36272,    525,   3928,   1119,
                   632,   2670,   3968,  15270,  77658,    915,    210,   1936, 106863,
                     3])
        ipdb> model_inputs['input_ids'][0].shape
        torch.Size([64])
        '''

        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        '''
        ipdb> model_inputs['attention_mask'][0]
        tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        ipdb> model_inputs['attention_mask'][0].shape
        torch.Size([64])
        '''

        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        '''ipdb> labels['input_ids'][0]
        tensor([  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
                  -100,   -100,   -100,   -100,   -100,   -100,   -100,   1936, 106863,
                     3])
        ipdb> labels['input_ids'][0].shape
        torch.Size([64])
        '''

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    '''ipdb> len(model_inputs['input_ids'])
    50, 一个batch里面有50个序列
    ipdb> model_inputs['input_ids'][0].shape
    torch.Size([64])
    ipdb> model_inputs['attention_mask'][0].shape
    torch.Size([64])
    ipdb> model_inputs['labels'][0].shape
    torch.Size([64])
    ipdb>
    '''

processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

import ipdb; ipdb.set_trace()

# In[ ]:


def test_preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]] # len(inputs)=50, inputs[0]='Tweet text : @HMRCcustomers No this is my first job Label : '
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
        '''
        ipdb> model_inputs['input_ids'][0]
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 227985, 5484, 915, 2566, 169403, 15296, 36272, 525, 3928, 1119, 632, 2670, 3968, 15270, 77658, 915, 210]
        ipdb> len(model_inputs['input_ids'][0])
        64

        ipdb> model_inputs['attention_mask'][0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ipdb> len(model_inputs['attention_mask'][0])
        64
        '''

        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        '''
        ipdb> model_inputs['input_ids'][0]
        tensor([     3,      3,      3,      3,      3,      3,      3,      3,      3,
                     3,      3,      3,      3,      3,      3,      3,      3,      3,
                     3,      3,      3,      3,      3,      3,      3,      3,      3,
                     3,      3,      3,      3,      3,      3,      3,      3,      3,
                     3,      3,      3,      3,      3,      3,      3,      3,      3,
                     3,      3, 227985,   5484,    915,   2566, 169403,  15296,  36272,
                   525,   3928,   1119,    632,   2670,   3968,  15270,  77658,    915,
                   210])
        ipdb> model_inputs['input_ids'][0].shape
        torch.Size([64])

        '''

        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        '''
        ipdb> model_inputs['attention_mask'][0]
        tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        ipdb> model_inputs['attention_mask'][0].shape
        torch.Size([64])

        '''
    return model_inputs


processed_datasets = dataset.map(
    test_preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

eval_dataset = processed_datasets["train"]
test_dataset = processed_datasets["test"]

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
print(next(iter(eval_dataloader)))
print(next(iter(test_dataloader)))


# In[5]:

import ipdb; ipdb.set_trace()
#from peft import PeftModel, PeftConfig

#max_memory = {0: "1GIB", 1: "1GIB", 2: "2GIB", 3: "10GIB", "cpu": "30GB"}
peft_model_id = "smangrul/twitter_complaints_bigscience_bloomz-7b1_LORA_CAUSAL_LM"

config = PeftConfig.from_pretrained(peft_model_id) #, cache_dir=cache_dir), PeftConfig(peft_type='LORA', base_model_name_or_path='bigscience/bloomz-7b1', task_type='CAUSAL_LM', inference_mode=True)

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
        device_map="auto", 
        #max_memory=max_memory,
        cache_dir=cache_dir) # 7,069,016,064=70.7B parameters; 这个是可以使用cache_dir的
'''
      (29): BloomBlock(
        (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear(in_features=4096, out_features=12288, bias=True)
          (dense): Linear(in_features=4096, out_features=4096, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)
        )
      )
'''
import ipdb; ipdb.set_trace()

from peft import LoraConfig, get_peft_model

config_restart = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['query_key_value'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
        )

model_restart = get_peft_model(model, config_restart)

import ipdb; ipdb.set_trace()

model = PeftModel.from_pretrained(model, # TODO, 这个出错了，上面的model传递过来了 NOTE
        peft_model_id, 
        device_map="auto") #, 
        #max_memory=max_memory)#,
        #cache_dir=cache_dir)
'''
     (29): BloomBlock(
        (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear(
            in_features=4096, out_features=12288, bias=True
            (lora_dropout): ModuleDict(
              (default): Dropout(p=0.1, inplace=False)
            )
            (lora_A): ModuleDict(
              (default): Linear(in_features=4096, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (default): Linear(in_features=8, out_features=12288, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
          )
          (dense): Linear(in_features=4096, out_features=4096, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)
        )
      )
'''

# TODO bug:
#         size mismatch for base_model.model.transformer.h.29.self_attention.query_key_value.lora_B.default.weight: copying a param with shape torch.Size([8192, 8, 1]) from checkpoint, the shape in current model is torch.Size([12288, 8]).


# In[35]:

print(model)

# In[7]:

print(model.hf_device_map)


# In[34]:


model.eval()
i = 89
inputs = tokenizer(f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', 
        return_tensors="pt")
print(dataset["test"][i]["Tweet text"])
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], 
            max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), 
        skip_special_tokens=True))


# In[9]:


model.eval()
eval_preds = []
for _, batch in enumerate(tqdm(eval_dataloader)):
    batch = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=10)
    preds = outputs[:, max_length:].detach().cpu().numpy()
    eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))


# In[11]:


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
    preds = outputs[:, max_length:].detach().cpu().numpy()
    test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
    if len(test_preds) > 100:
        break

print(test_preds)


# In[ ]:




