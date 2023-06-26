import os
import sys

sys.path.append(os.getcwd() + "/../../src")

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup

from peft import AdaLoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model

cache_dir=os.getcwd()


os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda"
model_name_or_path = "facebook/bart-base"
tokenizer_name_or_path = "facebook/bart-base"

checkpoint_name = "financial_sentiment_analysis_lora_v1.pt"
text_column = "sentence"
label_column = "text_label"
max_length = 128
lr = 1e-3
num_epochs = 2 #8
batch_size = 8


# creating model
peft_config = AdaLoraConfig(
    init_r=12,
    target_r=8,
    beta1=0.85,
    beta2=0.85,
    tinit=2,
    tfinal=100,
    deltaT=10,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
)

import ipdb; ipdb.set_trace()
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# trainable params: 2434176 || all params: 141854688 || trainable%: 1.715964438200308

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


# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir) # NOTE
# BartTokenizerFast(name_or_path='facebook/bart-base', vocab_size=50265, model_max_length=1024, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False)}, clean_up_tokenization_spaces=True)

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, 
            max_length=max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt")

    labels = tokenizer(targets, 
            max_length=3, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt")

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

train_dataloader = DataLoader(
    train_dataset, shuffle=True, 
    collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, 
        collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
#model.base_model.peft_config.total_step = len(train_dataloader) * num_epochs
model.base_model.peft_config['default'].total_step = len(train_dataloader) * num_epochs

import ipdb; ipdb.set_trace()
# training and evaluation
model = model.to(device)
global_step = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch) # odict_keys(['loss', 'logits', 'encoder_last_hidden_state'])
        # ipdb> tokenizer.decode(batch['labels'][0]) -> '<s>positive</s>'

        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # Update the importance of low-rank matrices
        # and allocate the budget accordingly.
        model.base_model.update_and_allocate(global_step) # TODO 终于找到了！核心的代码
        # ipdb> type(model.base_model)
        # <class 'peft.tuners.adalora.AdaLoraModel'>
        # ipdb> type(model)
        # <class 'peft.peft_model.PeftModelForSeq2SeqLM'>

        optimizer.zero_grad()
        global_step += 1

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

    eval_epoch_loss = eval_loss / len(train_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(eval_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


# print accuracy
correct = 0
total = 0
for pred, true in zip(eval_preds, dataset["validation"]["text_label"]):
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy=} % on the evaluation dataset")
print(f"{eval_preds[:10]=}")
print(f"{dataset['validation']['text_label'][:10]=}")


# saving model
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)


ckpt = f"{peft_model_id}/adapter_model.bin"
# get_ipython().system('du -h $ckpt')


peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

import ipdb; ipdb.set_trace()

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, cache_dir=cache_dir)
model = PeftModel.from_pretrained(model, peft_model_id) # peft_model_id='facebook/bart-base_ADALORA_SEQ_2_SEQ_LM'


model.eval()
i = 13
inputs = tokenizer(dataset["validation"][text_column][i], return_tensors="pt")
print(dataset["validation"][text_column][i])
print(inputs)

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))

