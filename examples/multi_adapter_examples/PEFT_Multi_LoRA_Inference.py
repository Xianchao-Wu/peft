#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q git+https://github.com/huggingface/transformers.git')
get_ipython().system('pip install -q git+https://github.com/huggingface/peft.git')
get_ipython().system('pip install -q git+https://github.com/huggingface/accelerate.git@main')
get_ipython().system('pip install huggingface_hub')
get_ipython().system('pip install bitsandbytes')
get_ipython().system('pip install SentencePiece')


# In[ ]:


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[ ]:


from huggingface_hub import notebook_login
import torch

notebook_login()


# In[ ]:


from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

model_name = "decapoda-research/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto", use_auth_token=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b", adapter_name="eng_alpaca")\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.load_adapter("22h/cabrita-lora-v0-1", adapter_name="portuguese_alpaca")\n')


# In[ ]:


model


# In[ ]:


model.to("cuda")


# In[ ]:


import torch

device = "cuda"


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=256,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        no_repeat_ngram_size=3,
        **kwargs,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.set_adapter("eng_alpaca")\n')


# In[ ]:


instruction = "Tell me about alpacas."

print(evaluate(instruction))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.set_adapter("portuguese_alpaca")\n')


# In[ ]:


instruction = "Invente uma desculpa criativa pra dizer que não preciso ir à festa."

print(evaluate(instruction))


# In[ ]:


with model.disable_adapter():
    instruction = "Invente uma desculpa criativa pra dizer que não preciso ir à festa."

    print(evaluate(instruction))

