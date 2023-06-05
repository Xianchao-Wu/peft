#!/usr/bin/env python
# coding: utf-8

# # Finetuning Whisper-large-V2 on Colab using PEFT-Lora + BNB INT8 training

# In this Colab, we present a step-by-step guide on how to fine-tune Whisper for any multilingual ASR dataset using Hugging Face 🤗 Transformers and 🤗 PEFT. Using 🤗 PEFT and `bitsandbytes`, you can train the `whisper-large-v2` seamlessly on a colab with T4 GPU (16 GB VRAM). In this notebook, with most parts from [fine_tune_whisper.ipynb](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb#scrollTo=BRdrdFIeU78w) is adapted to train using PEFT LoRA+BNB INT8.
# 
# For more details on model, datasets and metrics, refer blog [Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper)
# 
# 

# ## Inital Setup

# In[ ]:


get_ipython().system('add-apt-repository -y ppa:jonathonf/ffmpeg-4')
get_ipython().system('apt update')
get_ipython().system('apt install -y ffmpeg')


# In[ ]:


get_ipython().system('pip install datasets>=2.6.1')
get_ipython().system('pip install git+https://github.com/huggingface/transformers')
get_ipython().system('pip install librosa')
get_ipython().system('pip install evaluate>=0.30')
get_ipython().system('pip install jiwer')
get_ipython().system('pip install gradio')
get_ipython().system('pip install -q bitsandbytes datasets accelerate loralib')
get_ipython().system('pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git@main')


# Linking the notebook to the Hub is straightforward - it simply requires entering your Hub authentication token when prompted. Find your Hub authentication token [here](https://huggingface.co/settings/tokens):

# In[ ]:


from huggingface_hub import notebook_login

notebook_login()


# In[ ]:


# Select CUDA device index
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name_or_path = "openai/whisper-large-v2"
language = "Marathi"
language_abbr = "mr"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"


# ## Load Dataset

# In[ ]:


from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=True)

print(common_voice)


# In[ ]:


common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
)

print(common_voice)


# ## Prepare Feature Extractor, Tokenizer and Data

# In[ ]:


from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)


# In[ ]:


from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)


# In[ ]:


from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)


# ### Prepare Data

# In[ ]:


print(common_voice["train"][0])


# Since 
# our input audio is sampled at 48kHz, we need to _downsample_ it to 
# 16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model. 
# 
# We'll set the audio inputs to the correct sampling rate using dataset's 
# [`cast_column`](https://huggingface.co/docs/datasets/package_reference/main_classes.html?highlight=cast_column#datasets.DatasetDict.cast_column)
# method. This operation does not change the audio in-place, 
# but rather signals to `datasets` to resample audio samples _on the fly_ the 
# first time that they are loaded:

# In[ ]:


from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


# Re-loading the first audio sample in the Common Voice dataset will resample 
# it to the desired sampling rate:

# In[ ]:


print(common_voice["train"][0])


# Now we can write a function to prepare our data ready for the model:
# 1. We load and resample the audio data by calling `batch["audio"]`. As explained above, 🤗 Datasets performs any necessary resampling operations on the fly.
# 2. We use the feature extractor to compute the log-Mel spectrogram input features from our 1-dimensional audio array.
# 3. We encode the transcriptions to label ids through the use of the tokenizer.

# In[ ]:


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# We can apply the data preparation function to all of our training examples using dataset's `.map` method. The argument `num_proc` specifies how many CPU cores to use. Setting `num_proc` > 1 will enable multiprocessing. If the `.map` method hangs with multiprocessing, set `num_proc=1` and process the dataset sequentially.

# In[ ]:


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)


# In[ ]:


common_voice["train"]


# ## Training and Evaluation

# ### Define a Data Collator

# In[ ]:


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# Let's initialise the data collator we've just defined:

# In[ ]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# ### Evaluation Metrics

# We'll use the word error rate (WER) metric, the 'de-facto' metric for assessing 
# ASR systems. For more information, refer to the WER [docs](https://huggingface.co/metrics/wer). We'll load the WER metric from 🤗 Evaluate:

# In[ ]:


import evaluate

metric = evaluate.load("wer")


# We then simply have to define a function that takes our model 
# predictions and returns the WER metric. This function, called
# `compute_metrics`, first replaces `-100` with the `pad_token_id`
# in the `label_ids` (undoing the step we applied in the 
# data collator to ignore padded tokens correctly in the loss).
# It then decodes the predicted and label ids to strings. Finally,
# it computes the WER between the predictions and reference labels:

# In[ ]:


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ### Load a Pre-Trained Checkpoint

# Now let's load the pre-trained Whisper `small` checkpoint. Again, this 
# is trivial through use of 🤗 Transformers!

# In[ ]:


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")

# model.hf_device_map - this should be {" ": 0}


# Override generation arguments - no tokens are forced as decoder outputs (see [`forced_decoder_ids`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids)), no tokens are suppressed during generation (see [`suppress_tokens`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens)):

# In[ ]:


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# ### Post-processing on the model
# 
# Finally, we need to apply some post-processing on the 8-bit model to enable training, let's freeze all our layers, and cast all non `int8` layers in `float32` for stability.

# In[ ]:


from peft import prepare_model_for_int8_training

model = prepare_model_for_int8_training(model)


# ### Apply LoRA
# 
# Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.

# In[ ]:


from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()


# We are ONLY using **1%** of the total trainable parameters, thereby performing **Parameter-Efficient Fine-Tuning**

# ### Define the Training Configuration

# In the final step, we define all the parameters related to training. For more detail on the training arguments, refer to the Seq2SeqTrainingArguments [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

# In[ ]:


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="temp",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=25,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)


# **Few Important Notes:**
# 1. `remove_unused_columns=False` and `label_names=["labels"]` are required as the PeftModel's forward doesn't have the signature of the base model's forward.
# 
# 2. INT8 training required autocasting. `predict_with_generate` can't be passed to Trainer because it internally calls transformer's `generate` without autocasting leading to errors. 
# 
# 3. Because of point 2, `compute_metrics` shouldn't be passed to `Seq2SeqTrainer` as seen below. (commented out)

# In[ ]:


from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!


# In[ ]:


trainer.train()


# In[ ]:


model_name_or_path = "openai/whisper-large-v2"
peft_model_id = "smangrul/" + f"{model_name_or_path}-{model.peft_config.peft_type}-colab".replace("/", "-")
model.push_to_hub(peft_model_id)
print(peft_model_id)


# # Evaluation and Inference

# **Important points to note while inferencing**:
# 1. As `predict_with_generate` can't be used, we will write the eval loop with `torch.cuda.amp.autocast()` as shown below. 
# 2. As the base model is frozen, PEFT model sometimes fails ot recognise the language while decoding.Hence, we force the starting tokens to mention the language we are transcribing. This is done via `forced_decoder_ids = processor.get_decoder_prompt_ids(language="Marathi", task="transcribe")` and passing that too the `model.generate` call.
# 3. Please note that [AutoEvaluate Leaderboard](https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=mozilla-foundation%2Fcommon_voice_11_0&only_verified=0&task=automatic-speech-recognition&config=mr&split=test&metric=wer) for `mr` language on `common_voice_11_0` has a bug wherein openai's `BasicTextNormalizer` normalizer is used while evaluation leading to degerated output text, an example is shown below:
# ```
# without normalizer: 'स्विच्चान नरुवित्तीची पद्दत मोठ्या प्रमाणात आमलात आणल्या बसोन या दुपन्याने अनेक राथ प्रवेश केला आहे.'
# with normalizer: 'स व च च न नर व त त च पद दत म ठ य प रम ण त आमल त आणल य बस न य द पन य न अन क र थ प रव श क ल आह'
# ```
# Post fixing this bug, we report the 2 metrics for the top model of the leaderboard and the PEFT model:
# 1. `wer`: `wer` without using the `BasicTextNormalizer` as it doesn't cater to most indic languages. This is want we consider as true performance metric.
# 2. `normalized_wer`: `wer` using the `BasicTextNormalizer` to be comparable to the leaderboard metrics.
# Below are the results:
# 
# | Model          | DrishtiSharma/whisper-large-v2-marathi | smangrul/openai-whisper-large-v2-LORA-colab |
# |----------------|----------------------------------------|---------------------------------------------|
# | wer            | 35.6457                                | 36.1356                                     |
# | normalized_wer | 13.6440                                | 14.0165                                     |
# 
# We see that PEFT model's performance is comparable to the fully fine-tuned model on the top of the leaderboard. At the same time, we are able to train the large model in Colab notebook with limited GPU memory and the added advantage of resulting checkpoint being jsut `63` MB.
# 
# 

# In[ ]:


from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer

peft_model_id = "smangrul/openai-whisper-large-v2-LORA-colab"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)


# In[ ]:


from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import gc

eval_dataloader = DataLoader(common_voice["test"], batch_size=8, collate_fn=data_collator)

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )
    del generated_tokens, labels, batch
    gc.collect()
wer = 100 * metric.compute()
print(f"{wer=}")


# ## Using AutomaticSpeechRecognitionPipeline

# **Few important notes:**
# 1. `pipe()` should be in the autocast context manager `with torch.cuda.amp.autocast():`
# 2. `forced_decoder_ids` specifying the `language` being transcribed should be provided in `generate_kwargs` dict.
# 3. You will get warning along the below lines which is **safe to ignore**.
# ```
# The model 'PeftModel' is not supported for . Supported models are ['SpeechEncoderDecoderModel', 'Speech2TextForConditionalGeneration', 'SpeechT5ForSpeechToText', 'WhisperForConditionalGeneration', 'Data2VecAudioForCTC', 'HubertForCTC', 'MCTCTForCTC', 'SEWForCTC', 'SEWDForCTC', 'UniSpeechForCTC', 'UniSpeechSatForCTC', 'Wav2Vec2ForCTC', 'Wav2Vec2ConformerForCTC', 'WavLMForCTC'].
# 
# ```

# In[ ]:


import torch
import gradio as gr
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig


peft_model_id = "smangrul/openai-whisper-large-v2-LORA-colab"
language = "Marathi"
task = "transcribe"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)

model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)


def transcribe(audio):
    with torch.cuda.amp.autocast():
        text = pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
    return text


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="PEFT LoRA + INT8 Whisper Large V2 Marathi",
    description="Realtime demo for Marathi speech recognition using `PEFT-LoRA+INT8` fine-tuned Whisper Large V2 model.",
)

iface.launch(share=True)


# In[ ]:




