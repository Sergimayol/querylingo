"""
DEBUG=4 CACHE_DIR=/mnt/d/models/ PYTHONPATH="src" python tests/d.py
"""

import os, wandb, torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoConfig,
    TextGenerationPipeline,
    TextStreamer,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

from utils import CACHE_DIR, WEIGHTS, DEBUG

"b-mc2/sql-create-context"
dataset = load_dataset("ChrisHayduk/Llama-2-SQL-Dataset", split="train")
eval_dataset = load_dataset("ChrisHayduk/Llama-2-SQL-Dataset", split="eval")


def format_prompt(examples):
    return {"prompt": f"{examples['input']} {examples['output']}"}


dataset = dataset.map(format_prompt).select(range(10000))
eval_dataset = eval_dataset.map(format_prompt).select(range(500))

# base_model = "NousResearch/Llama-2-7b-chat-hf"
# base_model = "Felladrin/Llama-68M-Chat-v1"
# base_model = "openlm-research/open_llama_3b"
# base_model = "harborwater/open-llama-3b-everythingLM-2048"
base_model = "openai-community/gpt2"
# Fine-tune model name
new_model = "llama-2-68M-sql-coder"
# tokenizer = LlamaTokenizer.from_pretrained(base_model, cache_dir=CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=CACHE_DIR)
# In Llama2 we dont have the padding token which is a very big problem, because we have a dataset with different number of tokens in each row.
# So, we need to pad it so they all have the same length and here i am using end of sentence token and this will have an impact on the generation of our model
# I am using End of Sentence token for fine-tuning
tokenizer.pad_token = tokenizer.eos_token
if any(k in base_model for k in ("gpt", "opt", "bloom")):
    tokenizer.padding_side = "left"
else:
    tokenizer.padding_side = "right"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # Quant type
    # We will use the "nf4" format this was introduced in the QLoRA paper
    bnb_4bit_quant_type="nf4",
    ##As the model weights are stored using 4 bits and when we want to compute its only going to use 16 bits so we have more accuracy
    bnb_4bit_compute_dtype=torch.float16,
    ##Quantization parameters are quantized
    bnb_4bit_use_double_quant=True,
)
# LoRA configuration
peft_config = LoraConfig(
    # Alpha is the strength of the adapters. In LoRA, instead of training all the weights, we will add some adapters in some layers and we will only
    # train the added weights
    # We can merge these adapters in some layers in a very weak way using very low value of alpha (using very little weight) or using a high value of alpha
    # (using a big weight)
    # 15 is very big weight, usually 32 is considered as the standard value for this parameter
    lora_alpha=15,
    # 10% dropout
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)


# Load base moodel
def get_model(model: str, bnb_conf: BitsAndBytesConfig, with_weights: bool) -> AutoModelForCausalLM:
    if DEBUG >= 3:
        print(f"Loading model {model} with weights {with_weights}")
    return (
        AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb_conf, device_map={"": 0}, cache_dir=CACHE_DIR)
        if with_weights
        else AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model, quantization_config=bnb_conf, cache_dir=CACHE_DIR))
    )


# Load base moodel
model = get_model(base_model, bnb_config, WEIGHTS)

if DEBUG >= 3:
    print(f"Model loaded: {model}")

model.config.use_cache = False
model.config.pretraining_tp = 1

# Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
# prepare_model_for_kbit_training---> This function basically helps to built the best model possible
model = prepare_model_for_kbit_training(model)

if DEBUG >= 3:
    print(f"Model loaded: {model}")

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=15,  # 3,5 good for the Llama 2 Model
    per_device_train_batch_size=4,  # Number of batches that we are going to take for every step
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",  # Not helpful because we donot want to evaluate the model we just want to train it
    eval_steps=500,  # Evaluate the model after every 1000 steps
    logging_steps=500,
    optim="paged_adamw_8bit",  # Adam Optimizer we will be using but a version that is paged and in 8 bits, so it will lose less memory
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_steps=10,
    report_to="wandb",
    max_steps=-1,  # if maximum steps=2, it will stop after two steps
)

print(training_arguments.device)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,  # No separate evaluation dataset, i am using the same dataset
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=1024,  # In dataset creation we put a threshold 2k for context length (input token limit) but we dont have enough VRAM unfortunately it will take a lot of VRAM to put everything into memory so we are just gonna stop at 512
    tokenizer=tokenizer,
    args=training_arguments,
)


trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

wandb.finish()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    framework="pt",
)
while 1:
    print("Enter instruction: ")
    inst = input()
    if inst == "exit":
        break
    sequences = pipe(
        inst,
        max_length=1024,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        streamer=TextStreamer(tokenizer),
    )
    print(sequences)
