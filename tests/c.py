import os, wandb, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

from utils import CACHE_DIR

dataset = load_dataset("ChrisHayduk/Llama-2-SQL-Dataset", split="train")
eval_dataset = load_dataset("ChrisHayduk/Llama-2-SQL-Dataset", split="eval")

base_model = "NousResearch/Llama-2-7b-chat-hf"
base_model = "Felladrin/Llama-68M-Chat-v1"
base_model = "openlm-research/open_llama_3b"
base_model = "harborwater/open-llama-3b-everythingLM-2048"
# Fine-tune model name
new_model = "llama-2-7b-platypus"
# tokenizer = LlamaTokenizer.from_pretrained(base_model, cache_dir=CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=CACHE_DIR)
# In Llama2 we dont have the padding token which is a very big problem, because we have a dataset with different number of tokens in each row.
# So, we need to pad it so they all have the same length and here i am using end of sentence token and this will have an impact on the generation of our model
# I am using End of Sentence token for fine-tuning
tokenizer.pad_token = tokenizer.eos_token
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
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"": 0},
    cache_dir=CACHE_DIR,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
# prepare_model_for_kbit_training---> This function basically helps to built the best model possible
model = prepare_model_for_kbit_training(model)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,  # 3,5 good for the Llama 2 Model
    per_device_train_batch_size=8,  # Number of batches that we are going to take for every step
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",  # Not helpful because we donot want to evaluate the model we just want to train it
    eval_steps=1000,  # Evaluate the model after every 1000 steps
    logging_steps=1000,
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
    dataset_text_field="input",
    max_seq_length=1024,  # In dataset creation we put a threshold 2k for context length (input token limit) but we dont have enough VRAM unfortunately it will take a lot of VRAM to put everything into memory so we are just gonna stop at 512
    tokenizer=tokenizer,
    args=training_arguments,
)


trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

wandb.finish()


instruction = "Below is an instruction that describes a SQL generation task, paired with an input that provides further context about the available table schemas. Write SQL code that appropriately answers the request. ### Instruction: What is the release date of Milk and Money? ### Input: CREATE TABLE table_name_50 (release_date VARCHAR, title VARCHAR) ### Response:"
# Using Pipeline from the hugging face
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=256)
result = pipe(instruction)
# Trim the response, remove instruction manually
print(result[0]["generated_text"][len(instruction) :])
