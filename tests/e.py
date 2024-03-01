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


model_path = "./llama-2-3b-sql-coder"


tokenizer = AutoTokenizer.from_pretrained("harborwater/open-llama-3b-everythingLM-2048")
model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_path))


print("Model loaded", model)
pipe: TextGenerationPipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    framework="pt",
)

prompt = "Below is an instruction that describes a SQL generation task, paired with an input that provides further context about the available table schemas. Write SQL code that appropriately answers the request. ### Instruction: What is the release date of Milk and Money? ### Input: CREATE TABLE table_name_50 (release_date VARCHAR, title VARCHAR) ### Response:	"
sequences = pipe(
    prompt,
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    streamer=TextStreamer(tokenizer),
)
