# Install following libraries
# pip install torch transformers datasets bitsandbytes accelerate peft

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, LoftQConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import LlamaTokenizer, set_seed
from utils import CACHE_DIR

device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 1000
set_seed(seed)

# Load dataset
dataset = load_dataset("yelp_review_full", split="train[:1%]")
dataset = dataset.train_test_split(test_size=0.2, seed=seed)
dataset = dataset.rename_column("label", "labels")

model_name = "elyza/ELYZA-japanese-Llama-2-7b"
# model_name = "mistralai/Mistral-7B-v0.1"

if any(k in model_name for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side, cache_dir=CACHE_DIR)
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=4096, return_tensors="pt")


tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=1)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    save_in_8bit=True,
    quantize_in_8bit=True,
)

lora_config = LoraConfig(
    r=16, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS
)

# LoRA quantization
accelerator = Accelerator()
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, quantization_config=bnb_config, device_map=device)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


print(model)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=1,
    dataloader_pin_memory=False,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()
