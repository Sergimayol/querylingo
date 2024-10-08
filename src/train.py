import wandb, torch, argparse
from typing import List, Optional, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from utils import CACHE_DIR, DEBUG, WORKERS, WANDB, load_json


def get_tokenizer(base_model) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if any(k in base_model for k in ["gpt", "opt", "bloom"]) else "right"
    return tokenizer


def get_model(model: str, bnb_conf: BitsAndBytesConfig, use_kbit: bool = False) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb_conf, device_map={"": 0}, cache_dir=CACHE_DIR)
    return prepare_model_for_kbit_training(model) if use_kbit else model


def get_training_args(args: argparse.Namespace):
    if DEBUG >= 1: print(f"Hyperparameters: {args}")
    return TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        evaluation_strategy=args.eval_strategy,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        eval_accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        output_dir=f"{args.output_dir}_{wandb.run.name}" if WANDB else args.output_dir,
        optim=args.optim,
        report_to="wandb" if WANDB else None,
        save_steps=250,
    )


def get_target_modules(target_modules) -> Optional[List[str]]:
    return load_json(target_modules)["target_modules"] if target_modules is not None else None


def get_dataset(repo: str, dataset: str) -> Tuple[Dataset, Dataset]:
    dataset: Dataset = load_dataset(repo, dataset)["train"]
    dataset_split = dataset.train_test_split(test_size=0.3)
    print(f"Train size: {len(dataset_split['train'])}, Test size: {len(dataset_split['test'])}")
    return dataset_split["train"], dataset_split["test"].select(range(min(1000, len(dataset_split["test"]))))


def get_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", "-m", type=str, default="openai-community/gpt2", help="Model to train")
    parser.add_argument("--use_kbit", action="store_true", default=True, help="Use kbit training")
    # Hyperparameters
    parser.add_argument("--epochs", "-e", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--eval_accumulation_steps", type=int, default=8, help="Evaluation accumulation steps")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--warmup_steps", type=int, default=2, help="Warmup steps")
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit", help="Optimizer")
    parser.add_argument(
        "--target_modules",
        "-tm",
        type=str,
        help="Config file target modules for LoRA Config. Default: ./models/lora_target_modules.json",
    )
    # Dataset
    parser.add_argument("--hf_repo", "-hr", type=str, help="Hugging Face repository", required=True)
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Dataset to use. Provide the name between quotes. Example: 'Chatbot dataset'",
        choices=[
            "Chatbot dataset",
            "Chatbot instruct dataset",
            "System Prompt dataset",
            "Text completition dataset",
            "Text generation dataset",
            "Instruct dataset",
            "Translation dataset",
        ],
        required=True,
    )
    parser.add_argument("--text_field", type=str, default="text", help="Text field")
    parser.add_argument("--max_length", type=int, default=1024, help="Max length")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    base_model = args.model
    use_kbit = args.use_kbit

    tokenizer = get_tokenizer(base_model)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = get_model(base_model, bnb_config, use_kbit)
    if DEBUG >= 2: print(model, f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M parameters", sep="\n")

    tm = get_target_modules(args.target_modules)
    if DEBUG >= 2: print(f"Target modules: {tm}")
    lora_config = LoraConfig(r=8, target_modules=tm, task_type="CAUSAL_LM")

    train_dataset, eval_dataset = get_dataset(args.hf_repo, args.dataset)

    print(f"Using wandb: {WANDB}")
    name = f"{base_model.split('/')[-1]}_{args.dataset}".replace(" ", "_")
    wandb.init(mode="disabled" if not WANDB else "online", project=name, config=dict(args._get_kwargs()))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        args=get_training_args(args),
        dataset_text_field=args.text_field,
        max_seq_length=args.max_length,
        dataset_num_proc=WORKERS,
    )

    stats = trainer.train()

    trainer.save_model(f"outputs_{wandb.run.id}" if WANDB else "outputs")

    wandb.log(stats)
    wandb.finish()
