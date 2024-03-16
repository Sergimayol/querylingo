# Querylingo

Querylingo is a working in process LLM to parse Natural Language Queries to SQL and the other way around. This project is part of my Bachelor Thesis at the University.

## Main Goal

The main goal of this project is to create a tool that can parse natural language queries to SQL and the other way around. This tool should be able to understand the user's query and translate it to a SQL query that can be executed in a database. The tool should also be able to translate a SQL query to a natural language query that can be understood by the user.

## Getting Started

-   Create datasets:

```bash
WORKERS=8 DEBUG=3 python src/data.py -dw -d /mnt/d/tfg/ -e /mnt/d/tfg/processed/hf/ /mnt/d/tfg/processed/ -p -s jsonl -cd /mnt/d/tfg/processed/datasets.sqlite /mnt/d/tfg/processed/datasets/
```

Dataset can be downloaded from [here (hugginface)](https://huggingface.co/datasets/Sergi28/text-2-sql-4-llm)

-   Train model:

```bash
python src/train.py -h
```

```bash
usage: train.py [-h] [--model MODEL] [--use_kbit] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--accumulation_steps ACCUMULATION_STEPS] [--eval_strategy EVAL_STRATEGY]
                [--warmup_steps WARMUP_STEPS] [--eval_steps EVAL_STEPS] [--learning_rate LEARNING_RATE] [--fp16] [--logging_steps LOGGING_STEPS] [--output_dir OUTPUT_DIR]
                [--optim OPTIM] [--text_field TEXT_FIELD] [--max_length MAX_LENGTH]

Train a model

options:
  -h, --help            show this help message and exit
  --model MODEL         Model to train
  --use_kbit            Use kbit training
  --epochs EPOCHS       Number of epochs
  --batch_size BATCH_SIZE
                        Batch size
  --accumulation_steps ACCUMULATION_STEPS
                        Gradient accumulation steps
  --eval_strategy EVAL_STRATEGY
                        Evaluation strategy
  --warmup_steps WARMUP_STEPS
                        Warmup steps
  --eval_steps EVAL_STEPS
                        Evaluation steps
  --learning_rate LEARNING_RATE
                        Learning rate
  --fp16                Use fp16
  --logging_steps LOGGING_STEPS
                        Logging steps
  --output_dir OUTPUT_DIR
                        Output directory
  --optim OPTIM         Optimizer
  --text_field TEXT_FIELD
                        Text field
  --max_length MAX_LENGTH
                        Max length
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
