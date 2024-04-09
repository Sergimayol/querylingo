# Train Runs

## Systems prompt dataset

### OpenAI GPT2 (Samall)

WANDB=1 WORKERS=10 DEBUG=2 python src/train.py -m openai-community/gpt2 -hr Sergi28/text-2-sql-4-llm -d 'System Prompt dataset' -tm ./models/gpt2.tmodules.json --logging_steps 100 -e 5 --eval_steps 250

WANDB=1 WORKERS=10 DEBUG=2 python src/train.py -m openai-community/gpt2 -hr Sergi28/text-2-sql-4-llm -d 'System Prompt dataset' -tm ./models/gpt2.tmodules.json --logging_steps 100 -e 5 --eval_steps 250 -lr 2-e3

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m openai-community/gpt2 -hr Sergi28/text-2-sql-4-llm -d 'Text generation dataset' --logging_steps 100 -e 5 --eval_steps 250

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m openai-community/gpt2 -hr Sergi28/text-2-sql-4-llm -d 'Instruct dataset' --logging_steps 100 -e 5 --eval_steps 250

### OpenAI GPT2 (Large)

WANDB=1 WORKERS=10 DEBUG=2 python src/train.py -m openai-community/gpt2-large -hr Sergi28/text-2-sql-4-llm -d 'System Prompt dataset' -tm ./models/gpt2.tmodules.json --logging_steps 100 -e 5 --eval_steps 250

WANDB=1 WORKERS=10 DEBUG=2 python src/train.py -m openai-community/gpt2-large -hr Sergi28/text-2-sql-4-llm -d 'System Prompt dataset' -tm ./models/gpt2.tmodules.json --logging_steps 100 -e 5 --eval_steps 250 -lr 2-e3

### TinyLLama

WANDB=1 WORKERS=10 DEBUG=2 python src/train.py -m Maykeye/TinyLLama-v0 -hr Sergi28/text-2-sql-4-llm -d 'System Prompt dataset' -tm ./models/tinyllama.tmodules.json --logging_steps 100 -e 5 --eval_steps 250

WANDB=1 WORKERS=10 DEBUG=2 python src/train.py -m Maykeye/TinyLLama-v0 -hr Sergi28/text-2-sql-4-llm -d 'System Prompt dataset' -tm ./models/tinyllama.tmodules.json --logging_steps 100 -e 5 --eval_steps 250 -lr 2-e3

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m Maykeye/TinyLLama-v0 -hr Sergi28/text-2-sql-4-llm -d 'Text completition dataset' -tm ./models/tinyllama.tmodules.json --logging_steps 100 -e 5 --eval_steps 250

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m Maykeye/TinyLLama-v0 -hr Sergi28/text-2-sql-4-llm -d 'Chatbot instruct dataset' -tm ./models/tinyllama.tmodules.json --logging_steps 100 -e 5 --eval_steps 250

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m Maykeye/TinyLLama-v0 -hr Sergi28/text-2-sql-4-llm -d 'Translation dataset' -tm ./models/tinyllama.tmodules.json --logging_steps 100 -e 5 --eval_steps 250

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m Maykeye/TinyLLama-v0 -hr Sergi28/text-2-sql-4-llm -d 'Translation dataset' -tm ./models/tinyllama.tmodules.json --logging_steps 100 -e 5 --eval_steps 250

### Bloom

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m bigscience/bloomz-560m -hr Sergi28/text-2-sql-4-llm -d 'System Prompt dataset' -tm ./models/bloom.tmodules.json --logging_steps 100 -e 10 --eval_steps 250

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m bigscience/bloomz-560m -hr Sergi28/text-2-sql-4-llm -d 'Translation dataset' -tm ./models/bloom.tmodules.json --logging_steps 100 -e 10 --eval_steps 250 -lr 2e-3

## keeeeenw/MicroLlama

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m keeeeenw/MicroLlama -hr Sergi28/text-2-sql-4-llm -d 'Text generation dataset' -tm ./models/bloom.tmodules.json --logging_steps 100 -e 10 --eval_steps 250

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m keeeeenw/MicroLlama -hr Sergi28/text-2-sql-4-llm -d 'Translation dataset' -tm ./models/bloom.tmodules.json --logging_steps 100 -e 10 --eval_steps 250

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m keeeeenw/MicroLlama -hr Sergi28/text-2-sql-4-llm -d 'Instruct dataset' -tm ./models/bloom.tmodules.json --logging_steps 100 -e 10 --eval_steps 250

## google/flan-t5-base

WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m google/flan-t5-base -hr Sergi28/text-2-sql-4-llm -d 'Translation dataset' -tm ./models/bloom.tmodules.json --logging_steps 100 -e 10 --eval_steps 250
