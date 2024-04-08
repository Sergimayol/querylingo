# Translation dataset: google/flan-t5-base 
WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m google/flan-t5-base -hr Sergi28/text-2-sql-4-llm -d 'Translation dataset' -tm ./models/bloom.tmodules.json --logging_steps 150 -e 10 --eval_steps 700

# Translation dataset: mrm8488/t5-base-finetuned-wikiSQL
WANDB=1 WORKERS=$(nproc) DEBUG=2 python src/train.py -m mrm8488/t5-base-finetuned-wikiSQL -hr Sergi28/text-2-sql-4-llm -d 'Translation dataset' -tm ./models/bloom.tmodules.json --logging_steps 150 -e 10 --eval_steps 700