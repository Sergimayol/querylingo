""" Data sources for SQL-to-text and text-to-SQL datasets.
* Hugging face: https://huggingface.co/datasets?sort=likes&search=sql
    - https://huggingface.co/datasets/b-mc2/sql-create-context
    - https://huggingface.co/datasets/Clinton/Text-to-sql-v1
    - https://huggingface.co/datasets/ChrisHayduk/Llama-2-SQL-Dataset
    - https://huggingface.co/datasets/kaxap/pg-wikiSQL-sql-instructions-80k
    - https://huggingface.co/datasets/bugdaryan/sql-create-context-instruction
    - https://huggingface.co/datasets/lamini/spider_text_to_sql
    - https://huggingface.co/datasets/stjarvie/question_to_sql_with_ddl_test_2
    - https://huggingface.co/datasets/Viswa123/sql_context
    - https://huggingface.co/datasets/Rams901/sql-create-context-modified
    - https://huggingface.co/datasets/kaxap/llama2-sql-instruct
    - https://huggingface.co/datasets/NoobLoader/sql-query-db
    - https://huggingface.co/datasets/Mohanakrishnan/sql_query_example
    - https://huggingface.co/datasets/NumbersStation/NSText2SQL
    - https://huggingface.co/datasets/knowrohit07/know_sql
    - https://huggingface.co/datasets/kaxap/llama2-sql-instruct-sys-prompt
    - https://huggingface.co/datasets/kaxap/pg-gpt4SQL-sql-instructions-1k

* Kaggle: https://www.kaggle.com/datasets?search=sql
    - https://www.kaggle.com/datasets/thedevastator/dataset-for-developing-natural-language-interfac
    - https://www.kaggle.com/datasets/kaggle/meta-kaggle-code
    - https://www.kaggle.com/datasets/thedevastator/understanding-contextual-questions-answers

* Github:
    - https://github.com/NumbersStationAI/NSQL/blob/main/data_prep/data/download.sh
    - https://github.com/defog-ai/sql-eval
    - https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2
"""

dataset_endpoints = {
    # https://huggingface.co/datasets/{name}/resolve/main/{files}?download=true
    "huggingface-datasets": {
        "base_url": "https://huggingface.co/datasets/",
        "datasets": [
            {
                "name": "b-mc2/sql-create-context",
                "files": ["sql_create_context_v4.json"],
            },
            {
                "name": "Clinton/Text-to-sql-v1",
                "files": ["texttosqlv2.jsonl"],
            },
            {
                "name": "ChrisHayduk/Llama-2-SQL-Dataset",
                "files": [
                    "data/eval-00000-of-00001-6907aec719559d7d.parquet",
                    "data/train-00000-of-00001-922416e34c5bc71c.parquet",
                    "data/val-00000-of-00001-98c87bd893ed1bdb.parquet",
                ],
            },
            {
                "name": "kaxap/pg-wikiSQL-sql-instructions-80k",
                "files": ["dev.csv", "test.csv", "train.csv"],
            },
            {
                "name": "bugdaryan/sql-create-context-instruction",
                "files": ["data/train-00000-of-00001-ea1a61c2db38e8fc.parquet"],
            },
            {
                "name": "lamini/spider_text_to_sql",
                "files": [
                    "data/train-00000-of-00001-36a24700f19484dc.parquet",
                    "data/validation-00000-of-00001-fa01d04c056ac579.parquet",
                ],
            },
            {
                "name": "stjarvie/question_to_sql_with_ddl_test_2",
                "files": ["data/test-00000-of-00001-3b465c86756391a8.parquet"],
            },
            {
                "name": "Viswa123/sql_context",
                "files": ["train.csv"],
            },
            {
                "name": "Rams901/sql-create-context-modified",
                "files": ["data/train-00000-of-00001-5ac801388cd02781.parquet"],
            },
            {
                "name": "kaxap/llama2-sql-instruct",
                "files": ["train.csv"],
            },
            {
                "name": "NoobLoader/sql-query-db",
                "files": ["train.jsonl"],
            },
            {
                "name": "Mohanakrishnan/sql_query_example",
                "files": ["sql_data_training.csv.csv"],
            },
            {
                "name": "NumbersStation/NSText2SQL",
                "files": ["train.jsonl"],
            },
            {
                "name": "knowrohit07/know_sql",
                "files": ["know_sql_val3{ign}.json"],
            },
            {
                "name": "kaxap/llama2-sql-instruct-sys-prompt",
                "files": ["train.csv"],
            },
            {
                "name": "kaxap/pg-gpt4SQL-sql-instructions-1k",
                "files": ["train.csv"],
            },
        ],
    },
    "kaggle": {
        "base_url": "https://www.kaggle.com/datasets/",
        "datasets": [
            # TODO: Add files
            "thedevastator/dataset-for-developing-natural-language-interfac",
            "kaggle/meta-kaggle-code",
            "thedevastator/understanding-contextual-questions-answers",
        ],
    },
    "github": {
        "base_url": "https://github.com/",
        "datasets": [
            # TODO: Add files
            "NumbersStationAI/NSQL",
            "defog-ai/sql-eval",
        ],
    },
}
