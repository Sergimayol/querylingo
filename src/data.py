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
    - https://huggingface.co/datasets/teknium/openhermes

* Kaggle: https://www.kaggle.com/datasets?search=sql
    - https://www.kaggle.com/datasets/thedevastator/dataset-for-developing-natural-language-interfac
    - https://www.kaggle.com/datasets/kaggle/meta-kaggle-code
    - https://www.kaggle.com/datasets/thedevastator/understanding-contextual-questions-answers

* Github:
    - https://github.com/NumbersStationAI/NSQL/blob/main/data_prep/data/download.sh
    - https://github.com/defog-ai/sql-eval
    - https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2
"""
import argparse
from typing import Dict, List
from utils import Timing, fetch_url, load_json, create_dir, assert_dir, tree_files


def get_args():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--data-dir", "-d", type=str, default="data", help="Directory to save the datasets")
    parser.add_argument("--download", "-dw", type=str, default="none", help="Download datasets from huggingface, kaggle, github, or all", choices=["huggingface", "kaggle", "github", "all", "none"])
    parser.add_argument("--process", "-p", type=str, default="none", help="Process datasets", choices=["huggingface", "kaggle", "github", "all", "none"])
    return parser.parse_args()


# https://huggingface.co/datasets/{name}/resolve/main/{files}?download=true
def download_hf_dataset(dataset: List[Dict[str, List[str]]], base_url: str, data_dir="data") -> list[str]:
    data_dir = data_dir + "/raw/hf"
    create_dir(data_dir)
    for file in dataset["files"]:
        with Timing(f"fetch_url -> {file}: "):
            url = f"{base_url}/{dataset['name']}/resolve/main/{file}?download=true"
            file_name = file.split("/")[1] if "/" in file else file
            file_name = f"{dataset['name'].split('/')[-1]}-{file_name}"
            fetch_url(url, data_dir + "/" + file_name)


def download_kaggle_dataset(dataset: List[Dict[str, List[str]]], base_url: str, data_dir="data"): pass


def download_github_dataset(dataset: List[Dict[str, List[str]]], base_url: str, data_dir="data"): pass


def process_datasets(data_src_dir: str, data_dst_dir: str):
    data_src_dir = data_src_dir + "/raw"
    assert_dir(data_src_dir)
    files_map = tree_files(data_src_dir, exclude=["raw"])
    create_dir(data_dst_dir)
    for fd in files_map: 
        # TODO: Process the datasets
        print(fd, files_map[fd])


if __name__ == "__main__":
    args = get_args()
    if args.download != "none":
        dataset_endpoints = load_json("data/dataset_endpoints.json")
        hf_ds = dataset_endpoints["huggingface-datasets"]["datasets"]
        hf_base_url = dataset_endpoints["huggingface-datasets"]["base_url"]
        print("[INFO] Downloading Hugging Face datasets...")
        for ds in hf_ds: download_hf_dataset(ds, hf_base_url, args.data_dir)
        print("[INFO] Done!")
        # TODO: Add Kaggle and Github datasets
    if args.process != "none": 
        print("[INFO] Processing datasets...")
        process_datasets(args.data_dir, args.data_dir + "/processed")
        print("[INFO] Done!")
