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
import argparse, pandas as pd, re
from typing import Any, Dict, List, Optional, Tuple
from utils import Timing, fetch_url, load_json, create_dir, assert_dir, tree_files, load_jsonl


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

def _extract_patterns(text) -> Optional[Tuple[str | Any, ...]]:
    match = re.compile(r'###question:(.*?)###answer:(.*?)###context:(.*?)$', re.DOTALL).search(text)
    return match.groups() if match else None

def _process_csv(file: str) -> Optional[pd.DataFrame]:
    df = pd.read_csv(file)
    cols = df.columns
    if len(cols) < 3:
        if "combined_text" in cols: # Patern: ###question: ... ###answer: ... ###context: ...
            df[["question", "answer", "context"]] = df["combined_text"].apply(_extract_patterns).apply(pd.Series)
            df.rename(columns={"combined_text": "extra"}, inplace=True)
            return df[["question", "context", "answer", "extra"]]
        else: return None # System prompt, SQL, etc -> Instruct datasets
    if len(cols) > 3:
        df["extra"] = df[cols[3:]].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        df = pd.concat([df[cols[:3]], df["extra"]], axis=1)
    else: df["extra"] = "NULL"
    df.rename(columns={cols[0]: "question", cols[1]: "context", cols[2]: "anwser"}, inplace=True)
    return df

def _process_json(file: str) -> pd.DataFrame:
    df = pd.read_json(file)
    cols = df.columns
    df["extra"] = "NULL"
    df.rename(columns={cols[0]: "question", cols[1]: "context", cols[2]: "anwser"}, inplace=True)
    return df

def _process_jsonl(file: str) -> Optional[pd.DataFrame]:
    df = pd.DataFrame(load_jsonl(file))
    cols = df.columns
    if len(cols) < 3:
        if "Questions" in cols:
            df["extra"] = "NULL"
            df["context"] = "NULL"
            df.rename(columns={"Questions": "question", cols[1]: "anwser"}, inplace=True)
            return df[["question", "context", "anwser", "extra"]]
        else: return None # Unknown format
    if len(cols) > 3:
        df["extra"] = df[cols[3:]].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        df = pd.concat([df[cols[:3]], df["extra"]], axis=1)
    else: df["extra"] = "NULL"
    df.rename(columns={cols[0]: "question", cols[1]: "context", cols[2]: "anwser"}, inplace=True)
    return df

def _process_parquet(file: str) -> Optional[pd.DataFrame]:
    df = pd.read_parquet(file)
    cols = df.columns
    if len(cols) < 3: return None # Instructions, SQL, etc -> Instruct datasets
    if len(cols) > 3:
        df["extra"] = df[cols[3:]].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        df = pd.concat([df[cols[:3]], df["extra"]], axis=1)
    else: df["extra"] = "NULL"
    df = df[[cols[0], cols[2], cols[1], "extra"]]
    df.rename(columns={cols[0]: "question", cols[1]: "context", cols[2]: "anwser"}, inplace=True)
    return df

def process_datasets(data_src_dir: str, data_dst_dir: str):
    data_src_dir = data_src_dir + "/raw"
    assert_dir(data_src_dir)
    files_map = tree_files(data_src_dir, exclude=["raw"]) 
    for fd in files_map:
        create_dir(f"{data_dst_dir}/{fd}")
        for file in files_map[fd]:
            print(f"[INFO] Processing {file}...")
            with Timing(f"[INFO] {file} processed in: "):
                ext, df = file.split(".")[-1], None
                # TODO: See what to do with the Instruction datasets, for now just skip them
                if ext == "csv": df = _process_csv(f"{data_src_dir}/{fd}/{file}")
                elif ext == "json": df = _process_json(f"{data_src_dir}/{fd}/{file}")
                elif ext == "jsonl": df = _process_jsonl(f"{data_src_dir}/{fd}/{file}")
                elif ext == "parquet": df = _process_parquet(f"{data_src_dir}/{fd}/{file}")
                else: pass # Skip unknown formats
                if df is not None: df.to_csv(f"{data_dst_dir}/{fd}/{file.replace(f'.{ext}', '.csv')}", index=False)
                else: print(f"[WARN] Unable to process {file}...")

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
