"""Data sources for SQL-to-text and text-to-SQL datasets.
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
import argparse, pandas as pd, re, sqlite3
from typing import Any, Dict, List, Optional, Tuple
from utils import WORKERS, DEBUG, CACHE, Profiling, Timing, df_to_jsonl, fetch_url, load_json, create_dir, assert_dir, read_file, tree_files, load_jsonl, apply_parallelization


# https://huggingface.co/datasets/{name}/resolve/main/{files}?download=true
def download_hf_dataset(dataset: List[Dict[str, List[str]]], base_url: str, data_dir="data"):
    data_dir = data_dir + "/raw/hf"
    create_dir(data_dir)
    for file in dataset["files"]:
        with Timing(f"fetch_url -> {file}: ", enabled=DEBUG >= 1):
            url = f"{base_url}/{dataset['name']}/resolve/main/{file}?download=true"
            file_name = file.split("/")[1] if "/" in file else file
            file_name = f"{dataset['name'].split('/')[-1]}-{file_name}"
            fetch_url(url, data_dir + "/" + file_name, force_download=not CACHE)

def _extract_patterns(text) -> Optional[Tuple[str | Any, ...]]:
    match = re.compile(r'###question:(.*?)###answer:(.*?)###context:(.*?)$', re.DOTALL).search(text)
    return match.groups() if match else None

def _process_csv(file: str) -> Tuple[pd.DataFrame, bool]:
    df = pd.read_csv(file)
    cols = df.columns
    if len(cols) < 3:
        if "combined_text" in cols: # Patern: ###question: ... ###answer: ... ###context: ...
            df[["question", "answer", "context"]] = df["combined_text"].apply(_extract_patterns).apply(pd.Series)
            df.rename(columns={"combined_text": "extra"}, inplace=True)
            return df[["question", "context", "answer", "extra"]], False
        else: return df, True # System prompt, SQL, etc -> Instruct datasets
    if len(cols) > 3:
        df["extra"] = df[cols[3:]].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        df = pd.concat([df[cols[:3]], df["extra"]], axis=1)
    else: df["extra"] = "NULL"
    df.rename(columns={cols[0]: "question", cols[1]: "context", cols[2]: "answer"}, inplace=True)
    return df, False

def _process_json(file: str) -> Tuple[pd.DataFrame, bool]:
    df = pd.read_json(file)
    cols = df.columns
    df["extra"] = "NULL"
    df.rename(columns={cols[0]: "question", cols[1]: "context", cols[2]: "answer"}, inplace=True)
    return df, False

def _process_jsonl(file: str) -> Tuple[pd.DataFrame, bool]:
    df = pd.DataFrame(load_jsonl(file))
    cols = df.columns
    if len(cols) < 3:
        if "Questions" in cols:
            df["extra"] = "NULL"
            df["context"] = "NULL"
            df.rename(columns={"Questions": "question", cols[1]: "answer"}, inplace=True)
            return df[["question", "context", "answer", "extra"]], False
        else: return df, True # System prompt, SQL, etc -> Instruct datasets
    if len(cols) > 3:
        df["extra"] = df[cols[3:]].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        df = pd.concat([df[cols[:3]], df["extra"]], axis=1)
    else: df["extra"] = "NULL"
    if "source" in cols and len(cols) == 3:
        df["question"] = "NULL"
        df.rename(columns={cols[0]: "context", cols[1]: "answer"}, inplace=True)
    else: df.rename(columns={cols[0]: "question", cols[1]: "context", cols[2]: "answer"}, inplace=True)
    df = df[["question", "context", "answer", "extra"]]
    return df, False

def _process_parquet(file: str) -> Tuple[pd.DataFrame, bool]:
    df = pd.read_parquet(file)
    cols = df.columns
    if len(cols) < 3: return df, True # System prompt, SQL, etc -> Instruct datasets
    if len(cols) > 3:
        df["extra"] = df[cols[3:]].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)
        df = pd.concat([df[cols[:3]], df["extra"]], axis=1)
    else: df["extra"] = "NULL"
    df = df[[cols[0], cols[2], cols[1], "extra"]]
    cols = df.columns
    df.rename(columns={cols[0]: "question", cols[1]: "context", cols[2]: "answer"}, inplace=True)
    return df, False


def _process_dataset_wrapper(args: Tuple[str, str, str, str]): return _process_dataset(*args)
def _process_dataset(data_src_dir: str, data_dst_dir: str, fd: str, file: str, save: str = "jsonl"):
    create_dir(f"{data_dst_dir}/{fd}")
    if DEBUG >= 2: print(f"[INFO] Processing {file}...")
    with Timing(f"_process_dataset -> {file} processed in: ", enabled=DEBUG >= 1):
        ext, df, is_inst = file.split(".")[-1], None, False
        # TODO: See what to do with the Instruction datasets, for now just skip them
        if ext == "csv": df, is_inst = _process_csv(f"{data_src_dir}/{fd}/{file}")
        elif ext == "json": df, is_inst = _process_json(f"{data_src_dir}/{fd}/{file}")
        elif ext == "jsonl": df, is_inst = _process_jsonl(f"{data_src_dir}/{fd}/{file}")
        elif ext == "parquet": df, is_inst = _process_parquet(f"{data_src_dir}/{fd}/{file}")
        else: print(f"[WARN] Unknown format: {ext}...") # Skip unknown formats
        if df is not None:
            dir = f"{data_dst_dir}/{fd}/{"instruct" if is_inst else "generation"}"
            create_dir(dir)
            if save == "csv": df.to_csv(f"{dir}/{file.replace(f'.{ext}', '.csv')}", index=False)
            else: df_to_jsonl(df, f"{dir}/{file.replace(f'.{ext}', '.jsonl')}", index=False)
        else: print(f"[WARN] Unable to process {file}...")

def process_datasets(data_src_dir: str, data_dst_dir: str, save: str = "jsonl"):
    data_src_dir = data_src_dir + "/raw"
    assert_dir(data_src_dir)
    files_map = tree_files(data_src_dir, exclude=["raw"])
    if WORKERS > 1:
        for fd in files_map:
            apply_parallelization(_process_dataset_wrapper, [(data_src_dir, data_dst_dir, fd, file, save) for file in files_map[fd]], workers=WORKERS)
    else:
        for fd in files_map:
            for file in files_map[fd]:
                _process_dataset(data_src_dir, data_dst_dir, fd, file, save)

def export_processed_datasets(data_src_dir: str, data_dst_dir: str):
    data_src_dir = data_src_dir if data_src_dir[-1] != "/" else data_src_dir[:-1]
    data_dst_dir = data_dst_dir if data_dst_dir[-1] != "/" else data_dst_dir[:-1]
    assert_dir(data_src_dir)
    files_map = tree_files(data_src_dir, exclude=["processed"])
    files_map = {fd: [f for f in files_map[fd] if not f.endswith(".sqlite")] for fd in files_map}
    all_dfs: List[Tuple[pd.DataFrame, bool]] = []
    db_uri = f"{data_dst_dir}/datasets{"_debug" if DEBUG >= 4 else ""}.sqlite"
    conn = sqlite3.connect(db_uri)
    for fd in files_map:
        create_dir(f"{data_dst_dir}")
        for file in files_map[fd]:
            ext = file.split(".")[-1]
            if ext not in ["csv", "jsonl"]: continue
            print(f"[INFO] Exporting {file} to SQL ({db_uri}) ...")
            with Timing(f"export_processed_datasets -> {file} exported in: ", enabled=DEBUG >= 1):
                if ext == "csv": df = pd.read_csv(f"{data_src_dir}/{fd}/{file}")
                else: df = pd.DataFrame(load_jsonl(f"{data_src_dir}/{fd}/{file}"))
                if DEBUG >=2: df["source"] = file
                all_dfs.append((df, fd == "instruct"))
                df.to_sql(file.replace(f".{ext}", ""), conn, if_exists="replace")
    print(f"[INFO] Exporting all datasets to SQL ({db_uri}) ...")
    with Timing("export_processed_datasets -> All datasets exported in: ", enabled=DEBUG >= 1):
        # TODO: Apply parallelization
        pd.concat([df for df, is_inst in all_dfs if not is_inst], ignore_index=True).to_sql("raw_text_generation", conn, if_exists="replace")
        pd.concat([df for df, is_inst in all_dfs if is_inst], ignore_index=True).to_sql("raw_instructs", conn, if_exists="replace")
        pd.concat([df for df, _ in all_dfs], ignore_index=True).to_sql("all_datasets", conn, if_exists="replace")
    conn.close()

def create_dataset(data_src_dir: str, data_dst_dir: str, save: str = "jsonl"):
    create_dir(data_dst_dir)
    conn = sqlite3.connect(data_src_dir)
    cursor = conn.cursor()
    sql = read_file("./data/create_datasets.sql")
    if DEBUG >= 2: print(f"[INFO] Creating datasets from {data_src_dir}...")
    with Timing("create_dataset -> All datasets created in SQL in: ", enabled=DEBUG >= 1):
        cursor.executescript(sql)
    conn.commit()
    tables: List[Tuple[str]] = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    tables = [t[0] for t in tables if t[0].startswith("dataset_")]
    for table in tables:
        with Timing(f"create_dataset -> {table} created in: ", enabled=DEBUG >= 1):
            if DEBUG >= 2: print(f"[INFO] Creating {table}...")
            # TODO: Apply parallelization
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            if save == "jsonl": df_to_jsonl(df, f"{data_dst_dir}/{table}.jsonl", index=False)
            else: df.to_csv(f"{data_dst_dir}/{table}.csv", index=False)
    conn.close()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--data-dir", "-d", type=str, default="data", help="Directory to save the datasets")
    parser.add_argument("--download", "-dw", action="store_true", help="Download datasets from huggingface")
    parser.add_argument("--process", "-p", action="store_true", help="Process datasets and save them to CSV or JSONL files depending on the flag --save (default: JSONL)")
    parser.add_argument("--save", "-s", type=str, default="jsonl", help="Save processed datasets to CSV or JSONL files", choices=["csv", "jsonl"])
    parser.add_argument("--export", "-e", type=str, nargs=2, help="Export processed datasets to SQL")
    parser.add_argument("--create-dataset", "-cd", type=str, nargs=2, help="Path where to find the SQLite DB and path to create the datasets from the processed datasets to JSON or CSV file depending on the flag --save (default: JSONL)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(f"[INFO] Debug level: {DEBUG}")
    data_dir = args.data_dir if args.data_dir[-1] != "/" else args.data_dir[:-1]
    if args.download:
        print("[INFO] Downloading Hugging Face datasets...")
        with Profiling(enabled=DEBUG >= 3):
            dataset_endpoints = load_json("./data/dataset_endpoints.json")
            hf_ds = dataset_endpoints["huggingface-datasets"]["datasets"]
            for ds in hf_ds: download_hf_dataset(ds, dataset_endpoints["huggingface-datasets"]["base_url"], data_dir)
        print("[INFO] Done!")

    if args.process:
        print("[INFO] Processing datasets...")
        with Profiling(enabled=DEBUG >= 3):
            process_datasets(data_dir, f"{data_dir}/processed", args.save)
        print("[INFO] Done!")

    if args.export is not None:
        print("[INFO] Exporting datasets to SQL...")
        with Profiling(enabled=DEBUG >= 3):
            export_processed_datasets(args.export[0], args.export[1])
        print("[INFO] Done!")

    if args.create_dataset is not None:
        print("[INFO] Creating dataset from processed datasets...")
        with Profiling(enabled=DEBUG >= 3):
            create_dataset(args.create_dataset[0], args.create_dataset[1], args.save)
        print("[INFO] Done!")
