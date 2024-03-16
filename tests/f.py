import json, os
from typing import List

from tqdm import tqdm
from utils import load_jsonl, apply_parallelization


def get_files_from_dir(dir: str) -> List[str]: return [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith(".jsonl")]

def _process_file_wrapper(args): return process_file(*args)
def process_file(file: str) -> None:
    data = load_jsonl(file)
    file_name = file.split("/")[-1]
    with open(f"non_ascii_{file_name}", "a") as f:
        for line in tqdm(data, desc=f"Processing {file}"):
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    files = get_files_from_dir("/mnt/d/tfg/processed/datasets")
    print(files)
    apply_parallelization(_process_file_wrapper, [(file,) for file in files], workers=len(files))
