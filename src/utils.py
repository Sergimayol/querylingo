from pandas import DataFrame
from tqdm import tqdm
from torch import Tensor, device
from typing import Any, List, Optional, Dict
from safetensors import safe_open
import os, time, contextlib, cProfile, pstats, urllib3, json

DEBUG = int(os.getenv("DEBUG", 0)) # 0, ..., 4
CACHE_DIR = os.getenv("CACHE_DIR", os.path.expanduser("~/.cache"))
DEVICE = os.getenv("DEVICE", "cpu") # "cpu" or "cuda"
WANDB = bool(os.getenv("WANDB", 0)) # "True" or "False" | "1" or "0"
WEIGHTS = bool(os.getenv("WEIGHTS", 1)) # "True" or "False" | "1" or "0"
WORKERS = int(os.getenv("WORKERS", 1)) # 0, 1, 2, ...
CACHE = bool(os.getenv("CACHE", 1)) # "True" or "False" | "1" or "0"


def fetch_url(url: str, filename: str = None, buffer_size: int = 16384, force_download: bool = False) -> str:
    if not force_download and filename is not None and os.path.exists(filename): return filename
    if filename is None: filename = url.split("/")[-1]
    http = urllib3.PoolManager()
    with http.request("GET", url, preload_content=False) as response, open(filename, "wb") as out_file:
        if response.status not in range(200, 300): raise Exception(f"Failed to fetch {url}")
        total_size = int(response.headers.get("Content-Length", 0))
        p_bar = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, miniters=1)
        for data in response.stream(buffer_size):
            out_file.write(data)
            p_bar.update(len(data))
        p_bar.close()
    return filename

def read_lines(file: str) -> List[str]:
    with open(file, "r") as f: data = f.read()
    return data.split("\n")[:-1] if data.endswith("\n") else data.split("\n")
def read_file(file: str) -> str:
    with open(file, "r") as f:
        data = f.read()
    return data
def pt_device(dev: str = DEVICE) -> device: return device(dev)
def load_json(file: str) -> Dict: return json.load(open(file, "r"))
def load_jsonl(file: str) -> List[Any]: return [json.loads(line) for line in read_lines(file)]
def assert_dir(dir: str) -> None: assert os.path.exists(dir), f"Directory {dir} does not exist"
def create_dir(dir: str) -> None: os.makedirs(dir, exist_ok=True) if not os.path.exists(dir) else None
def tree_files(dir: str, exclude: List[str] = None) -> Dict[str, List[str]]: return {os.path.basename(folder): files for folder, _, files in os.walk(dir) if os.path.basename(folder) not in exclude}
def df_to_jsonl(df: DataFrame, file: str, orient="records", lines=True, **kwargs) -> None: df.to_json(file, orient=orient, lines=lines, **kwargs)
def load_safetenors(file: str, device="cpu", verbose=False) -> Dict[str, Tensor]:
    tensors = {}
    with safe_open(file, framework="pt", device=device) as f:
        t = tqdm(f.keys(), desc="Loading tensors", disable=not verbose)
        for key in t:
            tensors[key] = f.get_tensor(key)
            t.set_postfix_str(key)
    return tensors
def write_sf_keys(file: str, tensors: Dict[str, Tensor], verbose=False) -> None:
    with open(file, "w") as f:
        t = tqdm(tensors.keys(), desc="Writing keys", disable=not verbose)
        f.write("\n".join([f"{key}: {tensors[key].size()}" for key in t]))
def apply_parallelization(func, data, workers=WORKERS) -> List[Any]:
    from multiprocessing import Pool
    with Pool(workers) as p:
        data = list(p.imap(func, data))
    return data

# https://github.com/tinygrad/tinygrad/blob/ee25f732831b39c64698f8728cfe338ba9662866/tinygrad/helpers.py#L96
class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True):
        self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled

    def __enter__(self): self.st = time.perf_counter_ns()

    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled: print(f"{self.prefix}{self.et*1e-6:.2f} ms" + (self.on_exit(self.et) if self.on_exit else ""))

# https://github.com/tinygrad/tinygrad/blob/ee25f732831b39c64698f8728cfe338ba9662866/tinygrad/helpers.py#L24
def colored(st, color: Optional[str], background=False) -> str:
    return (
        f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m"
        if color is not None
        else st
    )

# https://github.com/tinygrad/tinygrad/blob/ee25f732831b39c64698f8728cfe338ba9662866/tinygrad/helpers.py#L103
def _format_fcn(fcn) -> str: return f"{fcn[0]}:{fcn[1]}:{fcn[2]}"

# https://github.com/tinygrad/tinygrad/blob/ee25f732831b39c64698f8728cfe338ba9662866/tinygrad/helpers.py#L104
class Profiling(contextlib.ContextDecorator):
    def __init__(self, enabled=True, sort="cumtime", frac=0.2, fn=None, ts=1):
        self.enabled, self.sort, self.frac, self.fn, self.time_scale = enabled, sort, frac, fn, 1e3 / ts

    def __enter__(self):
        self.pr = cProfile.Profile()
        if self.enabled: self.pr.enable()

    def __exit__(self, *exc):
        if not self.enabled: return
        self.pr.disable()
        if self.fn: self.pr.dump_stats(self.fn)
        stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
        for fcn in stats.fcn_list[0 : int(len(stats.fcn_list) * self.frac)]:
            (_, num_calls, tottime, cumtime, callers) = stats.stats[fcn]
            scallers = sorted(callers.items(), key=lambda x: -x[1][2])
            print(
                f"n:{num_calls:8d}  tm:{tottime*self.time_scale:7.2f}ms  tot:{cumtime*self.time_scale:7.2f}ms",
                colored(_format_fcn(fcn), "yellow") + " " * (50 - len(_format_fcn(fcn))),
                colored(f"<- {(scallers[0][1][2]/tottime)*100:3.0f}% {_format_fcn(scallers[0][0])}", "BLACK") if len(scallers) else "",
            )
