from tqdm import tqdm
from typing import Optional
import time, contextlib, cProfile, pstats, urllib3  # noqa: E401


def fetch_url(url: str, filename: str = None, buffer_size: int = 16384) -> str:
    if filename is None:
        filename = url.split("/")[-1]
    http = urllib3.PoolManager()
    with http.request("GET", url, preload_content=False) as response, open(filename, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        p_bar = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, miniters=1)
        for data in response.stream(buffer_size):
            out_file.write(data)
            p_bar.update(len(data))
        p_bar.close()
    return filename


# https://github.com/tinygrad/tinygrad/blob/ee25f732831b39c64698f8728cfe338ba9662866/tinygrad/helpers.py#L96
class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True):
        self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled

    def __enter__(self):
        self.st = time.perf_counter_ns()

    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled:
            print(f"{self.prefix}{self.et*1e-6:.2f} ms" + (self.on_exit(self.et) if self.on_exit else ""))


# https://github.com/tinygrad/tinygrad/blob/ee25f732831b39c64698f8728cfe338ba9662866/tinygrad/helpers.py#L24
def colored(st, color: Optional[str], background=False) -> str:
    return (
        f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m"
        if color is not None
        else st
    )


# https://github.com/tinygrad/tinygrad/blob/ee25f732831b39c64698f8728cfe338ba9662866/tinygrad/helpers.py#L103
def _format_fcn(fcn) -> str:
    return f"{fcn[0]}:{fcn[1]}:{fcn[2]}"


# https://github.com/tinygrad/tinygrad/blob/ee25f732831b39c64698f8728cfe338ba9662866/tinygrad/helpers.py#L104
class Profiling(contextlib.ContextDecorator):
    def __init__(self, enabled=True, sort="cumtime", frac=0.2, fn=None, ts=1):
        self.enabled, self.sort, self.frac, self.fn, self.time_scale = enabled, sort, frac, fn, 1e3 / ts

    def __enter__(self):
        self.pr = cProfile.Profile()
        if self.enabled:
            self.pr.enable()

    def __exit__(self, *exc):
        if self.enabled:
            self.pr.disable()
            if self.fn:
                self.pr.dump_stats(self.fn)
            stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
            for fcn in stats.fcn_list[0 : int(len(stats.fcn_list) * self.frac)]:
                (_, num_calls, tottime, cumtime, callers) = stats.stats[fcn]
                scallers = sorted(callers.items(), key=lambda x: -x[1][2])
                print(
                    f"n:{num_calls:8d}  tm:{tottime*self.time_scale:7.2f}ms  tot:{cumtime*self.time_scale:7.2f}ms",
                    colored(_format_fcn(fcn), "yellow") + " " * (50 - len(_format_fcn(fcn))),
                    colored(f"<- {(scallers[0][1][2]/tottime)*100:3.0f}% {_format_fcn(scallers[0][0])}", "BLACK") if len(scallers) else "",
                )
