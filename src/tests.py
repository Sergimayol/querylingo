from utils import fetch_url, Timing, Profiling
from data import dataset_endpoints

"""
with Timing("fetch_url: "):
    fetch_url("https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz")

with Profiling(enabled=True):
    fetch_url("https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz")

with Timing("fetch_url: "):
    dataset = dataset_endpoints["huggingface-datasets"]["datasets"][2]
    for file in dataset["files"]:
        fetch_url(f"https://huggingface.co/datasets/{dataset['name']}/resolve/main/{file}?download=true", file)
"""
