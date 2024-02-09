# pip install -q transformers accelerate
import torch, os, warnings  # noqa: E401
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, LlamaTokenizer, LlamaForCausalLM
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


warnings.filterwarnings("ignore")
checkpoint = "bigscience/bloomz-560m"
checkpoint = "bigscience/bloom-3b"
checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=os.path.join("D:", "models")).to(device)
tok = AutoTokenizer.from_pretrained(checkpoint)

print("[INFO] Using model:", checkpoint)

while True:
    print("Prompt: ")
    _input = input()
    if _input == "exit":
        break
    with Timing("model.generate: "):
        inputs = tok(_input, return_tensors="pt").to("cuda")
        streamer = TextStreamer(tok)
        _ = model.generate(**inputs, streamer=streamer, temperature=0.9)
