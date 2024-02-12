# pip install -q transformers accelerate
import os, warnings  # noqa: E401
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from utils import Timing
from safetensors import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys(): tensors[key] = f.get_tensor(key)

with open("keys.txt", "w") as f: 
    # Write the keys with the size of the tensor
    f.write("\n".join([f"{key}: {tensors[key].size()}" for key in tensors.keys()]))

exit()
warnings.filterwarnings("ignore")
checkpoint = "bigscience/bloomz-560m"
checkpoint = "bigscience/bloom-3b"
checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=os.path.join("D:", "models")).to(device)
tok = AutoTokenizer.from_pretrained(checkpoint, load_in_8bit=True)

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
