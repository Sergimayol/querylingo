import os
from utils import load_json

if __name__ == "__main__":
    base_path = os.path.join(__file__, "..", "..", "tests")
    models = load_json(os.path.join(base_path, "models.json"))
    tests = [load_json(os.path.join(base_path, f"{i}.json")) for i in range(0, 10)]
    print(tests)
