import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

from utils import CACHE_DIR

dataset = load_dataset("kaxap/llama2-sql-instruct")
checkpoint = "bigscience/bloomz-560m"
checkpoint = "elyza/ELYZA-japanese-Llama-2-7b"
device = "cuda"

if any(k in checkpoint for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side=padding_side, cache_dir=CACHE_DIR)
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=4096, return_tensors="pt")


if __name__ == "__main__":
    tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=1)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))

    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=CACHE_DIR).to(device)

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode("CREATE TABLE table_name (column_name column_type);", return_tensors="pt").to(device)
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        print(tokenizer.decode(output[0], skip_special_tokens=True))


exit()

dataset = load_dataset("yelp_review_full")


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    # Apply tokenization to both prompt and answer
    prompt = tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")
    answer = tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

    # Extract relevant information from BatchEncoding objects
    prompt_input_ids = prompt["input_ids"]
    prompt_attention_mask = prompt["attention_mask"]
    answer_input_ids = answer["input_ids"]
    answer_attention_mask = answer["attention_mask"]

    # Create the desired dictionary with expected types
    return {
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "answer_input_ids": answer_input_ids,
        "answer_attention_mask": answer_attention_mask,
    }


if __name__ == "__main__":
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=6)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5, cache_dir="D:models")
    from transformers import TrainingArguments

    training_args = TrainingArguments(output_dir="test_trainer")
    import numpy as np
    from datasets import load_metric

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

exit()
lines = open("data.txt").readlines()
corpus = [line.strip() for line in lines if line.strip() != "" and not line.startswith("#")]
corpus.extend(["El gato está en el tejado.", "El perro está en el jardín."])

vocab = set(" ".join(corpus).split())
word_to_index = {word: idx for idx, word in enumerate(vocab)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

data = [torch.tensor([word_to_index[word] for word in sentence.split()], device="cuda") for sentence in corpus]


class SimpleLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output


vocab_size = len(vocab)
embed_size = 50
hidden_size = 100
model = SimpleLM(vocab_size, embed_size, hidden_size).to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
num_epochs = 1
for epoch in range(num_epochs):
    for sequence in tqdm(data[1000:], desc="Training"):
        # print(sequence)
        optimizer.zero_grad()
        output = model(sequence.unsqueeze(0))
        # print(output)
        # tmp = output.view(-1, vocab_size)
        ## Softmax
        # tmp = torch.nn.functional.softmax(tmp, dim=-1)
        # tmp = torch.argmax(tmp, dim=-1)
        # print("HERE", tmp, sequence.view(-1))
        loss = criterion(output.view(-1, vocab_size), sequence.view(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


# Generar texto
def generate_text(seed_text, model, length=10):
    model.eval()
    with torch.no_grad():
        seed = torch.tensor([word_to_index[word] for word in seed_text.split()]).to("cuda")
        for _ in range(length):
            output = model(seed.unsqueeze(0))
            output = torch.nn.functional.softmax(output, dim=-1)
            token = torch.argmax(output)
            seed = torch.cat([seed, token.unsqueeze(0)]).to("cuda")

        seed = seed.cpu()
    return " ".join([index_to_word[idx.item()] for idx in seed])


print(generate_text("El gato", model, length=10))
