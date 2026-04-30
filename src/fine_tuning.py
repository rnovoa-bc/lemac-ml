import json
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss

# ── 1. Cargar datos ───────────────────────────────────────────────
with open("data/training_data.json", encoding="utf-8") as f:
    data = json.load(f)

# ── 2. Limpieza ───────────────────────────────────────────────────
for d in data:
    if isinstance(d["materies"], str):
        d["materies"] = [d["materies"]]
    d["materies"] = list(set(d["materies"]))

# ── 3. MultiLabelBinarizer ────────────────────────────────────────
mlb = MultiLabelBinarizer()
mlb.fit([d["materies"] for d in data])
num_materies = len(mlb.classes_)
print(f"Matèries úniques: {num_materies}")

# ── 4. Labels FIJOS (CLAVE) ───────────────────────────────────────
labels_matrix = mlb.transform([d["materies"] for d in data]).astype(np.float32)

# sanity check (esto debe ser constante)
print("Label shape:", labels_matrix.shape)

# ── 5. Dataset HF ─────────────────────────────────────────────────
dataset = Dataset.from_list(data)

# tokenización SOLO texto
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(examples):
    return tokenizer(
        examples["titol"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)

# 🔥 añadir labels DESPUÉS (esto evita el bug)
dataset = dataset.add_column("labels", labels_matrix.tolist())

# split
dataset = dataset.train_test_split(test_size=0.1)

# formato torch
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print(f"Train: {len(dataset['train'])}")
print(f"Test: {len(dataset['test'])}")

# ── 6. Pesos desbalance ───────────────────────────────────────────
pos_counts = labels_matrix.sum(axis=0)
neg_counts = len(labels_matrix) - pos_counts

pos_weight = torch.tensor(
    neg_counts / (pos_counts + 1e-5),
    dtype=torch.float32
)

# ── 7. Modelo ─────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_materies,
    problem_type="multi_label_classification"
)

# freeze base
for param in model.base_model.parameters():
    param.requires_grad = False

# unfreeze última capa
for param in model.roberta.encoder.layer[-1].parameters():
    param.requires_grad = True

# classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# ── 8. Trainer personalizado ──────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ── 9. Métricas ───────────────────────────────────────────────────
avg_labels = np.mean(labels_matrix.sum(axis=1))
top_k = max(1, int(avg_labels))

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = np.zeros_like(probs)

    for i, row in enumerate(probs):
        top_indices = row.argsort()[-top_k:]
        predictions[i, top_indices] = 1

    return {
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
    }

# ── 10. Training args ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./model-materies",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    warmup_steps=100,
    fp16=False,
    logging_dir="./logs"
)

# ── 11. Trainer ───────────────────────────────────────────────────

def debug_labels(ds, name):
    lengths = [len(x["labels"]) for x in ds]
    unique_lengths = set(lengths)
    print(f"{name} unique label lengths:", unique_lengths)

    if len(unique_lengths) > 1:
        print("❌ PROBLEM FOUND")
        for i, x in enumerate(ds):
            if len(x["labels"]) != list(unique_lengths)[0]:
                print("BAD SAMPLE:", i, len(x["labels"]))
                print(x["labels"])
                break

debug_labels(dataset["train"], "train")
debug_labels(dataset["test"], "test")


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

# ── 12. Guardar ───────────────────────────────────────────────────
trainer.save_model("./model-materies-final")
tokenizer.save_pretrained("./model-materies-final")

import pickle
with open("./model-materies-final/mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)

print("Model guardat!")