import json
import numpy as np
from datasets import Dataset
from transformers import (
  AutoTokenizer,
  AutoModelForSequenceClassification,
  TrainingArguments,
  Trainer
)
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import torch

# A partir de un model preentrenat: roberta-base fem el fine-tuning
# fent servir les dades que hem extret del catàleg i hem guardat en fromat JSON
# a: data/training_data.json

# ── 1. Carregar les dades ─────────────────────────────────────────
with open("data/training_data.json", encoding="utf-8") as f:
  data = json.load(f)

# ── 2. Construir el vocabulari de matèries ────────────────────────
# Com a model de classificació multi-etiqueta, necessitem convertir les matèries
# a vectors binaris. Per això fem servir MultiLabelBinarizer de sklearn.
mlb = MultiLabelBinarizer()
mlb.fit([d["materies"] for d in data])
num_materies = len(mlb.classes_)
print(f"Matèries úniques: {num_materies}")

# ── 3. Preparar el dataset ────────────────────────────────────────
model_name = "xlm-roberta-base"  # multilingüe: català, castellà, francès...
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preparar(exemples):
  tokens = tokenizer(
    exemples["titol"],
    truncation=True,
    padding="max_length",
    max_length=128
  )
  tokens["labels"] = [
    mlb.transform([m])[0].astype(np.float32).tolist()
    for m in exemples["materies"]
  ]
  return tokens

dataset = Dataset.from_list(data)
dataset = dataset.map(preparar, batched=True)
dataset = dataset.train_test_split(test_size=0.1)  # 90% train, 10% validació

print(f"Train: {len(dataset['train'])} registres")
print(f"Test:  {len(dataset['test'])} registres")

# ── 4. El model ───────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
  model_name,
  num_labels=num_materies,
  problem_type="multi_label_classification"
)

# ── 5. Mètriques ──────────────────────────────────────────────────
def compute_metrics(eval_pred):
  # agafem els n números reals assiciats a cada matèria
  # amb sigmoid els convertim a probabilitats entre 0 i 1
  # i amb el umbral de 0.5 decidim si la matèria està present o no => vector binari
  logits, labels = eval_pred
  predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy()
  f1 = f1_score(labels, predictions, average="macro", zero_division=0)
  f1_micro = f1_score(labels, predictions, average="micro", zero_division=0)
  return {
    "f1_macro": f1,
    "f1_micro": f1_micro
  }

# ── 6. Entrenament ────────────────────────────────────────────────
training_args = TrainingArguments(
  output_dir="./model-materies",
  num_train_epochs=3,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=32,
  eval_strategy="epoch",
  save_strategy="epoch",
  load_best_model_at_end=True,
  learning_rate=2e-5,
  warmup_steps=100,
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=dataset["train"],
  eval_dataset=dataset["test"],
  compute_metrics=compute_metrics,
)

trainer.train()

# ── 7. Guardar ────────────────────────────────────────────────────
trainer.save_model("./model-materies-final")
tokenizer.save_pretrained("./model-materies-final")

# Guardar també el MultiLabelBinarizer per usar-lo després
import pickle
with open("./model-materies-final/mlb.pkl", "wb") as f:
  pickle.dump(mlb, f)

print("Model guardat!")