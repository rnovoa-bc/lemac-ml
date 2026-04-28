import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Cargar modelo ─────────────────────────────────────────────────
print("Carregant model...")
model     = AutoModelForSequenceClassification.from_pretrained("./model-materies-final")
tokenizer = AutoTokenizer.from_pretrained("./model-materies-final")

with open("./model-materies-final/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

model.eval()
print("Model carregat!\n")

# ── Funció de predicció ───────────────────────────────────────────
def suggerir_materies(titol, threshold=0.07, top_n=5):
    inputs = tokenizer(
        titol,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilitats = torch.sigmoid(outputs.logits[0]).numpy()

    # ─────────────────────────────────────────────────────────────

    indices_ordenats = np.argsort(probabilitats)[::-1]
    suggeriments = []
    for idx in indices_ordenats[:top_n * 3]:
        prob = probabilitats[idx]
        if prob >= threshold:
            suggeriments.append({
                "materia":   mlb.classes_[idx],
                "confiança": prob
            })
        if len(suggeriments) >= top_n:
            break

    return suggeriments

# ── Tests ─────────────────────────────────────────────────────────
titols_test = [
    "Història de l'esport català al segle XX",
    "Gramàtica de la llengua catalana",
    "Cuina mediterrània tradicional",
    "Literatura castellana del Siglo de Oro",
    "Arquitectura gòtica a Catalunya",
]

for titol in titols_test:
    print(f"Títol:  {titol}")
    print(f"{'─'*60}")
    suggeriments = suggerir_materies(titol)

    if suggeriments:
        for s in suggeriments:
            print(f"  {s['confiança']:.2%}  {s['materia']}")
    else:
        print("  Cap suggeriment supera el threshold")

    print()