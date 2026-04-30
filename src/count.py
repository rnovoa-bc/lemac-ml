import json

with open("data/training_data.json", "r", encoding="utf-8") as f:
  training_data = json.load(f)

total_pairs = len(training_data)
print(f"Total de parells (títol, matèries): {total_pairs}")
subjects = sum(len(pair["materies"]) for pair in training_data)
print(f"Total de matèries associades: {subjects}")
unique_subjects = set(sub for pair in training_data for sub in pair["materies"])
print(f"Total de matèries úniques: {len(unique_subjects)}")