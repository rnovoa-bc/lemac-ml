import json
import urllib.request
import urllib.error
import time

materias = [
    "Arquitectura",
    "Arquitectura catalana",
    "Arquitectura--Modernisme (Art)",
    "Esports",
    "Esports--Accidents",
    "Futbol--Campoionats",
    "Literatura catalana",
    "Literatura catalana--S.XX",
    "Novel·la catalana",
    "Poesia catalana",
    "Ciència",
    "Ciència--Història",
    "Espanya--Història--1936-1939, Guerra Civil",
    "Llengua catalana",
    "Llenguatge i llengües--Gramàtiques",
    "Llenguatge i llengües",
    "Música",
    "Música--Catalunya--S.XX",
    "Biologia",
    "Medicina--Catalunya"
]

all_records = []
batch_size = 100  # titles per API call
total_target = 2000
calls_needed = total_target // batch_size

def call_api(prompt):
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 8000,
        "messages": [{"role": "user", "content": prompt}]
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    })
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["content"][0]["text"]

prompt_template = """Genera exactament {n} registres en format JSON per un conjunt de dades d'entrenament d'una biblioteca catalana.

Les matèries disponibles són:
{materias}

Regles:
- Els títols han de ser en català, castellà, anglès o llatí (barreja realista)
- Cada registre pot tenir 1, 2 o 3 matèries de la llista
- Els títols han de ser variats i realistes (com els d'una biblioteca real)
- Inclou subtítols de vegades, números de volum, anys, etc.
- Les combinacions de matèries han de tenir sentit (no barregis cuina amb medicina)

Retorna NOMÉS el JSON, sense cap text addicional, sense ```json, sense res més. Format:
[{{"titol": "...", "materies": ["..."]}}]

Genera exactament {n} registres."""

print("Generating titles...")
for i in range(calls_needed):
    print(f"Batch {i+1}/{calls_needed}...")
    prompt = prompt_template.format(
        n=batch_size,
        materias="\n".join(f"- {m}" for m in materias)
    )
    try:
        response = call_api(prompt)
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])
        batch = json.loads(response)
        all_records.extend(batch)
        print(f"  Got {len(batch)} records. Total: {len(all_records)}")
        time.sleep(1)
    except Exception as e:
        print(f"  Error in batch {i+1}: {e}")
        print(f"  Response preview: {response[:200] if 'response' in dir() else 'N/A'}")

print(f"\nTotal records generated: {len(all_records)}")

# Save
with open("data/training_data_test.json", "w", encoding="utf-8") as f:
    json.dump(all_records, f, ensure_ascii=False, indent=2)

print("Saved to training_data_test.json")