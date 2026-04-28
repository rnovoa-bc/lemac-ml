import json
from common import VALID_TAGS

def read_iso_records(filepath):
  """
  Llegeix un fitxer ISO2709 i retorna un registre cada vegada.

  Arguments:
  filepath -- el camí al fitxer ISO2709 a llegir

  Retorna:
  Un registre en brut (ISO2709)
  """
  with open(filepath, "rb") as f:
    buffer = b""

    # Llegim en troços de 8KB per eficiència
    while chunk := f.read(8192):
        buffer += chunk
        # 0x1d és el delimitador de registre en ISO2709
        parts = buffer.split(b"\x1d")

        # Retornem tots els registres complets
        for record in parts[:-1]:
            yield record

        # Mantenim el regitre non complet al buffer per a la següent iteració
        buffer = parts[-1]

    if buffer:
        yield buffer

def parse_iso_record(record):
  """
  Analitza un registre ISO2709 i retorna un diccionari amb els camps.

  Arguments:
  record -- el registre en brut (ISO2709)

  Retorna:
  Un diccionari amb els camps del registre
  """

  
  # El primer bloc de 24 bytes és la capçalara
  leader = record[:24]
  
  # El següent bloc és el directori, que acaba amb un delimitador de camp (0x1e)
  directory_end = record.find(b"\x1e", 24)
  directory = record[24:directory_end]

  # El contingut dels camps comença després del directori
  fields_data = record[directory_end + 1:]

  fields = {}
  
  # El directori està format per entrades de 12 bytes: tiqueta (3 bytes), long. (4 bytes), inici (5 bytes)
  for i in range(0, len(directory), 12):
    entry = directory[i:i+12]
    tag = entry[:3].decode('utf-8')
    if tag not in VALID_TAGS:
      continue # Ens saltem les etiquetes que no ens interesen
    length = int(entry[3:7].decode('utf-8'))
    start = int(entry[7:].decode('utf-8'))

    field_data = fields_data[start:start+length-1]

    if tag.startswith('00'):
      # Aquests són els camps de control, que no tenen indicadors ni subcamps
      value = field_data.decode('utf-8')
      fields.setdefault(tag, []).append(value)

    else:
      # Aquests són els camps de dades, que tenen indicadors i subcamps
      indicators = field_data[:2].decode('utf-8')
      subfield_data = field_data[2:]

      subfields = {}

      for subfield in subfield_data.split(b"\x1f")[1:]: # Cada subcamps comença amb un delimitador (0x1f) d'aquí el [1:]
        code = subfield[:1].decode('utf-8')
        value = subfield[1:].decode('utf-8')
        subfields.setdefault(code, []).append(value)

      field_object = {
        "indicators": indicators,
        "subfields": subfields
      }

      fields.setdefault(tag, []).append(field_object)
  return {
    "leader": leader.decode('utf-8'),
    "fields": fields
  }

def print_marc_record(record):
  """
  Imprimeix un registre MARC de manera llegible.

  Arguments:
  record -- el registre MARC a imprimir
  """
  print("-"*80)
  print(f"Capçalera: {record['leader']}")
  for tag, field_list in record["fields"].items():
    for field in field_list:
      if isinstance(field, dict):
        print(f"Etiqueta: {tag}; Indicadors: {field['indicators'].replace(" ", "□")}")

        for code, values in field["subfields"].items():
          for value in values:
            print(f"  ${code} {value}")
      else:
        print(f"Control {tag}: {field}")

def extract_training_pairs(filepath, output_file):
  """
  Extreu parells (títol, llista de matèries) per entrenar el model.
  Retorna una llista de diccionaris.
  """
  with open(output_file, "w", encoding="utf-8") as f:
    #f.write("[\n")  # Inici de la llista JSON

    pairs = []

    for raw in read_iso_records(filepath):
      record = parse_iso_record(raw)
      #print_marc_record(record)  # Per depurar, podem imprimir el registre analitzat
      fields = record["fields"]

      # Títol — 245 $a i opcionalment $b (subtítol)
      titol = ""
      if "245" in fields:
        subfields = fields["245"][0]["subfields"]
        parts = []
        if "a" in subfields:
            parts.append(subfields["a"][0].strip(" /:"))
        if "b" in subfields:
            parts.append(subfields["b"][0].strip(" /:"))
        titol = " ".join(parts).strip()

      if not titol:
        continue  # registre sense títol, el saltem

      # Matèries — 650 $a + $x + $y + $z + $v
      materies = []
      for field in fields.get("650", []) + fields.get("651", []):
        subfields = field["subfields"]
        # Només agafem les matèries del tesaure LEMAC
        if "2" not in subfields or subfields["2"][0].strip() != "lemac":
            continue
        parts = []
        for code in ("a", "x",): #, "x", "y", "z", "v"):
          if code in subfields:
              parts.append(subfields[code][0].strip(" ."))
        if parts:
          materia = "--".join(parts)
          materies.append(materia)

      if materies:
        dedup_materies = list(set(materies))  # Eliminar duplicats
        pairs.append({
          "titol": titol,
          "materies": dedup_materies
        })

    json.dump(pairs, f, ensure_ascii=False, indent=2)  # Escrivim la llista de parells al fitxer JSON
    #f.write("]\n")  # Fi de la llista JSON



