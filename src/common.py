import unicodedata
import re

VALID_TAGS = ["245", "650", "651"] # Lista de etiquetes que necessitem per extreure les dades d'interès

def normalize_label(text: str) -> str:
  """
  Normalitza el text: treu espais innecessaris, signes de puntuació i diacrítics

  Arguments:
  text -- cadena de caràcters que volem normalitzar

  Retorna:
  Un cadena de caràcters
  """
  # Minúscules
  text = text.lower()

  # Treu els diacrítics
  text = unicodedata.normalize("NFD", text)
  text = "".join(
      c for c in text
      if unicodedata.category(c) != "Mn"
  )

  # Treu signes de puntiació
  text = re.sub(r"[^\w\s]", " ", text)

  # Mira multiples espais
  text = re.sub(r"\s+", " ", text)

  return text.strip()

def escape(text: str) -> str:
  """
  Escapa cometes i altres caràcters problemàtics per a RDF

  Arguments:
  text -- cadena de caràcters que volem escapar

  Retorna:
  Un cadena de caràcters escapada
  """
  return text.replace('"', '\\"').strip()

def concat_subfields(field, allowed_codes):
  """
  Concatena els subcamps d'un camp MARC segons els codis indicats

  Arguments:
  field -- el camp MARC a processar
  allowed_codes -- llista de codis de subcamp a concatenar

  Retorna:
  Una cadena amb els valors concatenats dels subcamps indicats
  """
  subfields = field["subfields"]
  values = []
  for code in allowed_codes:
    if code in subfields:
      values.extend(subfields[code])
  return " ".join(values)

def personal_name(field):
    return concat_subfields(field, list("abcdefghijklmnopqrstuvwxyz"))

def corporate_name(field):
    return concat_subfields(field, list("abcdefghijklmnoprst"))

def conference_name(field):
    return concat_subfields(field, list("acdefghjklnpqst"))

def title_name(field):
    return concat_subfields(field, list("adfghklmnoprst"))

def geographic_name(field):
    return concat_subfields(field, ['a', 'g'])