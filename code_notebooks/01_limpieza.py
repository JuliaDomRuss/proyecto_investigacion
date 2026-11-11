"""
01_limpieza.py
Filtra el dataset "Customer Support on Twitter" (Thought Vector et al., 2017)
para quedarse SOLO con la conversación de @AppleSupport:
  - respuestas oficiales de Apple (author_id == "AppleSupport")
  - tweets de clientes dirigidos a Apple (mencionan @AppleSupport
    o fueron respondidos por Apple)

Estructura sugerida del repo:
  data/raw/twcs.csv            (NO subir a GitHub: agregar a .gitignore)
  data/processed/              (salida)
  src/01_limpieza.py
Ajusta el diccionario COLS si tu versión del CSV nombra distinto las columnas.
"""
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/twcs.csv")
OUT = Path("data/processed/apple_support_filtrado.csv")
BRAND = "AppleSupport"

COLS = {
    "id": "tweet_id",
    "autor": "author_id",
    "entrante": "inbound",              # True = escrito por cliente
    "fecha": "created_at",
    "texto": "text",
    "responde_a": "in_response_to_tweet_id",
}

OUT.parent.mkdir(parents=True, exist_ok=True)

# 1. Carga (ids como texto para no perder precisión)
df = pd.read_csv(RAW, dtype={COLS["id"]: str, COLS["responde_a"]: str})
conteo = {"total_original": len(df)}

# 2. Respuestas de Apple y tweets de clientes dirigidos a Apple
es_apple = df[COLS["autor"]] == BRAND
ids_respondidos = set(df.loc[es_apple, COLS["responde_a"]].dropna())
menciona = df[COLS["texto"]].str.contains(r"@applesupport", case=False, na=False)
es_cliente = df[COLS["entrante"]].astype(str).str.lower().eq("true")

filtro = es_apple | (es_cliente & (menciona | df[COLS["id"]].isin(ids_respondidos)))
df = df[filtro].copy()
df["rol"] = df[COLS["autor"]].eq(BRAND).map({True: "apple", False: "cliente"})
conteo["tras_filtro_apple"] = len(df)

# 3. Depuración
df = df.drop_duplicates(subset=COLS["id"])
conteo["tras_quitar_duplicados_id"] = len(df)
df = df.dropna(subset=[COLS["texto"]])
df = df[df[COLS["texto"]].str.strip().ne("")]
conteo["tras_quitar_vacios"] = len(df)

df["fecha"] = pd.to_datetime(df[COLS["fecha"]], errors="coerce", dayfirst=True)
conteo["fechas_invalidas"] = int(df["fecha"].isna().sum())
df = df.dropna(subset=["fecha"]).sort_values("fecha")


# 4. Texto limpio (se conserva el original en la columna "text")
def limpiar(t: str) -> str:
    t = t.replace("’", "'")
    t = re.sub(r"https?://\S+", "<URL>", t)
    t = re.sub(
        r"@\w+",
        lambda m: m.group(0) if m.group(0).lower() == "@applesupport" else "@USER",
        t,
    )
    return re.sub(r"\s+", " ", t).strip()


df["texto_limpio"] = df[COLS["texto"]].apply(limpiar)
df["n_palabras"] = df["texto_limpio"].str.split().str.len()

# 5. Salida y resumen (úsalo en la sección de Resultados)
df.to_csv(OUT, index=False)
conteo["tweets_clientes"] = int((df["rol"] == "cliente").sum())
conteo["respuestas_apple"] = int((df["rol"] == "apple").sum())
conteo["rango_fechas"] = f"{df['fecha'].min()} -> {df['fecha'].max()}"
for k, v in conteo.items():
    print(f"{k}: {v}")
print(df["n_palabras"].describe())
