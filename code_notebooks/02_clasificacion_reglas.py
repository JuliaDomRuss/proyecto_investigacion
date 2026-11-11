"""
02_clasificacion_reglas.py
Etiquetado por diccionarios de palabras (reglas) de los tweets de CLIENTES:
  1) intencion : falla | solicitud
  2) categoria : tema técnico del reporte
  3) urgencia  : baja | media | critica
Genera:
  - data/processed/clientes_etiquetados.csv
  - data/processed/muestra_revision_manual.csv  (para validar a mano)
Los diccionarios son un PUNTO DE PARTIDA: ajústalos viendo ejemplos reales.
"""
import re
from pathlib import Path

import pandas as pd

IN = Path("data/processed/apple_support_filtrado.csv")
OUT = Path("data/processed/clientes_etiquetados.csv")
MUESTRA = Path("data/processed/muestra_revision_manual.csv")


def rx(palabras):
    return re.compile(r"\b(?:" + "|".join(palabras) + r")", re.IGNORECASE)


# ---------- 1. INTENCIÓN ----------
FALLA = rx([
    r"not working", r"doesn'?t work", r"don'?t work", r"stopped working",
    r"won'?t", r"can'?t", r"cannot", r"unable", r"broken", r"crash", r"bug",
    r"error", r"stuck", r"freez", r"frozen", r"glitch", r"issue", r"problem",
    r"fail", r"down\b", r"outage", r"slow", r"drain", r"lost", r"missing",
    r"disappear", r"black screen", r"reboot", r"restart", r"keeps",
])
SOLICITUD = rx([
    r"how (?:do|can|to|does)", r"where (?:do|can|is)", r"what (?:is|does|are)",
    r"when (?:will|is|does)", r"is there (?:a|any)", r"does (?:anyone|apple)",
    r"can i", r"could you", r"any (?:way|idea|tips?)", r"please (?:help|advise)",
    r"how much", r"which",
])

# ---------- 2. CATEGORÍA ----------
CATEGORIAS = {
    "hardware_dispositivo": rx([r"battery", r"screen", r"charg", r"iphone", r"ipad",
                                r"macbook", r"mac\b", r"airpods", r"watch", r"camera",
                                r"speaker", r"button", r"cracked"]),
    "ios_software_actualizacion": rx([r"ios", r"update", r"upgrade", r"macos", r"beta",
                                      r"install", r"os\b", r"firmware", r"keyboard"]),
    "cuenta_icloud": rx([r"icloud", r"apple id", r"password", r"log ?in", r"sign ?in",
                         r"locked", r"verification", r"2fa", r"backup", r"storage"]),
    "apps_appstore": rx([r"app store", r"\bapps?\b", r"download", r"testflight",
                         r"safari", r"mail app"]),
    "pagos_facturacion": rx([r"charge", r"billing", r"refund", r"payment", r"subscription",
                             r"invoice", r"apple pay", r"credit card", r"purchase"]),
    "servicios_apple": rx([r"apple music", r"imessage", r"facetime", r"siri", r"icloud mail",
                           r"apple tv", r"podcast", r"maps", r"itunes"]),
    "conectividad": rx([r"wi-?fi", r"bluetooth", r"cellular", r"signal", r"network",
                        r"data\b", r"sim\b", r"carrier", r"roaming"]),
}

# ---------- 3. URGENCIA ----------
# Crítica: pérdida total de servicio, acceso, datos, dinero o seguridad
CRITICA = rx([
    r"outage", r"down for", r"\bis down\b", r"services? (?:is |are )?down",
    r"completely (?:dead|broken|unusable)", r"bricked", r"won'?t (?:turn on|boot|start)",
    r"boot ?loop", r"stuck on (?:apple )?logo", r"data loss", r"lost (?:all|everything)",
    r"all my (?:photos|data|contacts)", r"hacked", r"unauthori[sz]ed", r"stolen",
    r"fraud", r"locked out", r"can'?t (?:log ?in|sign ?in|access)",
    r"charged (?:twice|\d+)", r"disabled", r"can'?t (?:make|receive) calls?", r"911|emergency",
])
# Media: función degradada o parcial, con o sin solución alterna
MEDIA = rx([
    r"not working", r"doesn'?t work", r"stopped working", r"crash", r"freez",
    r"frozen", r"error", r"bug", r"glitch", r"stuck", r"slow", r"drain",
    r"won'?t (?:update|install|download|connect|sync)", r"keeps", r"missing",
    r"fail", r"can'?t",
])
# Amplificadores: sube un nivel (media -> crítica) si hay duración/alcance/apremio
AMPLIFICADORES = rx([
    r"urgent", r"asap", r"immediately", r"\bstill\b", r"for (?:days|hours|weeks)",
    r"since (?:yesterday|monday|this morning|last)", r"everyone", r"anyone else",
    r"all (?:users|of us)", r"again and again", r"work(?:ing)? from", r"business",
])


def intencion(t: str) -> str:
    if FALLA.search(t):
        return "falla"
    if SOLICITUD.search(t) or "?" in t:
        return "solicitud"
    return "otro"


def categoria(t: str) -> str:
    puntos = {c: len(r.findall(t)) for c, r in CATEGORIAS.items()}
    mejor = max(puntos, key=puntos.get)
    return mejor if puntos[mejor] > 0 else "otros"


def urgencia(t: str, intent: str) -> str:
    if intent != "falla":
        return "baja"
    if CRITICA.search(t):
        return "critica"
    if MEDIA.search(t):
        return "critica" if AMPLIFICADORES.search(t) else "media"
    return "baja"


df = pd.read_csv(IN)
cl = df[df["rol"] == "cliente"].copy()
t = cl["texto_limpio"].fillna("")
cl["intencion"] = t.apply(intencion)
cl["categoria"] = t.apply(categoria)
cl["urgencia"] = [urgencia(x, i) for x, i in zip(t, cl["intencion"])]
cl.to_csv(OUT, index=False)

# Distribuciones (tablas para Resultados)
print(cl["intencion"].value_counts(normalize=True).round(3))
print(cl[cl["intencion"] == "falla"]["urgencia"].value_counts(normalize=True).round(3))
print(pd.crosstab(cl["categoria"], cl["urgencia"]))

# Muestra estratificada para etiquetado manual (gold set)
muestra = (
    cl.groupby(["intencion", "urgencia"], group_keys=False)
    .apply(lambda g: g.sample(min(len(g), 40), random_state=42))
)
for c in ["intencion_manual", "categoria_manual", "urgencia_manual", "comentario_error"]:
    muestra[c] = ""
muestra[["tweet_id", "texto_limpio", "intencion", "categoria", "urgencia",
         "intencion_manual", "categoria_manual", "urgencia_manual",
         "comentario_error"]].to_csv(MUESTRA, index=False)
print(f"Muestra para revisión manual: {len(muestra)} tweets")
