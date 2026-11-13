"""Ingesta de resultados y cuotas de Liga MX desde football-data.co.uk.

El archivo MEX.csv trae todos los partidos desde 2012-13 con cuotas 1X2 de
cierre (Pinnacle, máximo, promedio, Betfair Exchange y Bet365). No trae cuotas
de over/under ni de hándicap.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL_MEX = "https://www.football-data.co.uk/new/MEX.csv"
DIR_CACHE = Path(__file__).resolve().parent.parent / "datos"

# Jornadas de fase regular por torneo (18 equipos, todos contra todos a una vuelta).
JORNADAS_REGULARES = 17

# Alias que la gente escribe -> nombre exacto en football-data.
ALIAS = {
    "america": "Club America",
    "club america": "Club America",
    "aguilas": "Club America",
    "chivas": "Guadalajara Chivas",
    "guadalajara": "Guadalajara Chivas",
    "leon": "Club Leon",
    "tijuana": "Club Tijuana",
    "xolos": "Club Tijuana",
    "san luis": "Atl. San Luis",
    "atletico san luis": "Atl. San Luis",
    "mazatlan": "Mazatlan FC",
    "pumas": "UNAM Pumas",
    "unam": "UNAM Pumas",
    "tigres": "Tigres UANL",
    "rayados": "Monterrey",
    "santos": "Santos Laguna",
    "bravos": "Juarez",
    "fc juarez": "Juarez",
    "gallos": "Queretaro",
    "la maquina": "Cruz Azul",
    "tuzos": "Pachuca",
    "diablos": "Toluca",
    "rayos": "Necaxa",
    "zorros": "Atlas",
    "franja": "Puebla",
    "potros": "Atlante",
}

COLUMNAS_CUOTAS = {
    "PSC": "pinnacle",
    "MaxC": "max",
    "AvgC": "prom",
    "BFEC": "betfair",
    "B365C": "b365",
}


def _sin_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c)).lower().strip()


def descargar(destino: Path | None = None, forzar: bool = False) -> Path:
    """Descarga MEX.csv a la caché local (una vez, salvo que se fuerce)."""
    destino = destino or DIR_CACHE / "MEX.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and not forzar:
        return destino
    resp = requests.get(URL_MEX, timeout=60, headers={"User-Agent": "predicciones-ligamx"})
    resp.raise_for_status()
    destino.write_bytes(resp.content)
    return destino


def leer_csv(rutas: list[Path] | Path) -> pd.DataFrame:
    """Lee uno o varios CSV con el formato de football-data (columnas vacías incluidas)."""
    if isinstance(rutas, (str, Path)):
        rutas = [Path(rutas)]
    partes = [pd.read_csv(r) for r in rutas]
    df = pd.concat(partes, ignore_index=True)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def limpiar(crudo: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas, fechas y etiqueta torneo y fase."""
    df = crudo.dropna(subset=["Home", "Away", "HG", "AG"]).copy()
    df = df.rename(columns={"Season": "temporada", "Home": "local", "Away": "visita",
                            "HG": "gl", "AG": "gv", "Res": "res"})
    df["fecha"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["gl"] = df["gl"].astype(int)
    df["gv"] = df["gv"].astype(int)
    for pref, nombre in COLUMNAS_CUOTAS.items():
        for lado, sufijo in (("H", "l"), ("D", "e"), ("A", "v")):
            col = f"{pref}{lado}"
            df[f"c_{nombre}_{sufijo}"] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    df = df.sort_values(["fecha", "Time", "local"], kind="mergesort").drop_duplicates(
        ["fecha", "local", "visita"]).reset_index(drop=True)
    df["torneo"] = df["fecha"].apply(nombre_torneo)
    df["fase"] = etiquetar_fase(df)
    columnas = ["fecha", "temporada", "torneo", "fase", "local", "visita", "gl", "gv", "res"]
    columnas += [c for c in df.columns if c.startswith("c_")]
    return df[columnas]


def nombre_torneo(fecha: pd.Timestamp) -> str:
    """Apertura = julio a diciembre; Clausura = enero a junio."""
    return f"Apertura {fecha.year}" if fecha.month >= 7 else f"Clausura {fecha.year}"


def etiquetar_fase(df: pd.DataFrame) -> pd.Series:
    """'regular' mientras ambos equipos lleven menos de 17 partidos en el torneo;
    después 'liguilla' (incluye Play-In y Repechaje de torneos anteriores)."""
    jugados: dict[tuple[str, str], int] = {}
    fases = []
    for torneo, local, visita in zip(df["torneo"], df["local"], df["visita"]):
        nl = jugados.get((torneo, local), 0)
        nv = jugados.get((torneo, visita), 0)
        fases.append("regular" if max(nl, nv) < JORNADAS_REGULARES else "liguilla")
        jugados[(torneo, local)] = nl + 1
        jugados[(torneo, visita)] = nv + 1
    return pd.Series(fases, index=df.index)


def cargar(rutas: list[Path] | Path | None = None, forzar_descarga: bool = False) -> pd.DataFrame:
    """Carga los partidos limpios. Sin rutas, descarga MEX.csv de football-data."""
    if rutas is None:
        rutas = descargar(forzar=forzar_descarga)
    return limpiar(leer_csv(rutas))


def resolver_equipo(nombre: str, equipos: list[str]) -> str:
    """Convierte lo que escribe el usuario en el nombre exacto del dataset."""
    clave = _sin_acentos(nombre)
    if clave in ALIAS and ALIAS[clave] in equipos:
        return ALIAS[clave]
    por_clave = {_sin_acentos(e): e for e in equipos}
    if clave in por_clave:
        return por_clave[clave]
    coincidencias = [e for k, e in por_clave.items() if clave in k]
    if len(coincidencias) == 1:
        return coincidencias[0]
    raise ValueError(f"No reconozco el equipo '{nombre}'. Opciones: {', '.join(sorted(equipos))}")


def probabilidades_mercado(df: pd.DataFrame, casa: str = "prom") -> pd.DataFrame:
    """Probabilidades 1X2 implícitas sin margen (normalización proporcional)."""
    inv = pd.DataFrame({s: 1.0 / df[f"c_{casa}_{s}"] for s in ("l", "e", "v")})
    return inv.div(inv.sum(axis=1), axis=0)
