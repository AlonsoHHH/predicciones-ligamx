"""Variables de contexto propias de la Liga MX: altitud, viaje y descanso."""

from __future__ import annotations

import numpy as np
import pandas as pd

# equipo -> (latitud, longitud, altitud en metros) de su estadio habitual.
# shortcut: valores aproximados de la ciudad, sin cambios de sede temporales
# (p. ej. América en Ciudad de los Deportes); afinar si el efecto resulta grande.
ESTADIOS = {
    "Club America": (19.30, -99.15, 2240), "Cruz Azul": (19.38, -99.18, 2240),
    "UNAM Pumas": (19.33, -99.19, 2240), "Toluca": (19.29, -99.67, 2660),
    "Pachuca": (20.10, -98.76, 2400), "Puebla": (19.08, -98.16, 2135),
    "Lobos BUAP": (19.08, -98.16, 2135), "Queretaro": (20.58, -100.37, 1820),
    "Club Leon": (21.12, -101.68, 1815), "Atl. San Luis": (22.15, -100.98, 1860),
    "Guadalajara Chivas": (20.68, -103.46, 1570), "Atlas": (20.70, -103.33, 1560),
    "Leones Negros": (20.70, -103.33, 1560), "Necaxa": (21.88, -102.28, 1880),
    "Monarcas": (19.70, -101.19, 1920), "Juarez": (31.70, -106.42, 1140),
    "Santos Laguna": (25.55, -103.40, 1120), "Monterrey": (25.67, -100.24, 540),
    "Tigres UANL": (25.72, -100.31, 520), "Chiapas": (16.75, -93.12, 530),
    "Club Tijuana": (32.51, -117.01, 30), "Mazatlan FC": (23.24, -106.42, 10),
    "Veracruz": (19.17, -96.13, 10), "Dorados de Sinaloa": (24.80, -107.39, 60),
    "Atlante": (19.38, -99.18, 2240),
}
ATLANTE_CANCUN = (21.16, -86.85, 10)   # Atlante jugó en Cancún hasta 2020

COVARIABLES = ["altitud", "viaje", "descanso"]


def _estadio(equipo: str, fecha: pd.Timestamp) -> tuple[float, float, float]:
    if equipo == "Atlante" and fecha.year < 2021:
        return ATLANTE_CANCUN
    return ESTADIOS.get(equipo, (np.nan, np.nan, np.nan))


def _km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def agregar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega por partido:
    - altitud: altitud del estadio menos la de casa del visitante, en km.
    - viaje: distancia que viaja el visitante, en miles de km.
    - descanso: días de descanso del local menos los del visitante, en semanas
      (sólo partidos de liga; recortado a 3-14 días)."""
    df = df.copy()
    el = np.array([_estadio(e, f) for e, f in zip(df["local"], df["fecha"])])
    ev = np.array([_estadio(e, f) for e, f in zip(df["visita"], df["fecha"])])
    df["altitud"] = np.nan_to_num((el[:, 2] - ev[:, 2]) / 1000)
    df["viaje"] = np.nan_to_num(_km(el[:, 0], el[:, 1], ev[:, 0], ev[:, 1]) / 1000)

    largo = pd.concat([df[["fecha", "local"]].rename(columns={"local": "e"}).reset_index(),
                       df[["fecha", "visita"]].rename(columns={"visita": "e"}).reset_index()])
    largo = largo.sort_values(["e", "fecha"])
    largo["dias"] = largo.groupby("e")["fecha"].diff().dt.days.clip(3, 14).fillna(14)
    dias = largo.groupby(["index", "e"])["dias"].first()
    dl = dias.loc[list(zip(df.index, df["local"]))].to_numpy()
    dv = dias.loc[list(zip(df.index, df["visita"]))].to_numpy()
    df["descanso"] = (dl - dv) / 7
    return df
