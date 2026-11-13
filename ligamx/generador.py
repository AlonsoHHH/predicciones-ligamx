"""Generador de apuestas por nivel de riesgo.

No conoce las cuotas de Draftea, así que no puede decir si una apuesta tiene
valor: propone selecciones según la probabilidad del modelo y da la cuota
mínima a la que valdría la pena. Si Draftea paga menos, no se apuesta.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import extras, mercados

MARGEN = 1.05   # la cuota de Draftea debe superar a la justa al menos 5%
# nivel -> probabilidad objetivo de (una sencilla, un parlay completo)
NIVELES = {"Bajo": (0.78, 0.65), "Medio": (0.58, 0.42), "Alto": (0.33, 0.18)}
STAKE = {"Bajo": 2.0, "Medio": 1.0, "Alto": 0.5}   # unidades sugeridas


def selecciones_partido(modelo, local: str, visita: str, jug: pd.DataFrame | None = None) -> pd.DataFrame:
    """Todas las selecciones candidatas de un partido con su probabilidad."""
    m = modelo.matriz(local, visita)
    lam, mu = modelo.goles_esperados(local, visita)
    p = mercados.uno_x_dos(m)
    x = extras.mitades_y_primer_gol(lam, mu)
    btts = mercados.ambos_anotan(m)
    s = {f"Gana {local}": p["local"], "Empate": p["empate"], f"Gana {visita}": p["visita"],
         f"{local} o empate": p["local"] + p["empate"], f"{visita} o empate": p["visita"] + p["empate"],
         "Ambos anotan: sí": btts, "Ambos anotan: no": 1 - btts,
         f"Anota primero {local}": x["anota_primero_local"], f"Anota primero {visita}": x["anota_primero_visita"],
         "Gol en la 1a mitad": x["gol_1a_mitad"]}
    for linea in (1.5, 2.5, 3.5):
        s[f"Más de {linea} goles"] = mercados.over(m, linea).gana
        s[f"Menos de {linea} goles"] = mercados.under(m, linea).gana
    if jug is not None:
        for equipo, tasa in ((local, lam), (visita, mu)):
            for j in extras.jugadores_partido(jug, equipo, tasa).head(2).itertuples():
                s[f"Anota {j.jugador}"] = j.p_gol
    return pd.DataFrame({"partido": f"{local} vs {visita}", "seleccion": list(s), "prob": list(s.values())})


def _fila(nivel: str, tipo: str, legs: pd.DataFrame) -> dict:
    prob = float(np.prod(legs["prob"]))
    return {"nivel": nivel, "tipo": tipo,
            "apuesta": " + ".join(f"{l.seleccion} ({l.partido})" for l in legs.itertuples()),
            "prob": prob, "cuota_justa": 1 / prob, "cuota_minima": MARGEN / prob,
            "unidades": STAKE[nivel], "cuota_draftea": None, "gano": None}


def generar(candidatas: pd.DataFrame, sencillas: int = 4, parlays: int = 3) -> pd.DataFrame:
    """Por nivel: las `sencillas` selecciones más cercanas a la probabilidad objetivo
    (una por partido) y `parlays` combinadas de 2 a 4 selecciones de partidos distintos
    cuya probabilidad conjunta queda más cerca del objetivo. Las piernas de un parlay
    salen de las dos selecciones más seguras de cada partido."""
    filas = []
    # La pierna más segura de cada partido que todavía paga algo (cuota justa de 1.15 o más).
    piernas = (candidatas[candidatas["prob"] <= 0.87].sort_values("prob", ascending=False)
               .groupby("partido").head(2))
    combos = [piernas.loc[list(idx)] for k in (2, 3, 4) for idx in combinations(piernas.index, k)]
    combos = [c for c in combos if c["partido"].is_unique]
    for nivel, (sencilla, parlay) in NIVELES.items():
        cerca = candidatas.assign(d=(candidatas["prob"] - sencilla).abs()).sort_values("d")
        for _, leg in cerca.groupby("partido", sort=False).head(1).head(sencillas).iterrows():
            filas.append(_fila(nivel, "Sencilla", leg.to_frame().T))
        usados: set[str] = set()   # cada parlay del nivel usa partidos distintos a los anteriores
        hechos = 0
        for legs in sorted(combos, key=lambda c: abs(c["prob"].prod() - parlay)):
            if hechos < parlays and usados.isdisjoint(legs["partido"]):
                usados |= set(legs["partido"])
                hechos += 1
                filas.append(_fila(nivel, f"Parlay de {len(legs)}", legs))
    return pd.DataFrame(filas)


def evaluar(registro: pd.DataFrame) -> pd.DataFrame:
    """Resultado por nivel de las apuestas ya capturadas (con cuota_draftea y gano)."""
    r = registro.dropna(subset=["cuota_draftea", "gano"])
    r = r[r["cuota_draftea"] >= r["cuota_minima"]]   # las que sí se habrían apostado
    r = r.assign(ganancia=r["unidades"] * (r["gano"] * r["cuota_draftea"] - 1))
    g = r.groupby("nivel", sort=False)
    return pd.DataFrame({"apuestas": g.size(), "aciertos": g["gano"].mean(), "prob_modelo": g["prob"].mean(),
                         "unidades_apostadas": g["unidades"].sum(), "ganancia": g["ganancia"].sum(),
                         "roi": g["ganancia"].sum() / g["unidades"].sum()})
