"""Mercados secundarios: tiros, córners, tarjetas por árbitro, goles por mitad,
quién anota primero y jugadores (tiros, gol y asistencia).

Son promedios recientes ajustados por rival, no modelos entrenados. Todas las
funciones reciben el historial y sólo usan lo que se les pasa: para un backtest,
filtra antes por fecha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICAS = ["tiros", "tiros_gol", "corners", "amarillas"]
PRIMERA_MITAD = 0.45   # fracción de los goles de la Liga MX que caen en la primera mitad
PESO_ARBITRO = 5       # partidos "ficticios" con el promedio de la liga para árbitros con pocos datos
ENCOGER_EQUIPO = 0.5   # cuánto de la diferencia de un equipo contra la liga se conserva
MIN_PREVIOS = 900       # minutos "ficticios" de jugador promedio que se suman a cada jugador


def esperados_equipo(stats: pd.DataFrame, local: str, visita: str, n: int = 10) -> pd.DataFrame:
    """Valor esperado de cada métrica para local y visitante: promedio de la liga
    por condición, ajustado por lo que genera cada equipo y lo que concede su rival
    en sus últimos `n` partidos. `stats` necesita columnas <métrica>_l y <métrica>_v."""
    stats = stats.sort_values("fecha")
    del_local = stats[(stats["local"] == local) | (stats["visita"] == local)].tail(n)
    del_visita = stats[(stats["local"] == visita) | (stats["visita"] == visita)].tail(n)

    def razon(partidos, equipo, metrica, a_favor):
        """Lo que hace (o concede) el equipo entre lo que haría un equipo promedio en su lugar."""
        es_local = partidos["local"] == equipo
        lado = np.where(es_local == a_favor, "l", "v")
        real = np.where(lado == "l", partidos[f"{metrica}_l"], partidos[f"{metrica}_v"]).sum()
        media = np.where(lado == "l", stats[f"{metrica}_l"].mean(), stats[f"{metrica}_v"].mean()).sum()
        return 1 + ENCOGER_EQUIPO * (real / media - 1) if media else 1.0

    filas = {}
    for m in METRICAS:
        if f"{m}_l" not in stats:
            continue
        el = stats[f"{m}_l"].mean() * razon(del_local, local, m, True) * razon(del_visita, visita, m, False)
        ev = stats[f"{m}_v"].mean() * razon(del_visita, visita, m, True) * razon(del_local, local, m, False)
        filas[m] = {"local": el, "visita": ev, "total": el + ev}
    return pd.DataFrame(filas).T


def tarjetas_arbitro(arbitros: pd.DataFrame, arbitro: str) -> tuple[float, int, float]:
    """(amarillas por partido del árbitro, partidos dirigidos, promedio de la liga).
    El promedio del árbitro se encoge hacia el de la liga cuando tiene pocos partidos."""
    total = arbitros["amarillas_l"] + arbitros["amarillas_v"]
    liga = total.mean()
    suyos = total[arbitros["arbitro"] == arbitro]
    return (suyos.sum() + PESO_ARBITRO * liga) / (len(suyos) + PESO_ARBITRO), len(suyos), liga


def mitades_y_primer_gol(lam: float, mu: float, f1: float = PRIMERA_MITAD) -> dict[str, float]:
    """Con goles Poisson de tasas `lam` (local) y `mu` (visita) repartidos f1 / 1-f1 por mitad."""
    p = lambda tasa: 1 - np.exp(-tasa)
    t = lam + mu
    return {"gol_1a_mitad": p(t * f1), "gol_2a_mitad": p(t * (1 - f1)),
            "local_anota_1a": p(lam * f1), "local_anota_2a": p(lam * (1 - f1)),
            "visita_anota_1a": p(mu * f1), "visita_anota_2a": p(mu * (1 - f1)),
            "mas_goles_2a_mitad": _mas_en_segunda(t, f1),
            "anota_primero_local": lam / t * p(t), "anota_primero_visita": mu / t * p(t),
            "sin_goles": float(np.exp(-t))}


def _mas_en_segunda(t: float, f1: float, tope: int = 12) -> float:
    from scipy.stats import poisson
    g = np.arange(tope)
    m = np.outer(poisson.pmf(g, t * f1), poisson.pmf(g, t * (1 - f1)))
    return float(np.triu(m, 1).sum())


def jugadores_partido(jug: pd.DataFrame, equipo: str, goles_esperados: float,
                      n: int = 15, recientes: int = 5) -> pd.DataFrame:
    """Para cada jugador que jugó en los últimos `recientes` partidos del equipo:
    minutos esperados, tiros y tiros a gol esperados, y probabilidad de anotar y de
    asistir. Las probabilidades reparten los `goles_esperados` del modelo según la
    parte que cada jugador aporta al equipo (goles y tiros a gol; asistencias y pases clave)."""
    jug = jug.assign(equipo=lambda d: d["local"].where(d["lado"] == "l", d["visita"]))
    conv_gol = jug["goles"].sum() / jug["tiros_gol"].sum()
    conv_asist = jug["asistencias"].sum() / jug["pases_clave"].sum()
    asist_por_gol = jug["asistencias"].sum() / jug["goles"].sum()

    eq = jug[jug["equipo"] == equipo]
    fechas = np.sort(eq["fecha"].unique())
    eq = eq[eq["fecha"].isin(fechas[-n:])]
    activos = eq[eq["fecha"].isin(fechas[-recientes:])]
    minutos = activos.groupby("player_id")["minutos"].sum() / min(recientes, len(fechas))

    s = eq.groupby("player_id").agg(jugador=("jugador", "last"), partidos=("fecha", "nunique"),
                                    min_total=("minutos", "sum"), tiros=("tiros", "sum"),
                                    tiros_gol=("tiros_gol", "sum"), goles=("goles", "sum"),
                                    asistencias=("asistencias", "sum"), pases_clave=("pases_clave", "sum"))
    s = s.join(minutos.rename("min_esp"), how="inner")
    por_min = s["min_esp"] / s["min_total"].clip(lower=90)   # shortcut: sin encoger a jugadores con pocos minutos
    s["tiros_esp"] = s["tiros"] * por_min
    s["tiros_gol_esp"] = s["tiros_gol"] * por_min
    # Encoge a cada jugador hacia el jugador promedio de la liga: sin esto, las rachas
    # cortas inflan las probabilidades más altas.
    liga = jug[["goles", "tiros_gol", "asistencias", "pases_clave"]].sum() / jug["minutos"].sum() * MIN_PREVIOS
    por_min = s["min_esp"] / (s["min_total"] + MIN_PREVIOS)
    aporte_gol = (s["goles"] + liga["goles"] + conv_gol * (s["tiros_gol"] + liga["tiros_gol"])) / 2 * por_min
    aporte_asist = (s["asistencias"] + liga["asistencias"]
                    + conv_asist * (s["pases_clave"] + liga["pases_clave"])) / 2 * por_min
    s["p_gol"] = 1 - np.exp(-goles_esperados * aporte_gol / aporte_gol.sum())
    s["p_asistencia"] = 1 - np.exp(-goles_esperados * asist_por_gol * aporte_asist / aporte_asist.sum())
    cols = ["jugador", "partidos", "min_esp", "tiros_esp", "tiros_gol_esp", "p_gol", "p_asistencia"]
    return s[cols].sort_values("p_gol", ascending=False).reset_index(drop=True)
