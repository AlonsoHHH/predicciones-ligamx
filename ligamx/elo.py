"""Ratings Elo para Liga MX.

Adaptado de `predict_today.py` de mar-antaya/world_cup_predictions (MIT,
Copyright (c) 2026 Mariana Antaya). Cambios para clubes: no hay sedes
neutrales, el rating se acerca a la media entre torneos y los equipos nuevos
entran por debajo de la media.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE = 1500.0
K = 20.0
VENTAJA_LOCAL = 70.0
REGRESION_TORNEO = 0.20   # fracción que se acerca a 1500 al cambiar de torneo
RATING_NUEVO = 1450.0     # equipos sin historial (franquicias nuevas o que regresan)


def calcular(df: pd.DataFrame, k: float = K, ventaja_local: float = VENTAJA_LOCAL,
             regresion: float = REGRESION_TORNEO) -> tuple[pd.DataFrame, dict[str, float]]:
    """Devuelve df con `elo_l`, `elo_v` (previos al partido) y los ratings finales."""
    df = df.sort_values("fecha", kind="mergesort").reset_index(drop=True)
    rating: dict[str, float] = {}
    pre_l = np.empty(len(df))
    pre_v = np.empty(len(df))
    torneo_actual = None
    for i, fila in enumerate(df.itertuples(index=False)):
        if fila.torneo != torneo_actual:
            if torneo_actual is not None:
                rating = {e: r + regresion * (BASE - r) for e, r in rating.items()}
            torneo_actual = fila.torneo
        rl = rating.get(fila.local, RATING_NUEVO)
        rv = rating.get(fila.visita, RATING_NUEVO)
        pre_l[i], pre_v[i] = rl, rv
        esperado = 1.0 / (1.0 + 10 ** (-((rl + ventaja_local) - rv) / 400.0))
        real = 1.0 if fila.gl > fila.gv else (0.5 if fila.gl == fila.gv else 0.0)
        margen = abs(fila.gl - fila.gv)
        mult = np.log(max(margen, 1) + 1) * (2.2 / (abs(rl - rv) * 0.001 + 2.2))
        delta = k * mult * (real - esperado)
        rating[fila.local] = rl + delta
        rating[fila.visita] = rv - delta
    df["elo_l"], df["elo_v"] = pre_l, pre_v
    df["elo_dif"] = pre_l - pre_v
    return df, rating

