"""Backtest walk-forward: cada semana se reajusta el modelo sólo con partidos
anteriores y se predicen los partidos de esa semana. Así no hay fuga de
información del futuro."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import elo, mercados
from .datos import probabilidades_mercado
from .dixon_coles import DixonColes

CLASES = ["l", "e", "v"]


def predecir_walk_forward(df: pd.DataFrame, desde: str, xi: float | None = None,
                          l2: float | None = None) -> pd.DataFrame:
    df = df.sort_values("fecha", kind="mergesort").reset_index(drop=True)
    df, _ = elo.calcular(df)
    df["semana"] = df["fecha"].dt.to_period("W-MON").dt.start_time
    modelo = DixonColes(**{k: v for k, v in {"xi": xi, "l2": l2}.items() if v is not None})
    filas = []
    for semana, bloque in df[df["fecha"] >= pd.Timestamp(desde)].groupby("semana"):
        historia = df[df["fecha"] < semana]
        modelo.ajustar(historia, fecha_corte=semana)
        tasa_over = ((historia["gl"] + historia["gv"]) > 2.5).tail(600).mean()
        logit = _ajustar_elo_logit(historia)
        for fila in bloque.itertuples(index=False):
            m = modelo.matriz(fila.local, fila.visita)
            p = mercados.uno_x_dos(m)
            lam, mu = modelo.goles_esperados(fila.local, fila.visita)
            pe = logit.predict_proba([[fila.elo_dif]])[0]
            filas.append({
                "fecha": fila.fecha, "torneo": fila.torneo, "fase": fila.fase,
                "local": fila.local, "visita": fila.visita, "gl": fila.gl, "gv": fila.gv,
                "dc_l": p["local"], "dc_e": p["empate"], "dc_v": p["visita"],
                "elo_l": pe[0], "elo_e": pe[1], "elo_v": pe[2],
                "xg_l": lam, "xg_v": mu,
                "dc_over25": mercados.over(m, 2.5).gana,
                "dc_btts": mercados.ambos_anotan(m),
                "base_over25": tasa_over,
                **{c: getattr(fila, c) for c in df.columns if c.startswith("c_")},
            })
    return pd.DataFrame(filas)


def _ajustar_elo_logit(historia: pd.DataFrame) -> LogisticRegression:
    h = historia.tail(1500)
    y = np.select([h["gl"] > h["gv"], h["gl"] == h["gv"]], [0, 1], 2)
    return LogisticRegression().fit(h[["elo_dif"]].to_numpy(), y)


def _log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    probs = np.clip(probs, 1e-9, 1)
    return float(-np.mean(np.log(probs[np.arange(len(y)), y])))


def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    uno = np.zeros_like(probs)
    uno[np.arange(len(y)), y] = 1
    return float(np.mean(((probs - uno) ** 2).sum(axis=1)))


def metricas(pred: pd.DataFrame, solo_regular: bool = True) -> dict:
    p = pred[pred["fase"] == "regular"] if solo_regular else pred
    y = np.select([p["gl"] > p["gv"], p["gl"] == p["gv"]], [0, 1], 2)
    out = {"partidos": len(p)}

    # 1X2: modelos vs mercado (sólo partidos con cuota promedio disponible).
    con_cuota = p["c_prom_l"].notna().to_numpy()
    mk = probabilidades_mercado(p).to_numpy()
    tabla = {}
    for nombre, cols in (("Dixon-Coles", ["dc_l", "dc_e", "dc_v"]),
                         ("Elo", ["elo_l", "elo_e", "elo_v"])):
        pr = p[cols].to_numpy()
        tabla[nombre] = (_log_loss(pr[con_cuota], y[con_cuota]), _brier(pr[con_cuota], y[con_cuota]))
    tabla["Mercado (promedio de cierre)"] = (_log_loss(mk[con_cuota], y[con_cuota]),
                                              _brier(mk[con_cuota], y[con_cuota]))
    frec = np.bincount(y, minlength=3) / len(y)
    tabla["Sin habilidad (frecuencias)"] = (_log_loss(np.tile(frec, (con_cuota.sum(), 1)), y[con_cuota]),
                                             _brier(np.tile(frec, (con_cuota.sum(), 1)), y[con_cuota]))
    out["1x2"] = pd.DataFrame(tabla, index=["log_loss", "brier"]).T

    # Over/under 2.5 (no hay cuotas históricas: se compara contra la tasa base).
    over = ((p["gl"] + p["gv"]) > 2.5).astype(int).to_numpy()
    po = np.column_stack([1 - p["dc_over25"], p["dc_over25"]])
    pb = np.column_stack([1 - p["base_over25"], p["base_over25"]])
    out["over25"] = pd.DataFrame({"Dixon-Coles": [_log_loss(po, over), _brier(po, over)],
                                  "Tasa base": [_log_loss(pb, over), _brier(pb, over)]},
                                 index=["log_loss", "brier"]).T
    out["calibracion_over25"] = calibracion(p["dc_over25"].to_numpy(), over)
    return out


def calibracion(prob: np.ndarray, real: np.ndarray, cortes=(0, .35, .45, .55, .65, 1)) -> pd.DataFrame:
    grupos = pd.cut(prob, cortes, include_lowest=True)
    return (pd.DataFrame({"grupo": grupos, "prob": prob, "real": real})
            .groupby("grupo", observed=True).agg(partidos=("real", "size"),
                                                prob_media=("prob", "mean"),
                                                frecuencia_real=("real", "mean")))


def simular_apuestas(pred: pd.DataFrame, casa: str = "prom", umbral: float = 0.05,
                     solo_regular: bool = True) -> pd.DataFrame:
    """Apuesta 1 unidad a cada resultado 1X2 con valor esperado > umbral."""
    p = pred[pred["fase"] == "regular"] if solo_regular else pred
    filas = []
    for fila in p.itertuples(index=False):
        real = "l" if fila.gl > fila.gv else ("e" if fila.gl == fila.gv else "v")
        for s in CLASES:
            cuota = getattr(fila, f"c_{casa}_{s}")
            prob = getattr(fila, f"dc_{s}")
            if not np.isfinite(cuota):
                continue
            ve = prob * cuota - 1
            if ve > umbral:
                filas.append({"fecha": fila.fecha, "partido": f"{fila.local} - {fila.visita}",
                              "seleccion": s, "cuota": cuota, "prob": prob, "ve": ve,
                              "ganancia": cuota - 1 if s == real else -1.0})
    return pd.DataFrame(filas)


def resumen_apuestas(ap: pd.DataFrame) -> dict:
    if ap.empty:
        return {"apuestas": 0, "ganancia": 0.0, "roi": float("nan")}
    return {"apuestas": len(ap), "ganancia": round(float(ap["ganancia"].sum()), 2),
            "roi": round(float(ap["ganancia"].mean()), 4),
            "aciertos": round(float((ap["ganancia"] > 0).mean()), 3)}
