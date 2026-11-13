"""Backtest walk-forward del modelo contra el mercado.

    python correr_backtest.py --desde 2023-07-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ligamx import backtest as bt
from ligamx import datos, sportmonks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2023-07-01", help="primera fecha a predecir")
    ap.add_argument("--datos", nargs="*", type=Path, help="CSV locales en vez de descargar MEX.csv")
    ap.add_argument("--xi", type=float, default=None, help="decaimiento temporal por día")
    ap.add_argument("--l2", type=float, default=None, help="regularización de ataque/defensa")
    ap.add_argument("--peso-xg", type=float, default=0.0,
                    help="mezcla goles con xG aproximado por tiros de Sportmonks (0 a 1)")
    ap.add_argument("--umbral", type=float, default=0.05, help="valor esperado mínimo para apostar")
    ap.add_argument("--salida", type=Path, help="guarda las predicciones en CSV")
    args = ap.parse_args()

    pd.set_option("display.width", 120)
    df = datos.cargar(args.datos)
    stats = sportmonks.descargar() if args.peso_xg else None
    pred = bt.predecir_walk_forward(df, args.desde, xi=args.xi, l2=args.l2,
                                    peso_xg=args.peso_xg, stats=stats)
    m = bt.metricas(pred)

    print(f"\nPartidos de fase regular evaluados: {m['partidos']} (desde {args.desde})")
    print("\n1X2 (menor es mejor)")
    print(m["1x2"].round(4).to_string())
    print("\nOver/under 2.5 (menor es mejor)")
    print(m["over25"].round(4).to_string())
    print("\nCalibración over 2.5")
    print(m["calibracion_over25"].round(3).to_string())
    print(f"\nApuestas simuladas 1X2 (1 unidad, valor esperado > {args.umbral:.0%})")
    for casa, nombre in (("prom", "Cuota promedio (aprox. Draftea)"), ("max", "Mejor cuota del mercado")):
        print(f"  {nombre}: {bt.resumen_apuestas(bt.simular_apuestas(pred, casa, args.umbral))}")
    if args.salida:
        pred.to_csv(args.salida, index=False)
        print(f"\nPredicciones guardadas en {args.salida}")


if __name__ == "__main__":
    main()
