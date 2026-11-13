"""Genera las apuestas sugeridas de los próximos partidos y las guarda en un CSV
para capturar la cuota de Draftea y el resultado.

    python generar_apuestas.py            # próximos 10 días
    python generar_apuestas.py --dias 3
    python generar_apuestas.py --evaluar  # resultado de lo ya capturado en apuestas/*.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from ligamx import datos, generador, sportmonks
from ligamx.dixon_coles import DixonColes


def apuestas(dias: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(partidos próximos, apuestas sugeridas)."""
    df = datos.cargar(forzar_descarga=True)
    modelo = DixonColes().ajustar(df)
    try:
        jug, _, _ = sportmonks.detalle(forzar=True)
    except requests.RequestException:
        # Sportmonks a veces da timeout con la descarga completa: sigue con la copia local si existe.
        jug, _, _ = sportmonks.detalle()
    partidos = sportmonks.proximos(dias)
    if partidos.empty:
        return partidos, pd.DataFrame()
    candidatas = pd.concat([generador.selecciones_partido(modelo, p.local, p.visita, jug)
                            for p in partidos.itertuples()], ignore_index=True)
    return partidos, generador.generar(candidatas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dias", type=int, default=10)
    ap.add_argument("--evaluar", action="store_true")
    args = ap.parse_args()
    if args.evaluar:
        registro = pd.concat([pd.read_csv(f) for f in sorted(Path("apuestas").glob("*.csv"))])
        print(generador.evaluar(registro).round(3).to_string())
        return
    partidos, sugeridas = apuestas(args.dias)
    if sugeridas.empty:
        raise SystemExit("No hay partidos programados en ese periodo.")
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(partidos.assign(inicio=partidos["inicio"].dt.strftime("%a %d/%m %H:%M")).to_string(index=False))
    for nivel, bloque in sugeridas.groupby("nivel", sort=False):
        print(f"\n=== Riesgo {nivel.lower()} ({bloque['unidades'].iloc[0]} unidades por apuesta) ===")
        for a in bloque.itertuples():
            print(f"  [{a.tipo}] {a.apuesta}\n      prob. {a.prob:.0%} · cuota justa {a.cuota_justa:.2f} · "
                  f"apostar sólo si Draftea paga {a.cuota_minima:.2f} o más")
    ruta = Path("apuestas") / f"{partidos['inicio'].min():%Y-%m-%d}.csv"
    ruta.parent.mkdir(exist_ok=True)
    sugeridas.round(3).to_csv(ruta, index=False)
    print(f"\nGuardado en {ruta}: llena cuota_draftea y, después del partido, gano (1 o 0).")


if __name__ == "__main__":
    main()
