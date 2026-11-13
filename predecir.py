"""Predice un partido de Liga MX: 1X2, over/under y hándicap, con cuotas justas.

Ejemplos:
    python predecir.py Toluca America
    python predecir.py Toluca America --cuota over2.5=1.85 --cuota local=1.70 --cuota hcp_local-1.5=3.10

Claves de --cuota: local, empate, visita, over<L>, under<L>, hcp_local<L>, hcp_visita<L>.
La cuota es decimal, tal como la muestra Draftea.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ligamx import datos, elo, mercados
from ligamx.dixon_coles import DixonColes


def leer_cuotas(pares: list[str]) -> dict[str, float]:
    cuotas = {}
    for par in pares:
        clave, _, valor = par.partition("=")
        if not valor:
            raise SystemExit(f"Cuota mal escrita: '{par}'. Usa clave=cuota, por ejemplo over2.5=1.90")
        cuotas[clave.strip().lower()] = float(valor)
    return cuotas


def probabilidad_de(clave: str, m) -> mercados.Resultado:
    p = mercados.uno_x_dos(m)
    if clave in ("local", "empate", "visita"):
        return mercados.Resultado(p[clave], 0.0, 1 - p[clave])
    if g := re.fullmatch(r"(over|under)([\d.]+)", clave):
        fn = mercados.over if g[1] == "over" else mercados.under
        return fn(m, float(g[2]))
    if g := re.fullmatch(r"hcp_(local|visita)([+-]?[\d.]+)", clave):
        return mercados.handicap(m, float(g[2]), g[1])
    raise SystemExit(f"No reconozco el mercado '{clave}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("local")
    ap.add_argument("visita")
    ap.add_argument("--cuota", action="append", default=[], help="mercado=cuota decimal (repetible)")
    ap.add_argument("--datos", nargs="*", type=Path, help="CSV locales en vez de descargar MEX.csv")
    ap.add_argument("--actualizar", action="store_true", help="vuelve a descargar MEX.csv")
    args = ap.parse_args()

    df = datos.cargar(args.datos, forzar_descarga=args.actualizar)
    equipos = sorted(set(df["local"]) | set(df["visita"]))
    local = datos.resolver_equipo(args.local, equipos)
    visita = datos.resolver_equipo(args.visita, equipos)

    modelo = DixonColes().ajustar(df)
    _, ratings = elo.calcular(df)
    m = modelo.matriz(local, visita)
    lam, mu = modelo.goles_esperados(local, visita)
    r = mercados.resumen(m)

    print("=" * 62)
    print(f"  {local}  vs  {visita}")
    print(f"  Datos hasta {df['fecha'].max():%d/%m/%Y}  ·  Elo {ratings.get(local, 0):.0f} vs {ratings.get(visita, 0):.0f}")
    print(f"  Goles esperados: {lam:.2f} - {mu:.2f}")
    print("=" * 62)
    print(f"  {'Mercado':<26}{'Prob.':>9}{'Cuota justa':>14}")
    for k, nombre in (("local", local), ("empate", "Empate"), ("visita", visita)):
        p = r["1x2"][k]
        print(f"  {nombre:<26}{p:>8.1%}{1 / p:>14.2f}")
    print("-" * 62)
    for l, d in r["totales"].items():
        for lado in ("over", "under"):
            res = d[lado]
            print(f"  {lado.capitalize()} {l:<21}{res.gana:>8.1%}{res.cuota_justa():>14.2f}")
    print(f"  {'Ambos anotan':<26}{r['ambos_anotan']:>8.1%}{1 / r['ambos_anotan']:>14.2f}")
    print("-" * 62)
    for l, d in r["handicap"].items():
        res = d["local"]
        etiqueta = f"{local[:16]} {l:+.1f}"
        nota = f"  (devuelve {res.anula:.0%})" if res.anula > 0.005 else ""
        print(f"  {etiqueta:<26}{res.gana:>8.1%}{res.cuota_justa():>14.2f}{nota}")
    print("=" * 62)

    cuotas = leer_cuotas(args.cuota)
    if cuotas:
        print(f"  {'Cuota Draftea':<20}{'Cuota':>8}{'Justa':>8}{'Valor esp.':>12}{'Kelly 1/4':>11}")
        for clave, cuota in cuotas.items():
            res = probabilidad_de(clave, m)
            ve = res.valor_esperado(cuota)
            marca = "  <- valor" if ve > 0.03 else ""
            print(f"  {clave:<20}{cuota:>8.2f}{res.cuota_justa():>8.2f}{ve:>+12.1%}"
                  f"{mercados.kelly(res, cuota):>11.1%}{marca}")
        print("=" * 62)
    print("  Ojo: en el backtest, el mercado de cierre todavía es más preciso que este modelo.")


if __name__ == "__main__":
    main()
