"""Probabilidades de mercados a partir de la matriz de marcadores, y valor esperado.

Convenciones (las mismas que usa Draftea en sus reglas generales):
- Totales y hándicap: si el resultado cae exactamente en la línea, la apuesta
  se anula y se devuelve lo apostado.
- Líneas de cuarto (2.25, -0.75, ...) se tratan como mitad en cada línea vecina,
  como en el hándicap asiático. Si Draftea no las ofrece, simplemente no se usan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Resultado:
    gana: float     # probabilidad de ganar completo
    anula: float    # probabilidad de que se devuelva lo apostado
    pierde: float   # probabilidad de perder completo

    def cuota_justa(self) -> float:
        """Cuota decimal con valor esperado cero."""
        if self.gana <= 0:
            return float("inf")
        return 1.0 + self.pierde / self.gana

    def valor_esperado(self, cuota: float) -> float:
        """Ganancia esperada por cada 1 apostado."""
        return self.gana * (cuota - 1.0) - self.pierde


def _diferencias(m: np.ndarray) -> np.ndarray:
    i, j = np.indices(m.shape)
    return i - j


def _totales(m: np.ndarray) -> np.ndarray:
    i, j = np.indices(m.shape)
    return i + j


def uno_x_dos(m: np.ndarray) -> dict[str, float]:
    d = _diferencias(m)
    return {"local": float(m[d > 0].sum()), "empate": float(m[d == 0].sum()),
            "visita": float(m[d < 0].sum())}


def _linea_simple(valor: np.ndarray, m: np.ndarray, linea: float, mayor: bool) -> Resultado:
    if mayor:
        gana, pierde = m[valor > linea].sum(), m[valor < linea].sum()
    else:
        gana, pierde = m[valor < linea].sum(), m[valor > linea].sum()
    anula = m[valor == linea].sum()
    return Resultado(float(gana), float(anula), float(pierde))


def _combinar_cuarto(fn, linea: float) -> Resultado:
    """Las líneas .25/.75 son media apuesta en cada línea vecina."""
    if abs((linea * 4) % 2 - 1) < 1e-9:   # es un cuarto
        a, b = fn(linea - 0.25), fn(linea + 0.25)
        return Resultado((a.gana + b.gana) / 2, (a.anula + b.anula) / 2, (a.pierde + b.pierde) / 2)
    return fn(linea)


def over(m: np.ndarray, linea: float) -> Resultado:
    t = _totales(m)
    return _combinar_cuarto(lambda l: _linea_simple(t, m, l, mayor=True), linea)


def under(m: np.ndarray, linea: float) -> Resultado:
    t = _totales(m)
    return _combinar_cuarto(lambda l: _linea_simple(t, m, l, mayor=False), linea)


def handicap(m: np.ndarray, linea: float, lado: str = "local") -> Resultado:
    """Hándicap sobre el equipo `lado`. linea=-1.5 local: el local debe ganar por 2+."""
    d = _diferencias(m) if lado == "local" else -_diferencias(m)
    # Gana si diferencia + línea > 0, se anula si = 0.
    return _combinar_cuarto(lambda l: _linea_simple(d, m, -l, mayor=True), linea)


def ambos_anotan(m: np.ndarray) -> float:
    return float(m[1:, 1:].sum())


def kelly(prob: Resultado, cuota: float, fraccion: float = 0.25) -> float:
    """Fracción del bankroll a apostar (Kelly fraccional). 0 si no hay valor."""
    b = cuota - 1.0
    p, q = prob.gana, prob.pierde
    if b <= 0 or p <= 0:
        return 0.0
    f = (b * p - q) / (b * (p + q)) if (p + q) > 0 else 0.0
    return max(0.0, f * fraccion)


def resumen(m: np.ndarray, lineas_totales=(1.5, 2.5, 3.5),
            lineas_handicap=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)) -> dict:
    """Todas las probabilidades y cuotas justas de un partido."""
    salida = {"1x2": uno_x_dos(m), "ambos_anotan": ambos_anotan(m), "totales": {}, "handicap": {}}
    for l in lineas_totales:
        salida["totales"][l] = {"over": over(m, l), "under": under(m, l)}
    for l in lineas_handicap:
        salida["handicap"][l] = {"local": handicap(m, l, "local"),
                                 "visita": handicap(m, -l, "visita")}
    return salida
