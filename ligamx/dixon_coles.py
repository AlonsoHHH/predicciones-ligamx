"""Modelo Dixon-Coles (Poisson bivariado corregido) con decaimiento temporal.

Cada equipo tiene un parámetro de ataque y uno de defensa; hay una ventaja de
local y un parámetro rho que corrige los marcadores bajos (0-0, 1-0, 0-1, 1-1).
Los partidos viejos pesan menos: peso = exp(-xi * días). La matriz de
marcadores que produce sirve para todos los mercados (1X2, over/under y
hándicap).

La corrección tau y la matriz de marcadores siguen la de
mar-antaya/world_cup_predictions (MIT); aquí los parámetros se ajustan por
máxima verosimilitud en lugar de fijarse a mano.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

XI = 0.0035          # por día: vida media de unos 200 días (~1 torneo)
L2 = 2.0             # encoge ataque/defensa hacia 0 (ayuda a equipos con pocos datos)
MAX_GOLES = 11


@dataclass
class DixonColes:
    xi: float = XI
    l2: float = L2
    equipos: list[str] = field(default_factory=list)
    ataque: dict[str, float] = field(default_factory=dict)
    defensa: dict[str, float] = field(default_factory=dict)
    local: float = 0.3
    base: float = 0.1
    rho: float = -0.05
    peso_xg: float = 0.0   # mezcla goles reales con xG aproximado (columnas xg_l, xg_v)
    fecha_corte: pd.Timestamp | None = None
    _x0: np.ndarray | None = None

    # ── ajuste ────────────────────────────────────────────────────────────
    def ajustar(self, df: pd.DataFrame, fecha_corte: pd.Timestamp | None = None) -> "DixonColes":
        """Ajusta con los partidos anteriores a `fecha_corte` (por defecto, todos)."""
        if fecha_corte is not None:
            df = df[df["fecha"] < fecha_corte]
        else:
            fecha_corte = df["fecha"].max() + pd.Timedelta(days=1)
        self.fecha_corte = pd.Timestamp(fecha_corte)
        dias = (self.fecha_corte - df["fecha"]).dt.days.to_numpy(float)
        w = np.exp(-self.xi * dias)
        mantener = w > 1e-3
        df, w = df[mantener], w[mantener]

        equipos = sorted(set(df["local"]) | set(df["visita"]))
        idx = {e: i for i, e in enumerate(equipos)}
        n = len(equipos)
        il = df["local"].map(idx).to_numpy()
        iv = df["visita"].map(idx).to_numpy()
        gl = df["gl"].to_numpy(float)
        gv = df["gv"].to_numpy(float)
        x, y = gl, gv
        if self.peso_xg and "xg_l" in df:
            # Poisson acepta conteos no enteros; la corrección tau sigue usando el marcador real.
            x = np.where(df["xg_l"].notna(), (1 - self.peso_xg) * gl + self.peso_xg * df["xg_l"], gl)
            y = np.where(df["xg_v"].notna(), (1 - self.peso_xg) * gv + self.peso_xg * df["xg_v"], gv)

        x0 = np.zeros(2 * n + 3)
        x0[2 * n:] = [0.1, 0.3, -0.05]
        if self._x0 is not None and self.equipos == equipos:
            x0 = self._x0  # arranque en caliente para el walk-forward

        def nll(p):
            att, dfn = p[:n], p[n:2 * n]
            base, loc, rho = p[2 * n:]
            log_lam = base + loc + att[il] + dfn[iv]
            log_mu = base + att[iv] + dfn[il]
            lam, mu = np.exp(log_lam), np.exp(log_mu)

            ll = x * log_lam - lam - gammaln(x + 1) + y * log_mu - mu - gammaln(y + 1)
            g_loglam = x - lam      # d ll / d log(lambda)
            g_logmu = y - mu
            g_rho = np.zeros_like(x)

            tau = np.ones_like(x)
            m00 = (gl == 0) & (gv == 0)
            m01 = (gl == 0) & (gv == 1)
            m10 = (gl == 1) & (gv == 0)
            m11 = (gl == 1) & (gv == 1)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            tau = np.maximum(tau, 1e-10)
            ll = ll + np.log(tau)

            g_loglam[m00] += -lam[m00] * mu[m00] * rho / tau[m00]
            g_logmu[m00] += -lam[m00] * mu[m00] * rho / tau[m00]
            g_rho[m00] = -lam[m00] * mu[m00] / tau[m00]
            g_loglam[m01] += lam[m01] * rho / tau[m01]
            g_rho[m01] = lam[m01] / tau[m01]
            g_logmu[m10] += mu[m10] * rho / tau[m10]
            g_rho[m10] = mu[m10] / tau[m10]
            g_rho[m11] = -1 / tau[m11]

            wl, wm = w * g_loglam, w * g_logmu
            grad = np.zeros_like(p)
            grad[:n] = np.bincount(il, wl, n) + np.bincount(iv, wm, n)
            grad[n:2 * n] = np.bincount(iv, wl, n) + np.bincount(il, wm, n)
            grad[2 * n] = wl.sum() + wm.sum()
            grad[2 * n + 1] = wl.sum()
            grad[2 * n + 2] = (w * g_rho).sum()

            valor = -(w * ll).sum()
            grad = -grad
            # Regularización: L2 en ataque/defensa y suma de ataques = 0.
            pen = self.l2 * (att @ att + dfn @ dfn) + 10.0 * att.sum() ** 2
            grad[:n] += 2 * self.l2 * att + 20.0 * att.sum()
            grad[n:2 * n] += 2 * self.l2 * dfn
            return valor + pen, grad

        limites = [(None, None)] * (2 * n + 2) + [(-0.3, 0.3)]
        res = minimize(nll, x0, jac=True, method="L-BFGS-B", bounds=limites,
                       options={"maxiter": 2000})
        p = res.x
        self._x0 = p
        self.equipos = equipos
        self.ataque = dict(zip(equipos, p[:n]))
        self.defensa = dict(zip(equipos, p[n:2 * n]))
        self.base, self.local, self.rho = p[2 * n:]
        return self

    # ── predicción ────────────────────────────────────────────────────────
    def goles_esperados(self, local: str, visita: str) -> tuple[float, float]:
        al, dl = self.ataque.get(local, 0.0), self.defensa.get(local, 0.0)
        av, dv = self.ataque.get(visita, 0.0), self.defensa.get(visita, 0.0)
        lam = np.exp(self.base + self.local + al + dv)
        mu = np.exp(self.base + av + dl)
        return float(lam), float(mu)

    def matriz(self, local: str, visita: str, max_goles: int = MAX_GOLES) -> np.ndarray:
        lam, mu = self.goles_esperados(local, visita)
        return matriz_marcadores(lam, mu, self.rho, max_goles)


def matriz_marcadores(lam: float, mu: float, rho: float, max_goles: int = MAX_GOLES) -> np.ndarray:
    """P(local = i, visita = j) para i, j en 0..max_goles-1, con la corrección tau."""
    g = np.arange(max_goles)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()
