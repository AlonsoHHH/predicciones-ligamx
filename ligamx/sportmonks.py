"""Estadísticas por partido de Liga MX desde la API de Sportmonks.

El token no se lee aquí: el entorno cloud lo agrega como encabezado
Authorization en cada llamada a api.sportmonks.com. Fuera de ese entorno,
define SPORTMONKS_API_TOKEN.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from .datos import DIR_CACHE

API = "https://api.sportmonks.com/v3/football"
LIGA_MX = 743
# type_id de Sportmonks -> nombre de columna
ESTADISTICAS = {86: "tiros_gol", 42: "tiros", 49: "tiros_area", 50: "tiros_fuera",
                580: "ocasiones_claras", 47: "penales", 45: "posesion", 34: "corners"}
# Nombres de Sportmonks que no coinciden con football-data.
NOMBRES = {"America": "Club America", "América": "Club America", "Club América": "Club America", "Guadalajara": "Guadalajara Chivas",
           "León": "Club Leon", "Leon": "Club Leon", "Tijuana": "Club Tijuana", "Atlético San Luis": "Atl. San Luis",
           "Atletico San Luis": "Atl. San Luis", "Mazatlán": "Mazatlan FC", "Mazatlan": "Mazatlan FC",
           "Pumas UNAM": "UNAM Pumas", "UNAM": "UNAM Pumas", "Tigres": "Tigres UANL", "Juárez": "Juarez",
           "FC Juárez": "Juarez", "Querétaro": "Queretaro", "Atlante FC": "Atlante"}


def _get(ruta: str, **params) -> dict:
    headers = {}
    if token := os.environ.get("SPORTMONKS_API_TOKEN"):
        headers["Authorization"] = token
    resp = requests.get(f"{API}/{ruta}", params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def temporadas() -> list[int]:
    datos = _get(f"leagues/{LIGA_MX}", include="seasons")["data"]
    return [s["id"] for s in sorted(datos["seasons"], key=lambda s: s["starting_at"] or "")]


def partidos_temporada(season_id: int) -> pd.DataFrame:
    filas = []
    # El include anidado trae todos los partidos de la temporada en una sola respuesta.
    r = _get(f"seasons/{season_id}", include="fixtures.statistics;fixtures.participants")
    for f in r["data"].get("fixtures") or []:
        if f.get("state_id") != 5:   # 5 = terminado
            continue
        equipos = {p["meta"]["location"]: p for p in f["participants"]}
        fila = {"fecha": pd.Timestamp(f["starting_at"]).normalize(),
                "local": NOMBRES.get(equipos["home"]["name"], equipos["home"]["name"]),
                "visita": NOMBRES.get(equipos["away"]["name"], equipos["away"]["name"])}
        for s in f.get("statistics") or []:
            col = ESTADISTICAS.get(s["type_id"])
            if col and s.get("location") in ("home", "away"):
                fila[f"{col}_{'l' if s['location'] == 'home' else 'v'}"] = s["data"]["value"]
        filas.append(fila)
    return pd.DataFrame(filas)


def descargar(forzar: bool = False) -> pd.DataFrame:
    """Todas las temporadas a las que da acceso el plan, en caché local."""
    ruta: Path = DIR_CACHE / "sportmonks_estadisticas.csv"
    if ruta.exists() and not forzar:
        return pd.read_csv(ruta, parse_dates=["fecha"])
    df = pd.concat([partidos_temporada(s) for s in temporadas()], ignore_index=True)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    return df


TIROS = ["tiros_area", "tiros_fuera", "tiros_gol", "ocasiones_claras", "penales"]


def unir(partidos: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Agrega las estadísticas a los partidos de football-data. Las fechas pueden
    diferir un día por la zona horaria, así que se une al partido más cercano (±2 días)."""
    stats = stats.sort_values("fecha").rename(columns={"fecha": "fecha_sm"})
    out = pd.merge_asof(partidos.sort_values("fecha"), stats, left_on="fecha", right_on="fecha_sm",
                        by=["local", "visita"], direction="nearest", tolerance=pd.Timedelta(days=2))
    return out.drop(columns="fecha_sm")


def ajustar_xg(df: pd.DataFrame):
    """Regresión de Poisson goles ~ tiros. Devuelve una función que da el xG aproximado
    de local y visita (NaN si el partido no tiene estadísticas)."""
    from sklearn.linear_model import PoissonRegressor

    def apilar(d):
        loc = d[[f"{c}_l" for c in TIROS]].set_axis(TIROS, axis=1)
        vis = d[[f"{c}_v" for c in TIROS]].set_axis(TIROS, axis=1)
        return pd.concat([loc, vis], ignore_index=True).fillna({"ocasiones_claras": 0, "penales": 0})

    con = df.dropna(subset=["tiros_gol_l", "tiros_gol_v"])
    modelo = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(apilar(con), pd.concat([con["gl"], con["gv"]]))

    def xg(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        hay = d["tiros_gol_l"].notna() & d["tiros_gol_v"].notna()
        xl = pd.Series(float("nan"), index=d.index)
        xv = xl.copy()
        if hay.any():
            pred = modelo.predict(apilar(d[hay]))
            xl[hay], xv[hay] = pred[: hay.sum()], pred[hay.sum():]
        return xl, xv

    return xg


def alineaciones(forzar: bool = False) -> pd.DataFrame:
    """Titulares por partido: fecha, local, visita, lado ('l'/'v'), player_id."""
    ruta: Path = DIR_CACHE / "sportmonks_alineaciones.csv"
    if ruta.exists() and not forzar:
        return pd.read_csv(ruta, parse_dates=["fecha"])
    filas = []
    for season_id in temporadas():
        r = _get(f"seasons/{season_id}", include="fixtures.lineups;fixtures.participants")
        for f in r["data"].get("fixtures") or []:
            if f.get("state_id") != 5:
                continue
            equipos = {p["meta"]["location"]: p for p in f["participants"]}
            lado = {equipos["home"]["id"]: "l", equipos["away"]["id"]: "v"}
            base = {"fecha": pd.Timestamp(f["starting_at"]).normalize(),
                    "local": NOMBRES.get(equipos["home"]["name"], equipos["home"]["name"]),
                    "visita": NOMBRES.get(equipos["away"]["name"], equipos["away"]["name"])}
            for j in f.get("lineups") or []:
                if j["type_id"] == 11 and j["team_id"] in lado:   # 11 = titular
                    filas.append({**base, "lado": lado[j["team_id"]], "player_id": j["player_id"]})
    df = pd.DataFrame(filas)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    return df


def regularidad_once(alin: pd.DataFrame, ventana: int = 10) -> pd.DataFrame:
    """Qué tan "de gala" es cada once: promedio, entre los 11 titulares, de la
    fracción de los `ventana` partidos anteriores del equipo que cada uno fue titular.
    1 = el once de siempre; valores bajos = rotación o bajas. Sólo usa partidos previos."""
    alin = alin.assign(equipo=lambda d: d["local"].where(d["lado"] == "l", d["visita"]))
    filas = []
    for equipo, d in alin.groupby("equipo"):
        partidos = d.groupby(["fecha", "local", "visita", "lado"])["player_id"].apply(set).sort_index(level=0)
        previos: list[set] = []
        for (fecha, local, visita, lado), titulares in partidos.items():
            if len(previos) >= 3:
                h = previos[-ventana:]
                valor = sum(sum(p in s for s in h) / len(h) for p in titulares) / len(titulares)
                filas.append({"fecha": fecha, "local": local, "visita": visita, f"once_{lado}": valor})
            previos.append(titulares)
    out = pd.DataFrame(filas)
    return out.groupby(["fecha", "local", "visita"], as_index=False).first()


# type_id de las estadísticas por jugador (lineups.details)
POR_JUGADOR = {119: "minutos", 42: "tiros", 86: "tiros_gol", 52: "goles", 79: "asistencias",
               117: "pases_clave", 84: "amarillas"}
GOL, PENAL, AUTOGOL = 14, 16, 15   # tipos de evento
ARBITRO_CENTRAL = 6


def detalle(forzar: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tres tablas por partido terminado:
    - jugadores: una fila por jugador que jugó, con minutos, tiros, tiros a gol, goles,
      asistencias, pases clave y amarillas.
    - goles: una fila por gol, con minuto y lado.
    - arbitros: árbitro central y amarillas de cada lado."""
    rutas = [DIR_CACHE / f"sportmonks_{n}.csv" for n in ("jugadores", "goles", "arbitros")]
    if all(r.exists() for r in rutas) and not forzar:
        return tuple(pd.read_csv(r, parse_dates=["fecha"]) for r in rutas)
    jugadores, goles, arbitros = [], [], []
    for season_id in temporadas():
        r = _get(f"seasons/{season_id}", include="fixtures.lineups.details;fixtures.events;"
                                                 "fixtures.referees.referee;fixtures.participants")
        for f in r["data"].get("fixtures") or []:
            if f.get("state_id") != 5:
                continue
            equipos = {p["meta"]["location"]: p for p in f["participants"]}
            lado = {equipos["home"]["id"]: "l", equipos["away"]["id"]: "v"}
            base = {"fecha": pd.Timestamp(f["starting_at"]).normalize(),
                    "local": NOMBRES.get(equipos["home"]["name"], equipos["home"]["name"]),
                    "visita": NOMBRES.get(equipos["away"]["name"], equipos["away"]["name"])}
            amarillas = {"l": 0, "v": 0}
            for j in f.get("lineups") or []:
                d = {POR_JUGADOR[s["type_id"]]: s["data"]["value"] for s in j.get("details") or []
                     if s["type_id"] in POR_JUGADOR}
                if j["team_id"] not in lado or not d.get("minutos"):
                    continue
                amarillas[lado[j["team_id"]]] += d.get("amarillas", 0)
                jugadores.append({**base, "lado": lado[j["team_id"]], "player_id": j["player_id"],
                                  "jugador": j["player_name"], "titular": j["type_id"] == 11, **d})
            for e in f.get("events") or []:
                if e["type_id"] in (GOL, PENAL, AUTOGOL) and e["participant_id"] in lado:
                    goles.append({**base, "lado": lado[e["participant_id"]], "minuto": e["minute"]})
            central = next((a["referee"]["common_name"] for a in f.get("referees") or []
                            if a["type_id"] == ARBITRO_CENTRAL and a.get("referee")), None)
            arbitros.append({**base, "arbitro": central, "amarillas_l": amarillas["l"],
                             "amarillas_v": amarillas["v"]})
    tablas = (pd.DataFrame(jugadores).fillna({c: 0 for c in POR_JUGADOR.values()}),
              pd.DataFrame(goles), pd.DataFrame(arbitros))
    for t, ruta in zip(tablas, rutas):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(ruta, index=False)
    return tablas
