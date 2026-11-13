import numpy as np
import pandas as pd
import pytest

from ligamx import datos, mercados
from ligamx.dixon_coles import matriz_marcadores


@pytest.fixture
def m():
    return matriz_marcadores(1.5, 1.1, -0.05)


def test_matriz_suma_uno(m):
    assert m.sum() == pytest.approx(1.0)


def test_1x2_suma_uno(m):
    assert sum(mercados.uno_x_dos(m).values()) == pytest.approx(1.0)


def test_over_under_complementarios(m):
    o, u = mercados.over(m, 2.5), mercados.under(m, 2.5)
    assert o.gana + u.gana == pytest.approx(1.0)
    assert o.anula == 0


def test_linea_entera_devuelve(m):
    o = mercados.over(m, 2.0)
    i, j = np.indices(m.shape)
    assert o.anula == pytest.approx(m[(i + j) == 2].sum())
    assert o.gana + o.anula + o.pierde == pytest.approx(1.0)


def test_handicap_medio_es_1x2(m):
    p = mercados.uno_x_dos(m)
    assert mercados.handicap(m, -0.5, "local").gana == pytest.approx(p["local"])
    assert mercados.handicap(m, 0.5, "visita").gana == pytest.approx(p["visita"] + 0 * p["empate"] + p["empate"])


def test_handicap_cero_anula_empate(m):
    p = mercados.uno_x_dos(m)
    h = mercados.handicap(m, 0.0, "local")
    assert h.anula == pytest.approx(p["empate"])


def test_cuarto_es_promedio(m):
    q = mercados.over(m, 2.25)
    a, b = mercados.over(m, 2.0), mercados.over(m, 2.5)
    assert q.gana == pytest.approx((a.gana + b.gana) / 2)


def test_valor_esperado_cero_en_cuota_justa(m):
    r = mercados.over(m, 2.0)
    assert r.valor_esperado(r.cuota_justa()) == pytest.approx(0.0, abs=1e-12)
    assert mercados.kelly(r, r.cuota_justa()) == pytest.approx(0.0, abs=1e-12)


def test_fase_y_torneo():
    filas = []
    fecha = pd.Timestamp("2025-07-10")
    equipos = [f"E{i}" for i in range(18)]
    # 17 jornadas regulares + 1 partido de liguilla
    for jornada in range(17):
        for k in range(9):
            filas.append({"fecha": fecha + pd.Timedelta(days=7 * jornada), "local": equipos[k],
                          "visita": equipos[17 - k], "torneo": "Apertura 2025"})
        equipos = [equipos[0]] + [equipos[-1]] + equipos[1:-1]
    filas.append({"fecha": fecha + pd.Timedelta(days=130), "local": "E0", "visita": "E1",
                  "torneo": "Apertura 2025"})
    fases = datos.etiquetar_fase(pd.DataFrame(filas))
    assert (fases[:-1] == "regular").all() and fases.iloc[-1] == "liguilla"
    assert datos.nombre_torneo(pd.Timestamp("2026-01-15")) == "Clausura 2026"
    assert datos.nombre_torneo(pd.Timestamp("2026-07-17")) == "Apertura 2026"


def test_resolver_equipo():
    eq = ["Club America", "Guadalajara Chivas", "UNAM Pumas", "Toluca"]
    assert datos.resolver_equipo("América", eq) == "Club America"
    assert datos.resolver_equipo("chivas", eq) == "Guadalajara Chivas"
    assert datos.resolver_equipo("toluca", eq) == "Toluca"
    with pytest.raises(ValueError):
        datos.resolver_equipo("Barcelona", eq)


def test_unir_tolera_un_dia_de_diferencia():
    from ligamx import sportmonks
    partidos = pd.DataFrame({"fecha": pd.to_datetime(["2026-11-08", "2026-11-09"]),
                             "local": ["Toluca", "Atlas"], "visita": ["Atlas", "Toluca"]})
    stats = pd.DataFrame({"fecha": pd.to_datetime(["2026-11-09", "2026-11-20"]),
                          "local": ["Toluca", "Atlas"], "visita": ["Atlas", "Toluca"], "tiros_gol_l": [5, 3]})
    u = sportmonks.unir(partidos, stats)
    assert u.loc[u.local == "Toluca", "tiros_gol_l"].item() == 5
    assert u.loc[u.local == "Atlas", "tiros_gol_l"].isna().item()


def test_contexto_altitud_viaje_descanso():
    from ligamx import contexto
    df = pd.DataFrame({"fecha": pd.to_datetime(["2026-08-01", "2026-08-08", "2026-08-11"]),
                       "local": ["Toluca", "Club Tijuana", "Toluca"],
                       "visita": ["Club Tijuana", "Toluca", "Club Tijuana"]})
    c = contexto.agregar(df)
    assert c.loc[0, "altitud"] == pytest.approx(2.63)
    assert c.loc[1, "altitud"] == pytest.approx(-2.63)
    assert 2.0 < c.loc[0, "viaje"] < 2.5          # Toluca-Tijuana, unos 2,300 km
    assert c.loc[2, "descanso"] == pytest.approx(0.0)  # ambos con 3 días


def test_regularidad_once():
    from ligamx import sportmonks
    filas = []
    fechas = pd.date_range("2025-08-01", periods=5, freq="7D")
    for i, f in enumerate(fechas):
        titulares = range(11) if i < 4 else list(range(6)) + list(range(100, 105))
        filas += [{"fecha": f, "local": "Toluca", "visita": f"R{i}", "lado": "l", "player_id": p}
                  for p in titulares]
    r = sportmonks.regularidad_once(pd.DataFrame(filas)).set_index("visita")["once_l"]
    assert r["R3"] == pytest.approx(1.0)        # mismo once las 3 previas
    assert r["R4"] == pytest.approx(6 / 11)     # 5 cambios
    assert "R0" not in r.index                  # sin historia suficiente


def test_mitades_y_primer_gol():
    from ligamx import extras
    x = extras.mitades_y_primer_gol(1.8, 1.2)
    assert x["anota_primero_local"] + x["anota_primero_visita"] + x["sin_goles"] == pytest.approx(1.0)
    assert x["anota_primero_local"] / x["anota_primero_visita"] == pytest.approx(1.5)
    assert x["gol_2a_mitad"] > x["gol_1a_mitad"]


def test_esperados_equipo_y_arbitro():
    from ligamx import extras
    f = pd.date_range("2026-08-01", periods=4, freq="7D")
    stats = pd.DataFrame({"fecha": f, "local": ["A", "B", "A", "C"], "visita": ["B", "C", "C", "A"],
                          "tiros_l": [20, 10, 20, 10], "tiros_v": [10, 10, 10, 20]})
    e = extras.esperados_equipo(stats, "A", "B")
    assert e.loc["tiros", "local"] > stats["tiros_l"].mean()      # A tira más que la liga
    arb = pd.DataFrame({"arbitro": ["X"] * 5 + ["Y"] * 5, "amarillas_l": [4] * 5 + [1] * 5,
                        "amarillas_v": [4] * 5 + [1] * 5})
    suyo, n, liga = extras.tarjetas_arbitro(arb, "X")
    assert (n, liga) == (5, 5.0) and suyo == pytest.approx(6.5)    # a medio camino entre 8 y 5
    assert extras.tarjetas_arbitro(arb, "nuevo")[0] == pytest.approx(5.0)


def test_jugadores_partido_reparte_los_goles():
    from ligamx import extras
    filas = []
    for i, f in enumerate(pd.date_range("2026-08-01", periods=6, freq="7D")):
        for pid, (goles, tg) in enumerate([(1, 3), (0, 1), (0, 0)]):
            filas.append({"fecha": f, "local": "A", "visita": f"R{i}", "lado": "l", "player_id": pid,
                          "jugador": f"J{pid}", "minutos": 90, "tiros": tg + 1, "tiros_gol": tg,
                          "goles": goles, "asistencias": int(pid == 1), "pases_clave": pid})
    t = extras.jugadores_partido(pd.DataFrame(filas), "A", 1.5)
    assert list(t["jugador"]) == ["J0", "J1", "J2"]
    assert -np.log(1 - t["p_gol"]).sum() == pytest.approx(1.5)     # los goles esperados se reparten completos


def test_generador_niveles_y_evaluacion():
    from ligamx import generador
    cand = pd.DataFrame({"partido": [f"P{i}" for i in range(5) for _ in range(3)],
                         "seleccion": ["segura", "media", "dificil"] * 5,
                         "prob": [0.80, 0.58, 0.30] * 5})
    g = generador.generar(cand)
    assert list(g["nivel"].unique()) == ["Bajo", "Medio", "Alto"]
    parlay = g[g["tipo"].str.startswith("Parlay")].iloc[0]
    assert parlay["prob"] == pytest.approx(0.8 * 0.8) and parlay["apuesta"].count("(P") == 2
    assert np.allclose(g["cuota_minima"], generador.MARGEN / g["prob"])
    # un parlay nunca repite partido
    for a in g["apuesta"]:
        partidos = [x.split(")")[0] for x in a.split("(")[1:]]
        assert len(partidos) == len(set(partidos))

    reg = pd.DataFrame({"nivel": ["Bajo", "Bajo", "Bajo"], "prob": [0.8] * 3, "unidades": [2.0] * 3,
                        "cuota_minima": [1.3] * 3, "cuota_draftea": [1.5, 1.5, 1.2], "gano": [1, 0, 1]})
    e = generador.evaluar(reg).loc["Bajo"]      # la de 1.2 no llega a la mínima y no cuenta
    assert e["apuestas"] == 2 and e["ganancia"] == pytest.approx(2 * 0.5 - 2)
