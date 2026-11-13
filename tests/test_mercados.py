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
