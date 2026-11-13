"""Interfaz web del modelo.

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ligamx import datos, elo, extras, mercados, sportmonks
from ligamx.dixon_coles import DixonColes

st.set_page_config(page_title="Predicciones Liga MX", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner="Cargando partidos y ajustando el modelo…")
def cargar(actualizar: bool = False):
    df = datos.cargar(forzar_descarga=actualizar)
    modelo = DixonColes().ajustar(df)
    _, ratings = elo.calcular(df)
    actuales = sorted(set(df.loc[df["torneo"] == df["torneo"].iloc[-1], ["local", "visita"]].stack()))
    return df, modelo, ratings, actuales


@st.cache_data(show_spinner="Descargando estadísticas de Sportmonks…")
def cargar_sportmonks(actualizar: bool = False):
    """Estadísticas de equipo (con amarillas y árbitro), jugadores y goles; None si no hay acceso."""
    try:
        jug, _, arb = sportmonks.detalle(forzar=actualizar)
        stats = sportmonks.descargar(forzar=actualizar).merge(arb, on=["fecha", "local", "visita"])
        return stats, jug, arb
    except Exception as e:   # sin token o sin red: la app sigue con el modelo de goles
        return str(e)


with st.sidebar:
    st.header("Datos")
    if st.button("Actualizar resultados"):
        cargar.clear()
        cargar(actualizar=True)
        cargar_sportmonks.clear()
        cargar_sportmonks(actualizar=True)
    df, modelo, ratings, equipos = cargar()
    st.caption(f"Partidos hasta el {df['fecha'].max():%d/%m/%Y} · {len(df):,} partidos desde 2012")
    st.subheader("Tabla de poder (Elo)")
    st.dataframe(pd.DataFrame({"Equipo": equipos, "Elo": [round(ratings[e]) for e in equipos]})
                 .sort_values("Elo", ascending=False), hide_index=True, width="stretch")

st.title("Predicciones Liga MX")
st.warning("En el backtest, el mercado de cierre todavía es más preciso que este modelo. "
           "Úsalo como referencia, no para apostar dinero real.")

c1, c2 = st.columns(2)
local = c1.selectbox("Local", equipos, index=equipos.index("Toluca") if "Toluca" in equipos else 0)
visita = c2.selectbox("Visitante", [e for e in equipos if e != local])

m = modelo.matriz(local, visita)
lam, mu = modelo.goles_esperados(local, visita)
r = mercados.resumen(m, lineas_totales=(0.5, 1.5, 2.5, 3.5, 4.5))
p = r["1x2"]

k = st.columns(5)
for col, nombre, prob in ((k[0], f"Gana {local}", p["local"]), (k[1], "Empate", p["empate"]),
                          (k[2], f"Gana {visita}", p["visita"]), (k[4], "Ambos anotan", r["ambos_anotan"])):
    col.metric(nombre, f"{prob:.0%}")
    col.caption(f"Cuota justa {1 / prob:.2f}")
k[3].metric("Goles esperados", f"{lam:.2f} - {mu:.2f}")



def tabla(filas: list[tuple[str, mercados.Resultado]]) -> None:
    t = pd.DataFrame({"Mercado": [n for n, _ in filas],
                      "Prob.": [f"{res.gana:.1%}" for _, res in filas],
                      "Devuelve": [f"{res.anula:.1%}" for _, res in filas],
                      "Cuota justa": [f"{res.cuota_justa():.2f}" for _, res in filas]})
    if not any(res.anula for _, res in filas):   # sólo las líneas enteras de hándicap devuelven
        t = t.drop(columns="Devuelve")
    st.dataframe(t, hide_index=True, width="stretch")


totales, handicaps = [], []
for linea, d in r["totales"].items():
    totales += [(f"Over {linea}", d["over"]), (f"Under {linea}", d["under"])]
for linea, d in r["handicap"].items():
    handicaps += [(f"{local} {linea:+.1f}", d["local"]), (f"{visita} {-linea:+.1f}", d["visita"])]
uno_x_dos = [(f"Gana {local}", mercados.Resultado(p["local"], 0, 1 - p["local"])),
             ("Empate", mercados.Resultado(p["empate"], 0, 1 - p["empate"])),
             (f"Gana {visita}", mercados.Resultado(p["visita"], 0, 1 - p["visita"]))]

st.subheader("¿Tiene valor la cuota de Draftea?")
c1, c2, c3 = st.columns([2, 1, 3])
todos = dict(totales + handicaps + uno_x_dos)
mercado = c1.selectbox("Mercado", list(todos), index=list(todos).index("Over 2.5"))
cuota = c2.number_input("Cuota Draftea", min_value=1.01, value=2.00, step=0.05, format="%.2f")
res = todos[mercado]
ve = res.valor_esperado(cuota)
with c3:
    st.write("")
    texto = f"Cuota justa {res.cuota_justa():.2f} · valor esperado **{ve:+.1%}**"
    if ve > 0.03:
        st.success(f"{texto} · Kelly 1/4: {mercados.kelly(res, cuota):.1%} del bankroll")
    else:
        st.info(f"{texto} · sin valor")

tab1, tab2, tab3 = st.tabs(["Over/under", "Hándicap", "1X2"])
with tab1:
    tabla(totales)
with tab2:
    tabla(handicaps)
with tab3:
    tabla(uno_x_dos)

st.subheader("Estadísticas del partido")
sm = cargar_sportmonks()
t1, t2, t3 = st.tabs(["Mitades y primer gol", "Tiros, córners y tarjetas", "Jugadores"])
with t1:
    x = extras.mitades_y_primer_gol(lam, mu)
    c = st.columns(3)
    c[0].metric(f"Anota primero {local}", f"{x['anota_primero_local']:.0%}")
    c[1].metric(f"Anota primero {visita}", f"{x['anota_primero_visita']:.0%}")
    c[2].metric("Sin goles", f"{x['sin_goles']:.0%}")
    st.dataframe(pd.DataFrame({
        "": ["Hay gol", f"Anota {local}", f"Anota {visita}"],
        "1a mitad": [f"{x[k]:.0%}" for k in ("gol_1a_mitad", "local_anota_1a", "visita_anota_1a")],
        "2a mitad": [f"{x[k]:.0%}" for k in ("gol_2a_mitad", "local_anota_2a", "visita_anota_2a")]}),
        hide_index=True, width="stretch")
    st.caption(f"Más goles en la 2a mitad que en la 1a: {x['mas_goles_2a_mitad']:.0%}. "
               "En la Liga MX, 45% de los goles caen en la primera mitad.")
if isinstance(sm, str):
    for t in (t2, t3):
        t.info(f"No pude leer Sportmonks ({sm}). Define SPORTMONKS_API_TOKEN para ver esta sección.")
else:
    stats, jug, arb = sm
    with t2:
        e = extras.esperados_equipo(stats, local, visita)
        conteo = arb["arbitro"].value_counts()
        arbitro = st.selectbox("Árbitro", ["(sin definir)"] + list(conteo.index))
        if arbitro != "(sin definir)":
            suyo, n, liga = extras.tarjetas_arbitro(arb, arbitro)
            e.loc["amarillas"] *= suyo / liga
            st.caption(f"{arbitro}: {suyo:.1f} amarillas por partido en {n} partidos (liga: {liga:.1f}).")
        e.index = ["Tiros", "Tiros a gol", "Córners", "Amarillas"]
        st.dataframe(e.rename(columns={"local": local, "visita": visita, "total": "Total"}).round(1),
                     width="stretch")
        st.caption("Promedio de los últimos 10 partidos de cada equipo, ajustado por rival. En el backtest "
                   "sólo tiros y córners mejoran un poco al promedio de la liga; tarjetas y tiros a gol no.")
    with t3:
        for col, equipo, tasa in zip(st.tabs([local, visita]), (local, visita), (lam, mu)):
            j = extras.jugadores_partido(jug, equipo, tasa).head(14)
            col.dataframe(pd.DataFrame({
                "Jugador": j["jugador"], "Anota": [f"{v:.0%}" for v in j["p_gol"]],
                "Cuota justa": [f"{1 / v:.2f}" for v in j["p_gol"]],
                "Asiste": [f"{v:.0%}" for v in j["p_asistencia"]],
                "Tiros": j["tiros_esp"].round(1), "A gol": j["tiros_gol_esp"].round(1),
                "Min.": j["min_esp"].round(0)}), hide_index=True, width="stretch")
        st.caption("Con los minutos promedio de los últimos 5 partidos: si un jugador no es titular, "
                   "su probabilidad real es menor. No considera lesiones ni suspensiones.")

with st.expander("Marcadores más probables"):
    top = sorted(((m[i, j], f"{i}-{j}") for i in range(6) for j in range(6)), reverse=True)[:10]
    st.dataframe(pd.DataFrame({"Marcador": [s for _, s in top], "Prob.": [f"{q:.1%}" for q, _ in top]}),
                 hide_index=True)
