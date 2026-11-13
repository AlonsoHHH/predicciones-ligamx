"""Apuestas sugeridas de los próximos partidos, por nivel de riesgo."""

import streamlit as st

import estilo
from generar_apuestas import apuestas

st.set_page_config(page_title="Apuestas de la jornada", page_icon="⚽", layout="wide")
estilo.aplicar()
st.title("Apuestas de la jornada")
st.warning("El modelo no conoce las cuotas de Draftea y todavía no le gana al mercado. Esto es para "
           "registrar y evaluar: apuesta sólo si Draftea paga la cuota mínima o más, y con montos chicos.")

dias = st.slider("Días hacia adelante", 1, 14, 10)
partidos, sugeridas = st.cache_data(apuestas, show_spinner="Actualizando datos y generando…", ttl=3600)(dias)
if sugeridas.empty:
    st.info("No hay partidos programados en ese periodo.")
    st.stop()

with st.expander(f"{len(partidos)} partidos"):
    estilo.tabla(partidos.assign(inicio=partidos["inicio"].dt.strftime("%a %d/%m %H:%M")))

for nivel, bloque in sugeridas.groupby("nivel", sort=False):
    st.subheader(f"Riesgo {nivel.lower()} · {bloque['unidades'].iloc[0]:g} unidades por apuesta")
    estilo.tabla(bloque.assign(prob=bloque["prob"].map("{:.0%}".format),
                               cuota_justa=bloque["cuota_justa"].map("{:.2f}".format),
                               cuota_minima=bloque["cuota_minima"].map("{:.2f}".format))
                 [["tipo", "apuesta", "prob", "cuota_justa", "cuota_minima"]]
                 .rename(columns={"tipo": "Tipo", "apuesta": "Apuesta", "prob": "Prob.",
                                  "cuota_justa": "Cuota justa", "cuota_minima": "Cuota mínima"}), ("Prob.",))

st.download_button("Descargar hoja de registro (CSV)", sugeridas.round(3).to_csv(index=False),
                   file_name=f"apuestas_{partidos['inicio'].min():%Y-%m-%d}.csv")
st.caption("En el CSV, llena cuota_draftea y después del partido gano (1 o 0). "
           "Luego corre `python generar_apuestas.py --evaluar` con los archivos en la carpeta apuestas/.")
