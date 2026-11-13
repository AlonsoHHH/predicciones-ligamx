"""Aspecto de la interfaz: fondo morado y negro, islas de vidrio y verde para lo probable."""

from __future__ import annotations

import pandas as pd
import streamlit as st

ALTA = 0.60   # desde aquí una probabilidad se marca en verde

CSS = """
<style>
.stApp { background:
    radial-gradient(1100px 620px at 12% -8%, rgba(124, 58, 237, .55), transparent 62%),
    radial-gradient(900px 560px at 105% 18%, rgba(192, 132, 252, .28), transparent 60%),
    radial-gradient(900px 700px at 50% 115%, rgba(91, 33, 182, .40), transparent 60%),
    #09080f; background-attachment: fixed; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.5rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -.02em; }

/* islas */
.isla, [data-testid="stTable"], [data-testid="stExpander"] details, [data-testid="stAlert"],
section[data-testid="stSidebar"] > div:first-child {
    background: rgba(255, 255, 255, .06); border: 1px solid rgba(255, 255, 255, .14);
    border-radius: 18px; backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, .35); }
section[data-testid="stSidebar"] { background: transparent; padding: 12px 0 12px 12px; }

.isla { padding: 14px 16px; margin-bottom: 1rem; min-height: 104px; }
.isla small { display: block; font-size: .68rem; letter-spacing: .09em; text-transform: uppercase;
              color: rgba(245, 243, 255, .62); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.isla b { display: block; font-size: 1.8rem; line-height: 1.25; }
.isla span { font-size: .78rem; color: rgba(245, 243, 255, .62); }
.isla.alta { border-color: rgba(52, 211, 153, .55); background: rgba(16, 185, 129, .12); }
.isla.alta b { color: #34d399; }

/* tablas */
[data-testid="stTable"] { padding: 4px 10px; overflow-x: auto; }
[data-testid="stTable"] table { border-collapse: collapse; }
[data-testid="stTable"] .row_heading, [data-testid="stTable"] .index_name { display: none; }  /* índice */
[data-testid="stTable"] th, [data-testid="stTable"] td { border: 0 !important;
    border-bottom: 1px solid rgba(255, 255, 255, .08) !important; }
[data-testid="stTable"] tr:last-child td { border-bottom: 0 !important; }
[data-testid="stTable"] th { font-size: .68rem; letter-spacing: .09em; text-transform: uppercase;
    color: rgba(245, 243, 255, .62) !important; font-weight: 500; }

/* controles */
[data-baseweb="select"] > div, [data-testid="stNumberInput"] > div, .stButton button, .stDownloadButton button {
    background: rgba(255, 255, 255, .07); border: 1px solid rgba(255, 255, 255, .16); border-radius: 12px; }
.stButton button, .stDownloadButton button { background: #fff; color: #09080f; font-weight: 600; border: 0; }
.stButton button:hover, .stDownloadButton button:hover { background: #a855f7; color: #fff; }
[role="tablist"] { gap: 6px; border: 0; padding-bottom: 6px; }
[data-testid="stTab"] { border-radius: 999px; padding: 4px 14px; height: auto;
    background: rgba(255, 255, 255, .06); border: 1px solid rgba(255, 255, 255, .12); }
[data-testid="stTab"][aria-selected="true"] { background: #a855f7; border-color: #a855f7; }
[data-testid="stTab"][aria-selected="true"] p { color: #fff; }
[data-testid="stTab"] .react-aria-SelectionIndicator { display: none; }
[data-testid="stSidebar"] { border-right: 0; }
</style>
"""
VERDE = "color: #34d399; font-weight: 600; background-color: rgba(16, 185, 129, .14)"


def aplicar() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def isla(en, titulo: str, valor: str, nota: str = "", alta: bool = False) -> None:
    """Tarjeta de vidrio con una cifra; `alta` la pinta de verde."""
    en.markdown(f'<div class="isla{" alta" if alta else ""}"><small>{titulo}</small>'
                f"<b>{valor}</b><span>{nota}</span></div>", unsafe_allow_html=True)


def es_alta(v, umbral: float = ALTA) -> bool:
    """Acepta 0.72 o '72%'."""
    return (float(v.rstrip("%")) / 100 if isinstance(v, str) else v) >= umbral


def tabla(df: pd.DataFrame, verdes: tuple[str, ...] = (), umbral: float = ALTA, en=st) -> None:
    """Tabla (el índice lo oculta el CSS); en las columnas `verdes` resalta las probabilidades de `umbral` o más."""
    s = df.style.format(precision=1)
    if verdes:
        s = s.map(lambda v: VERDE if es_alta(v, umbral) else "", subset=list(verdes))
    en.table(s)
