# Predicciones Liga MX

Motor de predicción para apuestas de la Liga MX: 1X2, over/under y hándicap,
pensado para comparar contra las cuotas de Draftea.

## Cómo funciona

1. **Datos** (`ligamx/datos.py`): descarga `MEX.csv` de
   [football-data.co.uk](https://www.football-data.co.uk/mexico.php), con todos
   los partidos desde 2012-13 y cuotas 1X2 de cierre (Pinnacle, Bet365, Betfair,
   promedio y máximo del mercado). Etiqueta cada partido con su torneo (Apertura
   o Clausura) y su fase (regular o liguilla).
2. **Elo** (`ligamx/elo.py`): rating de cada equipo que se arrastra entre
   torneos con una regresión del 20% hacia la media.
3. **Dixon-Coles** (`ligamx/dixon_coles.py`): ataque y defensa por equipo,
   ventaja de local y corrección de marcadores bajos, ajustado por máxima
   verosimilitud. Los partidos viejos pesan menos (vida media de unos 200 días).
   Produce la probabilidad de cada marcador.
4. **Mercados** (`ligamx/mercados.py`): de la matriz de marcadores salen 1X2,
   over/under, hándicap (con devolución en líneas enteras, como en Draftea),
   ambos anotan, cuota justa, valor esperado y Kelly fraccional.
5. **Sportmonks** (`ligamx/sportmonks.py`): estadísticas por partido (tiros,
   tiros a gol, ocasiones claras). Con `--peso-xg` el modelo mezcla los goles
   con un xG aproximado a partir de los tiros. El token lo agrega el entorno
   cloud; en otra máquina, define `SPORTMONKS_API_TOKEN`.
6. **Contexto** (`ligamx/contexto.py`): altitud del estadio contra la de casa
   del visitante, distancia de viaje y días de descanso. Con `--contexto` entran
   al modelo como covariables.
7. **Alineaciones** (`sportmonks.alineaciones`, `sportmonks.regularidad_once`):
   para cada once titular, qué tan seguido jugó cada titular en los 10 partidos
   previos del equipo (1 = el once de siempre). Con `--alineaciones` entra al
   modelo. Sólo hay datos desde 2024-25.
8. **Backtest** (`ligamx/backtest.py`): walk-forward semanal, sin ver el
   futuro, contra el mercado de cierre.

9. **Extras** (`ligamx/extras.py`, con `sportmonks.detalle`): tiros, tiros a gol,
   córners y amarillas esperados por equipo, ajuste de tarjetas por árbitro,
   goles por mitad, quién anota primero, y por jugador: tiros esperados y
   probabilidad de anotar y de asistir. Se ven en la interfaz.

## Uso

```bash
pip install -r requirements.txt

# Interfaz web: elige el partido, ve probabilidades y revisa si la cuota de Draftea tiene valor
streamlit run app.py

# Apuestas sugeridas de los próximos partidos, en tres niveles de riesgo (también en la
# página "Apuestas de la jornada" de la interfaz). Guarda un CSV en apuestas/ para capturar
# la cuota de Draftea y el resultado; --evaluar resume cómo va cada nivel.
python generar_apuestas.py
python generar_apuestas.py --evaluar

# Predicción de un partido, con cuotas de Draftea opcionales
python predecir.py Toluca America --cuota over2.5=1.85 --cuota local=1.70 --cuota hcp_local-1.5=3.10

# Backtest contra el mercado
python correr_backtest.py --desde 2023-07-01

# Pruebas
python -m pytest -q
```

Claves de `--cuota`: `local`, `empate`, `visita`, `over<línea>`,
`under<línea>`, `hcp_local<línea>`, `hcp_visita<línea>`. La cuota va en formato
decimal.

## Resultados actuales

Backtest con 1,061 partidos de fase regular (Apertura 2023 a Apertura 2026),
con todo el historial desde 2012 (las cifras casi no cambian si sólo se usa
2021 en adelante):

| 1X2 | Log-loss | Brier |
|---|---|---|
| Dixon-Coles | 1.008 | 0.602 |
| Elo | 1.011 | 0.603 |
| **Mercado (promedio de cierre)** | **0.987** | **0.588** |
| Sin habilidad | 1.064 | 0.643 |

| Over/under 2.5 | Log-loss | Brier |
|---|---|---|
| Dixon-Coles | 0.691 | 0.498 |
| Tasa base | 0.689 | 0.496 |

**El modelo todavía no le gana al mercado.** Apostar 1X2 con valor esperado
mayor a 5% contra la cuota promedio perdió alrededor de 9% por apuesta. Para
over/under, el modelo aún no supera a la tasa base y es demasiado confiado en
los extremos. Es una línea base: los siguientes pasos son agregar features propias de la
Liga MX (altitud, viajes, descanso) y cuotas históricas de over/under para
medir valor en ese mercado.

Con las estadísticas de Sportmonks (sólo hay 3 temporadas en el plan Starter),
el xG aproximado por tiros **no mejoró** el modelo en 449 partidos desde julio de
2025: el log-loss 1X2 pasó de 1.030 a 1.031 con `--peso-xg 0.3`, y en over/under
2.5 de 0.703 a 0.702, ambos aún peores que el mercado o la tasa base.

Con `--contexto` (altitud, viaje y descanso), el log-loss 1X2 baja apenas de
1.008 a 1.0075 y over/under 2.5 empeora un poco (0.691 a 0.692). Los efectos sí
tienen sentido: cada 1,000 m de diferencia de altitud suben 5% los goles del
local y bajan 5% los del visitante, y cada 1,000 km de viaje bajan 10% los del
visitante. Altitud y viaje explican más de la mitad de la ventaja de local,
pero eso ya lo descuenta el mercado.

Las alineaciones son lo que más ayuda hasta ahora. En 449 partidos desde julio
de 2025, el log-loss 1X2 baja de 1.030 a 1.026 (mercado: 1.005) y over/under 2.5
de 0.703 a 0.701 (tasa base: 0.690). Un once 10 puntos más "titular" de lo
normal sube unos 4% los goles del local. La simulación de apuestas pasa de
-12% a -6% por apuesta, pero con unas 300 apuestas eso todavía es ruido. Para
usarlo al predecir hace falta la alineación confirmada, que sale cerca de una
hora antes del partido.

Lo que se probó y **no** ayudó (por eso no está en el código):
- Rating por jugador con regresión ridge tipo "más/menos" sobre goles y tiros a
  gol: 1.030, igual que sin él. Con sólo 3 temporadas no se separa la calidad
  del jugador de la del equipo.
- Lesionados y suspendidos de Sportmonks (`sidelined`), ponderados por qué tan
  titular era cada ausente: empeora a 1.034. Hay pocos registros (unos 1.5 por
  partido) y la alineación confirmada ya refleja las bajas.

Validación de los extras (479 partidos desde julio de 2025, sólo con datos
previos a cada partido):
- Quién anota primero y gol en la primera mitad: bien calibrados (53% local
  predicho contra 53% real; 72% contra 72%).
- Gol y asistencia por jugador: calibrados entre 3% y 40% (por ejemplo, a los
  que se les da 19% anotan 19%). Supone los minutos recientes del jugador.
- Tiros y córners del partido: un poco mejores que usar el promedio de la liga
  (error medio 4.9 contra 5.2 tiros; 2.69 contra 2.72 córners).
- Tiros a gol y amarillas: **no** mejoran al promedio de la liga, ni con el
  ajuste por árbitro. Tómalos como referencia descriptiva.
- Ninguno se ha comparado contra cuotas reales de esos mercados.

**No uses estas probabilidades para apostar dinero real hasta que el backtest
muestre ventaja sobre el mercado.**

## Créditos

La fórmula de Elo y la corrección de Dixon-Coles están adaptadas de
[mar-antaya/world_cup_predictions](https://github.com/mar-antaya/world_cup_predictions)
(MIT). Ver [NOTICE.md](NOTICE.md).
