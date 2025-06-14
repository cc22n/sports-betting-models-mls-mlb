# sports-betting-models-mls-mlb
Modelos básicos en Python para predecir resultados y encontrar valor en apuestas de MLS y MLB.
Este repositorio reúne modelos sencillos de aprendizaje automático diseñados para predecir resultados y encontrar valor en apuestas deportivas de la MLS (fútbol) y la MLB (béisbol).

🎯 Objetivo
Generar predicciones sobre resultados de partidos y detectar valor (positive EV) al comparar probabilidades estimadas con cuotas del mercado.

⚙️ Características
Modelos incluidos:

Regresión logística

Random Forest

Ajustes básicos tipo bayesiano

Salidas:

Probabilidad de victoria, empate o derrota,ambos anotan, tiros de esquina,total de goles (MLS)

Moneyline y over/under (MLB)


Datos:

Históricos de partidos 

Estadísticas por equipo 

APIs deportivas

📡 APIs utilizadas
✅ MLS – API-Football (RapidAPI)
Se utiliza para obtener datos actualizados de la liga MLS.

El usuario debe crear una cuenta en RapidAPI y generar su propia clave (API Key).

El modelo incluye un control inteligente de tokens para evitar exceder el límite del plan gratuito.

✅ MLB – API oficial de MLB
Datos obtenidos directamente desde la API pública de la liga.

Sin límite de consultas, ideal para actualizaciones frecuentes sin restricciones.

🚀 Ejecución
Los notebooks están preparados para correr en Google Colab.

Solo es necesario configurar las rutas a los archivos y añadir tu clave API para el modelo de MLS.

Código limpio y modular, pensado para exploración, ajuste o extensión.

⚠️ Nota: Este proyecto es educativo y experimental. No garantiza beneficios ni precisión en apuestas reales. Úsalo bajo tu propio criterio y responsabilidad.

