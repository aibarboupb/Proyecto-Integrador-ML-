# Predicción del nivel de desempeño - Pruebas Saber 11 (Medellín)

Trabajo integrador desarrollado siguiendo la metodología **CRISP-DM** (6 fases),
sobre la versión **depurada** del histórico de resultados Saber 11 de Medellín.
Predice, a partir de variables institucionales de un colegio (sector, comuna,
tamaño, participación y comparación con promedios de ciudad/comuna), el nivel
de desempeño esperado en las Pruebas Saber 11: **Bajo, Medio, Alto o Superior**.

## Dataset

- **Fuente:** Histórico de resultados Pruebas Saber 11, Alcaldía de Medellín (Medata), versión depurada.
- **Filas:** 1.660 (colegio × periodo, grado 11, 2014-2 a 2018-2).
- **Variable objetivo:** `nivel_desempeno`, construida en 4 cuartiles sobre `puntaje_global`.
- Esta versión depurada ya NO incluye los puntajes por área (lectura, matemáticas, etc.)
  ni los promedios por sector, así que no hay riesgo de fuga de información: todas
  las columnas disponibles se usan como variables predictoras.

## Estructura del repositorio

```
├── app.py                  # App de Streamlit (interfaz de predicción)
├── pipeline.pkl             # Pipeline serializado completo (preprocesamiento + modelo ganador)
├── requirements.txt        # Dependencias para ejecutar la app
├── data/                   # Dataset original y procesado + ejemplo de datos nuevos
├── notebooks/               # Notebook completo (Google Colab) con las 6 fases CRISP-DM
├── src/                     # Scripts Python de cada fase (reproducibles paso a paso)
├── reports/                 # Diccionario de datos, reglas de calidad, métricas, reporte ydata-profiling
└── figures/                  # Gráficos generados durante el análisis
```

## Metodología (resumen de las 6 fases CRISP-DM)

1. **Entendimiento del negocio:** problema de clasificación multiclase (4 niveles)
   a nivel de institución educativa.
2. **Entendimiento de los datos:** diccionario de datos, reglas de calidad,
   distribución de la variable objetivo, gráficos de descripción estadística
   de cada variable, y 4 sesgos documentados (ver `reports/`).
3. **Preparación de los datos:** selección de variables, limpieza de errores
   y nulos, variables derivadas, análisis de relaciones lineales/no lineales,
   y justificación de no aplicar reducción de dimensiones ni balanceo.
4. **Modelamiento:** 5 modelos clásicos (Regresión Logística, KNN, Árbol de
   Decisión, Naive Bayes, SVM), 3 modelos de ensamble (Voting, Bagging,
   Boosting/GBM) y ajuste de hiperparámetros con `RandomizedSearchCV` (5-fold,
   `StratifiedGroupKFold` agrupado por colegio, métrica F1-macro).
5. **Evaluación:** comparación final en conjunto de prueba (30%, por colegio,
   nunca visto durante el entrenamiento). Modelo ganador: **Bagging**.
6. **Despliegue:** pipeline serializado, predicciones sobre datos nuevos
   simulados, app de Streamlit, y este repositorio público.

## Cómo correr la app localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cómo desplegar en Streamlit Community Cloud

1. Sube este repositorio a tu cuenta de GitHub.
2. Entra a https://share.streamlit.io/ con tu cuenta de GitHub.
3. "New app" → selecciona el repositorio, rama `main` y archivo `app.py`.
4. Click en "Deploy". En 1-2 minutos tendrás la URL pública.

## Autor

Ana Milena Ibarbo G · Trabajo Integrador CRISP-DM
