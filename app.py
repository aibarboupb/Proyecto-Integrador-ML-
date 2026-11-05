# -*- coding: utf-8 -*-
"""
App de Streamlit - Predicción de nivel de desempeño Saber 11 (Medellín)
=========================================================================
Trabajo Integrador - Metodología CRISP-DM
Autor: Jorge
"""
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Predicción Saber 11 - Medellín", page_icon="🎓", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipeline.pkl")

@st.cache_resource
def cargar_modelo():
    paquete = joblib.load(MODEL_PATH)
    return paquete["pipeline"], paquete["metadatos"]

pipeline, metadatos = cargar_modelo()

st.title("🎓 Predicción del nivel de desempeño - Pruebas Saber 11")
st.caption("Proyecto integrador CRISP-DM · Histórico de resultados Saber 11, Medellín (Medata)")

st.markdown(
    f"**Modelo en producción:** `{metadatos['modelo_nombre']}` · "
    f"Predice 4 categorías: **Bajo, Medio, Alto, Superior** "
    f"(construidas como cuartiles del puntaje global histórico)."
)

with st.expander("ℹ️ ¿Cómo funciona esta app?"):
    st.write(
        "Completa los datos institucionales de un colegio (o de un periodo hipotético) y la app "
        "usará el modelo entrenado para predecir el nivel de desempeño más probable, junto con la "
        "probabilidad de cada una de las 4 categorías. El modelo solo usa variables de contexto "
        "institucional (sector, comuna, tamaño, participación y comparación con promedios de la "
        "ciudad/comuna), nunca los puntajes por área."
    )

st.header("1. Datos del colegio")
col1, col2 = st.columns(2)
with col1:
    comuna = st.selectbox("Comuna / corregimiento", metadatos["comunas_validas"])
    sector = st.selectbox("Tipo de colegio (sector)", metadatos["sectores_validos"])
    anio = st.number_input("Año", min_value=2014, max_value=2035, value=2018, step=1)
    semestre = st.selectbox("Semestre de presentación", [1, 2], index=1)
with col2:
    matriculados = st.number_input("Estudiantes matriculados en grado 11", min_value=1, value=60)
    registros = st.number_input("Estudiantes inscritos a la prueba", min_value=1, value=58)
    presentes = st.number_input("Estudiantes que presentaron la prueba", min_value=1, value=55)
    publicados = st.number_input("Estudiantes con resultado publicado", min_value=1, value=55)

st.header("2. Puntos de comparación (contexto)")
st.caption("Si no los conoces, deja los valores sugeridos (promedios históricos de Medellín).")
col3, col4 = st.columns(2)
with col3:
    punt_global_med = st.number_input("Promedio de la ciudad ese periodo",
                                       value=round(metadatos["rangos_numericos"]["punt_global_med"]["mediana"], 1))
with col4:
    punt_global_comu = st.number_input("Promedio de la comuna ese periodo",
                                        value=round(metadatos["rangos_numericos"]["punt_global_comu"]["mediana"], 1))

if st.button("🔮 Predecir nivel de desempeño", type="primary"):
    if registros > matriculados:
        st.warning("Ojo: inscritos > matriculados. Puede pasar (estudiantes de otras cohortes), pero revisa los datos.")
    if presentes > registros or publicados > presentes:
        st.error("Los datos no son consistentes: se espera matriculados ≥ registros ≥ presentes ≥ publicados.")
    else:
        entrada = pd.DataFrame([{
            "matriculados": matriculados, "registros": registros, "presentes": presentes, "publicados": publicados,
            "punt_global_med": punt_global_med, "punt_global_comu": punt_global_comu,
            "tasa_presentismo": min(presentes / registros, 1.0),
            "tasa_publicacion": min(publicados / presentes, 1.0),
            "brecha_vs_ciudad": punt_global_comu - punt_global_med,
            "anio": anio, "semestre": semestre,
            "comuna": comuna, "prestacion_servicio": sector,
        }])

        prediccion = pipeline.predict(entrada)[0]
        probabilidades = pipeline.predict_proba(entrada)[0]
        clases = pipeline.classes_

        st.header("3. Resultado")
        colores = {"Bajo": "🔴", "Medio": "🟠", "Alto": "🟡", "Superior": "🟢"}
        st.subheader(f"{colores.get(prediccion, '')} Nivel predicho: **{prediccion}**")

        df_proba = pd.DataFrame({"Nivel": clases, "Probabilidad": probabilidades}).sort_values("Probabilidad", ascending=False)
        st.bar_chart(df_proba.set_index("Nivel"))
        st.dataframe(df_proba.style.format({"Probabilidad": "{:.1%}"}), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Trabajo integrador construido siguiendo la metodología CRISP-DM. "
    "Dataset: Histórico de resultados Pruebas Saber 11 - Alcaldía de Medellín (Medata)."
)
