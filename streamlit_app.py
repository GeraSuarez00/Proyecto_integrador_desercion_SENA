"""
Aplicación web para predecir la deserción en fichas de Formación del SENA.
Proyecto integrador - Aprendizaje de Máquinas - UPB
Integrantes: Geraldine Suárez Solano, John Jairo Sevilla Rodríguez
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# 1) Carga de los activos entrenados (modelo + objetos de preparación de datos)
#    @st.cache_resource evita que se vuelvan a cargar en cada clic del usuario.
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_activos():
    modelo = joblib.load("model.joblib")
    encoder = joblib.load("encoder.joblib")
    scaler = joblib.load("scaler.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    columnas_modelo = joblib.load("columnas_modelo.joblib")
    return modelo, encoder, scaler, label_encoder, columnas_modelo

modelo, encoder, scaler, label_encoder, columnas_modelo = cargar_activos()

COLUMNAS_CATEGORICAS = ['NOMBRE_REGIONAL', 'NOMBRE_CENTRO', 'VERSION_PROGRAMA',
                         'NIVEL_FORMACION', 'MODALIDAD_FORMACION', 'MES_INICIO_FICHA']
COLUMNAS_NUMERICAS = ['TOTAL_APRENDICES_MATRICULADOS', 'DURACION_DIAS']

# Listas de opciones para los menús desplegables (tomadas de las categorías vistas en el entrenamiento)
OPCIONES_REGIONAL = sorted([c.replace('NOMBRE_REGIONAL_', '') for c in encoder.get_feature_names_out(COLUMNAS_CATEGORICAS) if c.startswith('NOMBRE_REGIONAL_')])
OPCIONES_CENTRO = sorted([c.replace('NOMBRE_CENTRO_', '') for c in encoder.get_feature_names_out(COLUMNAS_CATEGORICAS) if c.startswith('NOMBRE_CENTRO_')])
OPCIONES_VERSION = sorted([c.replace('VERSION_PROGRAMA_', '') for c in encoder.get_feature_names_out(COLUMNAS_CATEGORICAS) if c.startswith('VERSION_PROGRAMA_')])
OPCIONES_NIVEL = sorted([c.replace('NIVEL_FORMACION_', '') for c in encoder.get_feature_names_out(COLUMNAS_CATEGORICAS) if c.startswith('NIVEL_FORMACION_')])
OPCIONES_MODALIDAD = sorted([c.replace('MODALIDAD_FORMACION_', '') for c in encoder.get_feature_names_out(COLUMNAS_CATEGORICAS) if c.startswith('MODALIDAD_FORMACION_')])
OPCIONES_MES = sorted([c.replace('MES_INICIO_FICHA_', '') for c in encoder.get_feature_names_out(COLUMNAS_CATEGORICAS) if c.startswith('MES_INICIO_FICHA_')], key=lambda x: int(x))

# ---------------------------------------------------------------------------
# 2) Interfaz de usuario
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Predicción de deserción SENA", page_icon="🎓", layout="centered")

st.title("Modelo de clasificación basado en Machine Learning para la predicción de la deserción en fichas de Formación Profesional Integral del SENA")
st.markdown(
    "Esta herramienta estima, a partir de las características que se conocen "
    "**al momento de crear una ficha**, si es probable que presente **deserción** "
    "durante el proceso formativo. Está pensada como apoyo para priorizar el "
    "acompañamiento académico, no como una decisión definitiva."
)

st.subheader("Datos de la ficha de formación")

col1, col2 = st.columns(2)
with col1:
    regional = st.selectbox("Regional", OPCIONES_REGIONAL)
    centro = st.selectbox("Centro de formación", OPCIONES_CENTRO)
    version = st.selectbox("Versión del programa", OPCIONES_VERSION)
with col2:
    nivel = st.selectbox("Nivel de formación", OPCIONES_NIVEL)
    modalidad = st.selectbox("Modalidad de formación", OPCIONES_MODALIDAD)
    mes_inicio = st.selectbox("Mes de inicio de la ficha", OPCIONES_MES)

total_matriculados = st.number_input("Número de aprendices matriculados", min_value=1, max_value=500, value=30)
duracion_dias = st.number_input("Duración estimada de la formación (en días)", min_value=1, max_value=2000, value=180)

# ---------------------------------------------------------------------------
# 3) Predicción
# ---------------------------------------------------------------------------
if st.button("Predecir"):
    entrada = pd.DataFrame([{
        'NOMBRE_REGIONAL': regional,
        'NOMBRE_CENTRO': centro,
        'VERSION_PROGRAMA': version,
        'NIVEL_FORMACION': nivel,
        'MODALIDAD_FORMACION': modalidad,
        'MES_INICIO_FICHA': mes_inicio,
        'TOTAL_APRENDICES_MATRICULADOS': total_matriculados,
        'DURACION_DIAS': duracion_dias,
    }])

    # Se aplica EXACTAMENTE el mismo proceso de preparación que se usó al entrenar el modelo
    entrada_codificada = encoder.transform(entrada[COLUMNAS_CATEGORICAS])
    entrada_codificada_df = pd.DataFrame(entrada_codificada, columns=encoder.get_feature_names_out(COLUMNAS_CATEGORICAS))
    entrada_final = pd.concat([entrada_codificada_df, entrada[COLUMNAS_NUMERICAS]], axis=1)[columnas_modelo]
    entrada_final[COLUMNAS_NUMERICAS] = scaler.transform(entrada[COLUMNAS_NUMERICAS])

    prediccion = modelo.predict(entrada_final)[0]
    probabilidad = modelo.predict_proba(entrada_final)[0][1]
    etiqueta = label_encoder.inverse_transform([prediccion])[0]

    st.subheader("Resultado")
    if etiqueta == 1:
        st.error(f"⚠️ Riesgo de deserción — probabilidad estimada: {probabilidad:.1%}")
        st.markdown("Se recomienda priorizar el acompañamiento y seguimiento académico de esta ficha.")
    else:
        st.success(f"✅ Baja probabilidad de deserción — probabilidad estimada: {probabilidad:.1%}")

st.markdown("---")
st.caption(
    "Modelo: Voting Classifier (Soft Voting). "
    "Proyecto integrador de Aprendizaje de Máquinas - UPB. "
)
