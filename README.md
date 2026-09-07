# **Proyecto_integrador_desercion_SENA**
# **Modelo de clasificación basado en Machine Learning para la predicción de la deserción en fichas de Formación Profesional Integral del SENA**

Proyecto integrador — **Aprendizaje de Máquinas**, Universidad Pontificia Bolivariana (UPB)
Metodología: **CRISP-DM**

**Integrantes:** Geraldine Suárez Solano, John Jairo Sevilla Rodríguez

---

## **¿Qué hace este proyecto?**

Predice, a partir de las características que se conocen **al momento de crear una ficha de formación** (regional, centro, versión del programa, nivel, modalidad, número de aprendices matriculados, mes de inicio y duración), si esa ficha va a **presentar deserción** o **no la va a presentar**.

La idea es anticiparse: hoy el SENA solo sabe si una ficha tuvo deserción *después* de que ya ocurrió. Este modelo permite identificar fichas de riesgo *antes* de que empiecen, para poder priorizar el acompañamiento académico.

- **Tipo de problema:** Clasificación binaria (0 = No presenta deserción, 1 = Presenta deserción)
- **Modelo final:** Voting Classifier
- **Fuente de datos:** [Datos Abiertos Colombia — SENA](https://www.datos.gov.co/Trabajo/DESERCION-DE-LA-FORMACI-N-PROFESIONAL-INTEGRAL/u4ze-bi7k/about_data) (42.080 fichas)

## URL del despliegue
https://proyectointegradordesercionsena-lppappeb5wkyksbtyrhitbv.streamlit.app/

## Estructura del repositorio

```
├── streamlit_app.py                    # Aplicación web en Streamlit
├── requirements.txt          # Dependencias del proyecto
├── model.joblib               # Modelo entrenado (Gradient Boosting)
├── scaler.joblib               # MinMaxScaler ajustado con el train
├── encoder.joblib               # OneHotEncoder ajustado con todo el dataset
├── label_encoder.joblib          # LabelEncoder de la variable objetivo
├── columnas_modelo.joblib         # Orden exacto de columnas que espera el modelo
├── Proyecto_Desercion_SENA_CRISP_DM.ipynb   # Notebook completo (Google Colab)
└── README.md                  # Este archivo
```

## Cómo correr la app localmente

```bash
# 1. Clona el repositorio
git clone https://github.com/TU-USUARIO/Proyecto_integrador_desercion_SENA.git
cd prediccion-desercion-sena

# 2. Instala las dependencias
pip install -r requirements.txt

# 3. Ejecuta la app
streamlit run app.py
```

## Metodología CRISP-DM (Resumen)

Se desarrolló el proyecto siguiendo las seis fases de la metodología CRISP-DM:

1. **Entendimiento del negocio:** Se definió el problema predictivo de identificar fichas con posible presencia de deserción utilizando variables disponibles antes del proceso formativo, evitando fuga de información.

2. **Entendimiento de los datos:** Se exploró el dataset compuesto por 42.080 fichas de formación y se generó un reporte de perfilado con YData Profiling para analizar estructura, calidad y distribución de los datos.

3. **Preparación de los datos:** Se realizó limpieza, creación de la variable objetivo `PRESENTA_DESERCION`, generación de variables derivadas, selección de características, codificación de variables categóricas, escalamiento y tratamiento del desbalance de clases.

4. **Modelamiento:** Se implementaron modelos de clasificación supervisada como Regresión Logística, SVM, Red Neuronal, Random Forest, Gradient Boosting y Voting Classifier, utilizando validación cruzada y ajuste de hiperparámetros.

5. **Evaluación:** Los modelos fueron evaluados mediante F1-Score Macro como métrica principal, complementada con Accuracy, Precision, Recall, matriz de confusión y ROC-AUC.

6. **Despliegue:** El modelo seleccionado fue serializado mediante `joblib` y preparado para su integración en una aplicación interactiva desarrollada en Streamlit.
