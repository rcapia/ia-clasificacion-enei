import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# -------------------------INSTRUCCIONES DE EJECUCIÓN--------------------------
# En la terminal de VS Code con tu entorno virtual activo, ejecuta:
# streamlit run app_streamlit_susalud.py
# -----------------------------------------------------------------------------

# 01 --------------------------Cargar el Modelo de Susalud---------------------
# El archivo .joblib creado en el paso anterior de 1.22 MB
clf = load('modelo_susalud_rf.joblib')

# 02---------------- Variables globales para los campos del formulario-----------------------
total_ciruj_may = 0
total_ciruj_men = 0
horas_programadas = 0.0
horas_efectivas = 0.0
horas_act_quirurj = 0.0
ciruj_suspend = 0

# 03 Reseteo------------- Flag para rastrear errores---------------------------------
error_flag = False

def reset_inputs():
    global total_ciruj_may, total_ciruj_men, horas_programadas, horas_efectivas, horas_act_quirurj, ciruj_suspend, error_flag
    total_ciruj_may = 0
    total_ciruj_men = 0
    horas_programadas = 0.0
    horas_efectivas = 0.0
    horas_act_quirurj = 0.0
    ciruj_suspend = 0
    error_flag = False

# Inicializar variables por defecto
if 'initialized' not in st.session_state:
    reset_inputs()
    st.session_state['initialized'] = True

# ------------------------Título e Interfaz Web------------------------------------------
st.set_page_config(page_title="Predicción SUSALUD", page_icon="🏥", layout="centered")

st.title("🏥 Sistema Predictivo de Programación Quirúrgica (SUSALUD)")
st.markdown("Esta aplicación web utiliza un modelo **Random Forest** masivo para clasificar el estado del indicador **`DE_PROGRAMC`** según la actividad del establecimiento de salud.")
st.markdown("Implemented by Reynaldo Capia Capia")
st.markdown("---")

# ------------------------------------ Formulario en dos columnas------------------------------------
with st.form("susalud_form"):
    col1, col2 = st.columns(2)

    # Input fields en la primera columna (Cirugías y suspensiones)
    with col1:
        st.subheader("📊 Volúmenes de Cirugía")
        total_ciruj_may = st.number_input("**Total Cirugías Mayores**", min_value=0, value=int(total_ciruj_may), step=1)
        total_ciruj_men = st.number_input("**Total Cirugías Menores**", min_value=0, value=int(total_ciruj_men), step=1)
        ciruj_suspend = st.number_input("**Cirugías Suspendidas**", min_value=0, value=int(ciruj_suspend), step=1)
        
    # Input fields en la segunda columna (Tiempos y horas de quirófano)
    with col2:
        st.subheader("⏳ Tiempos Quirúrgicos (Horas)")
        horas_programadas = st.number_input("**Horas Programadas**", min_value=0.0, value=float(horas_programadas), step=0.5)
        horas_efectivas = st.number_input("**Horas Efectivas**", min_value=0.0, value=float(horas_efectivas), step=0.5)
        horas_act_quirurj = st.number_input("**Horas Actividad Quirúrgica**", min_value=0.0, value=float(horas_act_quirurj), step=0.5)

    st.markdown("---")
    # Botón de Predecir dentro del formulario
    predict_button = st.form_submit_button("🔮 Ejecutar Predicción Inteligente")

# ----------------------------------------- Lógica de Predicción -------------------------------------------------
if predict_button:
    # 1. Crear DataFrame con los nombres exactos de las columnas numéricas que el modelo espera
    data = {
        'TOTAL_CIRUJ_MAY': [total_ciruj_may],
        'TOTAL_CIRUJ_MEN': [total_ciruj_men],
        'HORAS_PROGRAMADAS': [horas_programadas],
        'HORAS_EFECTIVAS': [horas_efectivas],
        'HORAS_ACT_QUIRURJ': [horas_act_quirurj],
        'CIRUJ_SUSPEND': [ciruj_suspend]
    }
    df_nuevo = pd.DataFrame(data)

    try:
        # 2. Realizar predicción y cálculo de probabilidades probabilísticas
        probabilities_classes = clf.predict_proba(df_nuevo)[0]
        class_predicted = clf.predict(df_nuevo)[0] # Nos dará 1 o 2 directamente

        # 3. Formatear la respuesta según las clases reales detectadas (Clase 1 y Clase 2)
        if class_predicted == 1:
            outcome = "Estado de Programación: CLASE 1"
            probability_final = probabilities_classes[0] # Probabilidad de la clase 1 (Índice 0)
            style_result = 'background-color: #d4edda; color: #155724; border-color: #c3e6cb; padding: 15px; border-radius: 5px; font-weight: bold; font-size: larger;'
        else:
            outcome = "Estado de Programación: CLASE 2"
            probability_final = probabilities_classes[1] # Probabilidad de la clase 2 (Índice 1)
            style_result = 'background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; padding: 15px; border-radius: 5px; font-weight: bold; font-size: larger;'

        # 4. Mostrar resultado renderizado con HTML personalizado
        st.markdown("### 🎯 Resultado del Análisis:")
        result_html = f"<div style='{style_result}'> {outcome} <br><span style='font-size: small; font-weight: normal;'>Confianza del modelo: {round(float(probability_final) * 100, 2)}%</span></div>"
        st.markdown(result_html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error interno al procesar la predicción: {e}")

# --------------------------- Botón de Limpieza (Resetear fuera del formulario) -------------------------------------
if st.button("🔄 Limpiar Formulario"):
    reset_inputs()
    st.rerun()

#Activar entorno
#.\.venv2\Scripts\activate

#Ejecutar Streamlit
#streamlit run app_streamlit_susalud.py
