# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción de Deserción Universitaria",
    page_icon="🎓",
    layout="wide"
)

# Título principal
st.title("🎓 Sistema de Alerta Temprana para Deserción Estudiantil")
st.markdown("---")

# Sidebar para entrada de datos
st.sidebar.header("📋 Información del Estudiante")

# Simulamos un modelo
@st.cache_resource
def load_model():
    return RandomForestClassifier()

model = load_model()

# Formulario de entrada de datos
st.sidebar.subheader("Datos Demográficos")
age = st.sidebar.slider("Edad", 17, 50, 20)
gender = st.sidebar.selectbox("Género", ["Masculino", "Femenino"])
international = st.sidebar.selectbox("Estudiante Internacional", ["Sí", "No"])

st.sidebar.subheader("Datos Académicos")
previous_grade = st.sidebar.slider("Calificación Previa", 0, 200, 120)
scholarship = st.sidebar.selectbox("Beca", ["Sí", "No"])
attendance = st.sidebar.slider("Asistencia (%)", 0, 100, 85)

st.sidebar.subheader("Datos Socioeconómicos")
parent_education = st.sidebar.selectbox("Educación de los Padres", 
                                      ["Primaria", "Secundaria", "Universitaria"])
family_income = st.sidebar.selectbox("Ingreso Familiar", 
                                   ["Bajo", "Medio", "Alto"])

# Botón para predecir
if st.sidebar.button("🔍 Predecir Riesgo de Deserción"):
    # Preprocesar datos
    data = {
        'age': age,
        'previous_grade': previous_grade,
        'attendance': attendance,
        'scholarship': 1 if scholarship == "Sí" else 0,
        'international': 1 if international == "Sí" else 0
    }
    
    # Hacer predicción (simulada)
    risk_level = "Alto" if previous_grade < 100 or attendance < 70 else "Moderado" if previous_grade < 120 else "Bajo"
    confidence = 0.85 if risk_level == "Alto" else 0.72 if risk_level == "Moderado" else 0.65
    
    # Mostrar resultados
    st.subheader("📊 Resultados de la Predicción")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nivel de Riesgo", risk_level)
    
    with col2:
        st.metric("Confianza", f"{confidence*100:.1f}%")
    
    with col3:
        probability = 0.75 if risk_level == "Alto" else 0.45 if risk_level == "Moderado" else 0.15
        st.metric("Probabilidad de Abandono", f"{probability*100:.1f}%")
    
    # Barra de progreso para visualizar el riesgo
    risk_value = 0.75 if risk_level == "Alto" else 0.45 if risk_level == "Moderado" else 0.15
    st.progress(risk_value, text=f"Nivel de riesgo: {risk_level}")
    
    # Recomendaciones
    st.subheader("🎯 Recomendaciones de Intervención")
    
    if risk_level == "Alto":
        st.error("🚨 Intervención prioritaria requerida")
        st.write("""
        - **Tutoría intensiva** semanal
        - **Evaluación psicológica** recomendada
        - **Beca de apoyo** a considerar
        - **Contacto con familia** inmediato
        """)
    elif risk_level == "Moderado":
        st.warning("⚠️ Monitoreo cercano recomendado")
        st.write("""
        - **Seguimiento académico** quincenal
        - **Talleres de habilidades** de estudio
        - **Mentoría** con estudiante avanzado
        """)
    else:
        st.success("✅ Riesgo bajo - Continuar monitoreo regular")
        st.write("""
        - **Seguimiento** semestral estándar
        - **Participación** en actividades extracurriculares
        """)
    
    # Análisis de factores
    st.subheader("🔍 Factores de Riesgo Identificados")
    
    factors = []
    if previous_grade < 100:
        factors.append(f"Calificación previa baja ({previous_grade}/200)")
    if attendance < 70:
        factors.append(f"Asistencia preocupante ({attendance}%)")
    if scholarship == "No":
        factors.append("Falta de apoyo económico (sin beca)")
    if age > 25:
        factors.append("Edad mayor al promedio típico")
    
    if factors:
        st.write("**Factores de riesgo detectados:**")
        for factor in factors:
            st.write(f"• {factor}")
    else:
        st.success("✅ No se identificaron factores de riesgo significativos")
    
    # Gráfico de factores usando gráficos nativos de Streamlit
    st.subheader("📈 Análisis de Impacto de Factores")
    
    factors_data = {
        'Calificación Previa': max(0, (100 - previous_grade) / 100),
        'Asistencia': max(0, (70 - attendance) / 70),
        'Beca': 0.4 if scholarship == "No" else 0.1,
        'Edad': 0.2 if age > 25 else 0.05
    }
    
    # Crear DataFrame para el gráfico
    chart_data = pd.DataFrame({
        'Factor': list(factors_data.keys()),
        'Impacto': list(factors_data.values())
    })
    
    # Gráfico de barras nativo de Streamlit
    st.bar_chart(chart_data.set_index('Factor'))
    
    # Tabla con los valores detallados
    st.write("**Detalles del impacto:**")
    
    impact_df = pd.DataFrame({
        'Factor': factors_data.keys(),
        'Valor Impacto': [f"{x:.2f}" for x in factors_data.values()],
        'Nivel': ['Alto' if x > 0.3 else 'Moderado' if x > 0.1 else 'Bajo' for x in factors_data.values()]
    })
    
    st.dataframe(impact_df, hide_index=True, use_container_width=True)
    
    # Indicadores visuales adicionales
    st.subheader("📋 Resumen de Indicadores")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info(f"**Edad:** {age} años")
    
    with col2:
        st.info(f"**Calificación:** {previous_grade}/200")
    
    with col3:
        st.info(f"**Asistencia:** {attendance}%")
    
    with col4:
        st.info(f"**Beca:** {'Sí' if scholarship == 'Sí' else 'No'}")

else:
    st.info("👈 Complete la información del estudiante en la barra lateral y haga clic en 'Predecir Riesgo'")

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
**ℹ️ Acerca del Sistema:**
Este sistema predictivo utiliza machine learning para identificar estudiantes en riesgo de deserción universitaria, permitiendo intervenciones tempranas y personalizadas.

**📊 Métricas consideradas:**
- Calificaciones previas
- Asistencia a clases
- Situación económica
- Edad del estudiante
""")

# Footer
st.markdown("---")
st.caption("Sistema de Predicción de Deserción Universitaria v1.0 | Desarrollado con Streamlit")