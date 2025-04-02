
import streamlit as st
import pandas

# Configuración de la interfaz con Streamlit
st.set_page_config(page_title="OCR de Tarjetas", layout="centered")
st.subheader("🔍 Extracción de Datos de Tarjetas de Presentación")

# Permitir la subida de múltiples imágeneses
uploaded_files = st.file_uploader("Sube imágenes de tarjetas", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# local test

##**********************************************************************

