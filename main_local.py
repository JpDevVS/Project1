import base64
from PIL import Image
import pandas as pd
import os
from io import BytesIO
import streamlit as st
import re
import xlsxwriter
import requests
import json
import pytesseract
from PIL import Image
import io

# -----------------------------------------------------------
# CONFIGURACIÓN STREAMLIT
# -----------------------------------------------------------

st.set_page_config(
    page_title="Tarjetas de Negocios - Active Re",
    page_icon="icono.png",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.image("AC_Grande.1.1.png")
st.subheader("🔍 Extracción de Datos de Tarjetas de Presentación")

uploaded_files = st.file_uploader(
    "Paso 1: Sube imágenes de tarjetas",
    label_visibility="collapsed",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# -----------------------------------------------------------
# FUNCIÓN LOCAL: ENVÍO AL MODELO gpt-oss:20b
# -----------------------------------------------------------

def extract_structured_data_local(question):
    data = {
        "model": "gpt-oss:20b",
        "prompt": f"{question}",
        "stream": False
    }

    url = 'http://10.0.0.51:11434/api/generate'
    #url = 'http://127.0.0.1:11434/api/generate'
    response = requests.post(url, json=data)

    if response.status_code == 200:
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                except ValueError as error:
                    return {"Error": str(error)}
    else:
        return {"Error": f"HTTP status {response.status_code}"}

    return json_data.get('response', "")


# -----------------------------------------------------------
# OCR PARA EXTRAER TEXTO DE LA IMAGEN
# -----------------------------------------------------------

def ocr_extract_text(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image, lang='eng')
    return text


# -----------------------------------------------------------
# NUEVA FUNCIÓN PARA OBTENER TEXTO ESTRUCTURADO (OCR + GPT LOCAL)
# -----------------------------------------------------------

def extract_text_from_image(image_bytes):
    """Extrae texto usando OCR y luego estructura con gpt-oss:20b"""

    raw_text = ocr_extract_text(image_bytes)

    if not raw_text.strip():
        return "No se pudo extraer texto con OCR."

    question = f"""
Estructura estos datos extraídos de una tarjeta de presentación:

Texto detectado:
{raw_text}

Devuelve exclusivamente este formato EXACTO:

Nombre:
Cargo:
Empresa:
Teléfono:
Celular:
Website:
Correo:
Dirección:

Si algún campo no aparece, deja el valor vacío.
"""

    response_text = extract_structured_data_local(question)

    st.write("📝 Respuesta del modelo:")
    st.write(response_text)
    st.write("----------------------------------")

    return response_text


# -----------------------------------------------------------
# LIMPIEZA DE TEXTO
# -----------------------------------------------------------

def limpia_asteriscos(texto):
    return texto.replace("*", "").replace("* ", "")

def limpia_giones(texto):
    return texto.replace("-", "").replace("- ", "")


# -----------------------------------------------------------
# PARSEADOR PARA CONVERTIR TEXTO EN CAMPOS
# -----------------------------------------------------------

def parse_card_data(text):
    lines = text.split("\n")

    data = {
        "Nombre": "",
        "Cargo": "",
        "Empresa": "",
        "Teléfono": "",
        "Celular": "",
        "Website": "",
        "Correo": "",
        "Dirección": ""
    }

    for line in lines:
        clean = line.strip()

        if clean.lower().startswith("nombre:"):
            data["Nombre"] = clean.replace("Nombre:", "").strip()

        elif clean.lower().startswith("cargo:"):
            data["Cargo"] = clean.replace("Cargo:", "").strip()

        elif clean.lower().startswith("empresa:"):
            data["Empresa"] = clean.replace("Empresa:", "").strip()

        elif clean.lower().startswith("teléfono:"):
            data["Teléfono"] = clean.replace("Teléfono:", "").strip()

        elif clean.lower().startswith("celular:"):
            data["Celular"] = clean.replace("Celular:", "").strip()

        elif clean.lower().startswith("website:"):
            data["Website"] = clean.replace("Website:", "").strip()

        elif clean.lower().startswith("correo:"):
            data["Correo"] = clean.replace("Correo:", "").strip()

        elif clean.lower().startswith("dirección:"):
            data["Dirección"] = clean.replace("Dirección:", "").strip()

    return data


# -----------------------------------------------------------
# COMBINA DATOS DE AMBAS CARAS
# -----------------------------------------------------------

def merge_card_data(front_data, back_data):
    merged_data = {}

    for key in front_data.keys():
        if key == "Archivo":
            merged_data[key] = front_data[key]
            continue

        front_value = front_data.get(key, "")
        back_value = back_data.get(key, "")

        merged_data[key] = front_value if front_value else back_value

    return merged_data


# -----------------------------------------------------------
# PROCESAMIENTO DE IMÁGENES
# -----------------------------------------------------------

excel_data = BytesIO()

if uploaded_files:
    extracted_data = []
    progress_bar = st.progress(0)

    i = 0
    while i < len(uploaded_files):
        current_image = Image.open(uploaded_files[i])
        st.image(current_image, caption=f"Procesando: {uploaded_files[i].name}", use_container_width=True)

        is_double_sided = False
        if i + 1 < len(uploaded_files):
            is_double_sided = st.checkbox(
                f"Esta tarjeta tiene doble cara (frente: {uploaded_files[i].name}, reverso: {uploaded_files[i+1].name})",
                key=f"double_sided_{i}"
            )

        with st.spinner(f"Analizando imagen {i+1} de {len(uploaded_files)}..."):

            front_image_bytes = uploaded_files[i].getvalue()
            front_text = extract_text_from_image(front_image_bytes)
            front_data = parse_card_data(front_text)
            front_data["Archivo"] = uploaded_files[i].name

            if is_double_sided:
                i += 1
                back_image = Image.open(uploaded_files[i])
                st.image(back_image, caption=f"Reverso: {uploaded_files[i].name}", use_container_width=True)

                back_image_bytes = uploaded_files[i].getvalue()
                back_text = extract_text_from_image(back_image_bytes)
                back_data = parse_card_data(back_text)

                merged_data = merge_card_data(front_data, back_data)
                extracted_data.append(merged_data)

            else:
                extracted_data.append(front_data)

        progress_bar.progress((i+1) / len(uploaded_files))
        i += 1

    st.html("<h3>Paso 2: Revisa los datos y verifica si están correctos</h3>")

    edited_df = pd.DataFrame(extracted_data)
    edited_df = st.data_editor(edited_df)

    if st.button("🟩 Generar Archivo Excel"):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Tarjetas')

        excel_data = buffer.getvalue()

        st.download_button(
            label="📥 Descargar Excel",
            data=excel_data,
            file_name="tarjetas_procesadas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("✅ Archivo generado correctamente!")
