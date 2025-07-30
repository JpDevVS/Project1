import json
import tempfile

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import base64
from PIL import Image
import pandas as pd
import os
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
import re
import xlsxwriter
import pdfplumber
from pyarrow import nulls
from pydantic_core.core_schema import none_schema

# Cargar clave de API desde el archivo .env
load_dotenv()

# Configuración de la interfaz con Streamlit
st.set_page_config(page_title="Control de Cambios a Excel", page_icon="📊", layout="centered")
# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .title {
        font-size: 2.5rem;
        color: #4F46E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .desc {
        text-align: center;
        font-size: 1.1rem;
        color: #333;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #4338CA;
    }
    </style>
""", unsafe_allow_html=True)

# st.subheader("🔍 Extracción de Datos de Facturas en Pdf")

# Encabezado con logo y título
# st.image("logo.png", width=120)
st.markdown('<div class="title">🔍 Extracción de Datos de Control de Cambios en Pdf 📄➡️📊</div>', unsafe_allow_html=True)

openai_api = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
if not openai_api:
    st.info("Por favor escriba su OpenAI API key para continuar.")
    st.stop()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = openai_api

# Permitir la subida de múltiples imágenes
uploaded_files = st.file_uploader("📎 Selecciona un archivo PDF de Control de Cambios", type=["pdf"], accept_multiple_files=True)


##**********************************************************************

def extraer_texto_pdf(file):
    texto_total = ""
    with pdfplumber.open(file) as pdf:
        for pagina in pdf.pages:
            texto_total += pagina.extract_text() + "\n"
    return texto_total.strip()


# Función para extraer texto de una imagen con GPT-4 Vision
def extract_text_from_image(texto_pdf):
    #llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY) # $2.50 - $1.25 - $10.00
    #llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY) # $0.15 - $0.075 - $0.60

    #llm = ChatOpenAI(model="gpt-4.5-preview", api_key=OPENAI_API_KEY) # $75.00 - $37.50 - $150.00 ******

    llm = ChatOpenAI(model="gpt-4.1", api_key=OPENAI_API_KEY) # $2.00 - $0.50 - $8.00
    #llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY) # $0.40 - $0.10 - $1.60
    #llm = ChatOpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY) # $0.10 - $0.025 - $0.40

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Eres un asistente que extrae información de un documento de Control de Cambios o Solicitud de Requerimientos y esta en formato PDF y finalmente traduce todo el texto al idioma Inglés."),
            (
                "human",
                [
                    {"type": "text", "text": "{input}"}
                ],
            ),
        ]
    )

    chain = prompt | llm

    question = f"""
        Extrae los siguientes datos estructurados del documento de Control de Cambios o Solicitud de Requerimientos. IMPORTANTE: Devuelve solo el JSON, sin explicación ni encabezado.
        :
        - Fecha de Solicitud: (Dale el formato de DD/MM/YYYY).
        - Aplicaciones: selecciona la aplicación y módulos aplicables al requerimiento con “x” o con un gancho de selección. 
        - Prioridad de la Solicitud: (Selecciona solo el campo con “x” o que tiene el ganchito en la parte izquierda).
        - En Información del Solicitante genera los siguientes campos:
            - Nombre
            - Cargo
            - Departamento
        - Tipo: (Selecciona solo el campo que tiene el ganchito en la parte izquierda).
        - Estructura del Reporte.

        Control de Cambios:
        {texto_pdf}

        Formato JSON esperado:
        {{
          "fecha_de_solicitud": "...",
          "aplicaciones": "...",
          "prioridad_de_la_solicitud": "...",
          "nombre": "...",
          "cargo": "...",
          "departamento": "...",
          "tipo": "...",
          "funcionalidad_solicitada": "..."
        }}
        """

    if question:
        response = chain.invoke({"input": question})

    # st.write("response --->>>", response)
    # st.write("response.content --->>>", response.content)

    # st.write(response.content)
    # st.write("------------------------------------")
    # st.write("")
    return response.content


def extraer_json(texto):
    try:
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return None
    except:
        return None


# Función para estructurar datos extraídos
def parse_total_factura(datos_json, file_name):
    # st.write(datos_json)
    datos_json = extraer_json(datos_json)
    # st.write(datos_json)
    if not datos_json:
        raise ValueError("❌ No se pudo extraer JSON de la respuesta de OpenAI.")

    datos = json.loads(datos_json)
    data = {"Fecha de Solicitud": "", "Aplicaciones": "", "Prioridad de la Solicitud": "", "Nombre": "", "Cargo": "", "Departamento": "", "Tipo": "", "Funcionalidad Solicitada": ""}

    data["Fecha de Solicitud"] = datos.get("fecha_de_solicitud", "")
    data["Aplicaciones"] = datos.get("aplicaciones", "")
    data["Prioridad de la Solicitud"] = datos.get("prioridad_de_la_solicitud", "")
    data["Nombre"] = datos.get("nombre", "")
    data["Cargo"] = datos.get("cargo", "")
    data["Departamento"] = datos.get("departamento", "")
    data["Tipo"] = datos.get("tipo", "")
    data["Funcionalidad Solicitada"] = datos.get("funcionalidad_solicitada", "")

    # st.write(productos)
    return data


##**********************************************************************

def parse_card_data1(datos_json):
    datos_json = extraer_json(datos_json)

    if not datos_json:
        raise ValueError("❌ No se pudo extraer JSON de la respuesta de OpenAI.")

    datos = json.loads(datos_json)

    df_general = pd.DataFrame({
        "Campo": ["Proveedor", "Fecha", "Número de factura", "Cliente", "SubTotal", "Impuestos", "Total"],
        "Valor": [
            datos.get("proveedor", ""),
            datos.get("fecha", ""),
            datos.get("numero_factura", ""),
            datos.get("cliente"),
            datos.get("subtotal", ""),
            datos.get("impuestos", ""),
            datos.get("total", "")
        ]
    })

    # df_productos = pd.DataFrame(datos["productos"])
    return df_general


##**********************************************************************


# Procesar imágenes subidas
if uploaded_files:
    extracted_data = []
    extracted_data_productos = []
    progress_bar = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):

        with st.spinner(f"Analizando imagen {i + 1} de {len(uploaded_files)}..."):

            st.info(f"🔍 Extrayendo texto del PDF --> {uploaded_file.name}")
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
            text = extraer_texto_pdf(tmp_path)

            st.info(f"🧠 Enviando a OpenAI para estructuración --> {uploaded_file.name}")
            json_structurado = extract_text_from_image(text)

            # Extraer y organizar los datos
            factura_total = parse_total_factura(json_structurado, uploaded_file.name)
            factura_total["Archivo Pdf"] = uploaded_file.name
            extracted_data.append(factura_total)

        # Actualizar barra de progreso
        progress_bar.progress((i + 1) / len(uploaded_files))

    # Mostrar resultados en una tabla editable
    st.write("📋 **Revisa y edita los datos si es necesario:**")
    edited_df = pd.DataFrame(extracted_data)
    edited_df = st.data_editor(edited_df)

    edited_products_df = pd.DataFrame(extracted_data_productos)
    edited_products_df = st.data_editor(edited_products_df)

    st.success("✅ Análisis completado con éxito!")

    # Guardar datos en Excel -
    if st.button("📥 Guardar en Excel"):
        # st.info("ℹ️ Generando Excel...")

        # Crear un buffer para guardar el archivo Excel
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Facturas_Totales')
            edited_products_df.to_excel(writer, index=False, sheet_name='Facturas_Productos')

        # Obtener los bytes del archivo
        excel_data = buffer.getvalue()

        # Crear botón de descarga
        st.download_button(
            label="📂 Descargar Excel",
            data=excel_data,
            file_name="facturas-procesadas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("✅ Archivo Excel generado con éxito!")


