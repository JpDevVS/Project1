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
st.set_page_config(page_title="Extracción de Datos de PDF", page_icon="📊", layout="centered")

st.image("AC_Grande.1.1.png")

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

st.markdown('<div class="title">Extracción de Datos de PDFs con IA</div>', unsafe_allow_html=True)

openai_api = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
if not openai_api:
    st.info("Por favor escriba su OpenAI API key para continuar.")
    st.stop()

OPENAI_API_KEY = openai_api

# Permitir la subida de múltiples imágenes
uploaded_files = st.file_uploader("📎 Selecciona archivos PDF para analizar", type=["pdf"], accept_multiple_files=True)

# Opción para personalizar la extracción
document_type = st.selectbox(
    "Tipo de documento a analizar",
    ["Estado de Cuenta", "Factura", "Control de Cambios", "Otro", "Automático (detectar)"]
)

# Campo de texto opcional para dar contexto adicional
additional_context = st.text_area(
    "Contexto adicional (opcional)",
    placeholder="Añade información adicional que pueda ayudar a identificar campos importantes, por ejemplo: 'Es un documento financiero', 'Busca nombres de clientes y montos'..."
)


##**********************************************************************

def extraer_texto_pdf(file):
    texto_total = ""
    with pdfplumber.open(file) as pdf:
        for pagina in pdf.pages:
            texto_total += pagina.extract_text() + "\n"
    return texto_total.strip()


# Función para extraer texto de un PDF con GPT-4
def extract_structured_data(texto_pdf, doc_type, context=""):
    #llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY) # $2.50 - $1.25 - $10.00
    #llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY) # $0.15 - $0.075 - $0.60

    #llm = ChatOpenAI(model="gpt-4.5-preview", api_key=OPENAI_API_KEY) # $75.00 - $37.50 - $150.00 ******

    #llm = ChatOpenAI(model="gpt-4.1", api_key=OPENAI_API_KEY) # $2.00 - $0.50 - $8.00
    llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY) # $0.40 - $0.10 - $1.60
    #llm = ChatOpenAI(model="gpt-4.1-nano", api_key=OPENAI_API_KEY) # $0.10 - $0.025 - $0.40

    # Instrucciones basadas en el tipo de documento
    type_instructions = {
        "Control de Cambios": "un documento de Control de Cambios que puede contener información sobre solicitudes, aplicaciones, prioridades, etc.",
        "Factura": "una factura que puede contener información sobre productos, precios, impuestos, cliente, proveedor, etc.",
        "Estado de Cuenta": "un estado de cuenta de una Compañía Reaseguradora, que puede contener información sobre transacciones, saldos, fechas, etc.",
        "Automático (detectar)": "un documento que necesita ser analizado para identificar su tipo y los campos de información relevantes",
        "Otro": "un documento que contiene información estructurada que debe ser extraída"
    }

    instruction = type_instructions.get(doc_type, type_instructions["Estado de Cuenta"])
    #instruction = "un estado de cuenta de una Compañía Reaseguradora, que puede contener información sobre transacciones, saldos, fechas, etc."

    context_prompt = f"Contexto adicional proporcionado: {context}" if context else ""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"""Eres un analista experto que extrae información estructurada de documentos PDF. 
            Estás analizando {instruction}.
            IMPORTANTE: Debes mantener el nombre de los campos EXACTAMENTE como aparecen en el documento original, 
            sin traducirlos. Por ejemplo, si un campo aparece como "Invoice Number", debes nombrarlo así y no como "Número de Factura"."""),
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
        Analiza el siguiente texto extraído de un PDF y devuelve la información en formato JSON estructurado:

        {texto_pdf}

        Instrucciones importantes:
        1. Identifica automáticamente todos los campos relevantes y sus valores
        2. MANTÉN LOS NOMBRES DE CAMPOS EN EL IDIOMA ORIGINAL del documento (no los traduzcas)
        3. Usa nombres de campos descriptivos y significativos como aparecen en el documento original
        4. No te limites a campos predefinidos, extrae todos los que encuentres
        5. Si detectas tablas o listas, estructúralas adecuadamente
        6. Devuelve solo el JSON, sin explicación ni encabezado

        {context_prompt}

        Formato JSON esperado:
        {{
          "document_type": "el tipo de documento que detectaste",
          "identified_fields": {{
            "campo1": "valor1",
            "campo2": "valor2",
            ...
          }},
          "tables": [
            {{
              "table_name": "nombre descriptivo si lo hay",
              "data": [
                {{
                  "columna1": "valor1",
                  "columna2": "valor2",
                  ...
                }},
                ...
              ]
            }}
          ]
        }}
        """

    response = chain.invoke({"input": question})
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


# Función para estructurar datos extraídos de forma dinámica
def parse_extracted_data(datos_json, file_name):
    datos_json = extraer_json(datos_json)

    if not datos_json:
        raise ValueError("❌ No se pudo extraer JSON de la respuesta de OpenAI.")

    try:
        datos = json.loads(datos_json)

        # Información general sobre el documento
        info_general = {
            "File_Name": file_name,
            "Document_Type": datos.get("document_type", "Not specified")
        }

        # Campos identificados (dinámicos) - mantener nombres originales
        campos = datos.get("identified_fields", {})

        # Combinar información general con campos identificados
        todos_campos = {**info_general, **campos}

        # Extraer tablas si existen - manteniendo nombres de columnas originales
        tablas = datos.get("tables", [])
        tablas_procesadas = []

        for tabla in tablas:
            nombre_tabla = tabla.get("table_name", f"Table from {file_name}")
            datos_tabla = tabla.get("data", [])

            for fila in datos_tabla:
                fila["File_Name"] = file_name
                fila["Table_Name"] = nombre_tabla
                tablas_procesadas.append(fila)

        return todos_campos, tablas_procesadas

    except json.JSONDecodeError:
        st.error(f"Error al decodificar JSON para {file_name}")
        return {"File_Name": file_name, "Error": "Invalid JSON format"}, []


##**********************************************************************

# Inicializar las variables de sesión
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = False
    st.session_state.processed_fields = []
    st.session_state.processed_tables = []
    st.session_state.edited_df_fields = pd.DataFrame()
    st.session_state.edited_df_tables = pd.DataFrame()

# Procesar archivos subidos
if uploaded_files and not st.session_state.processed_data:
    extracted_fields = []
    extracted_tables = []
    progress_bar = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f"Analizando documento {i + 1} de {len(uploaded_files)}..."):
            st.info(f"🔍 Extrayendo texto del PDF --> {uploaded_file.name}")
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

            text = extraer_texto_pdf(tmp_path)

            st.info(f"🧠 Enviando a OpenAI para estructuración --> {uploaded_file.name}")
            json_estructurado = extract_structured_data(text, document_type, additional_context)

            # Extraer y organizar los datos de forma dinámica
            campos_documento, tablas_documento = parse_extracted_data(json_estructurado, uploaded_file.name)

            extracted_fields.append(campos_documento)

            # Agregar tablas si existen
            for tabla in tablas_documento:
                nombre_tabla = tabla.get("nombre_tabla", f"Tabla de {uploaded_file.name}")
                datos_tabla = tabla.get("datos", [])

                for fila in datos_tabla:
                    fila["Archivo PDF"] = uploaded_file.name
                    fila["Nombre Tabla"] = nombre_tabla
                    extracted_tables.append(fila)

        # Actualizar barra de progreso
        progress_bar.progress((i + 1) / len(uploaded_files))

    # Guardar los datos procesados en la sesión
    st.session_state.processed_data = True
    st.session_state.processed_fields = extracted_fields
    st.session_state.processed_tables = extracted_tables

# Botón para reiniciar el proceso
if st.session_state.processed_data:
    if st.button("🔄 Procesar nuevos documentos"):
        st.session_state.processed_data = False
        st.rerun()

# Mostrar resultados si hay datos procesados
if st.session_state.processed_data:
    # Mostrar resultados en tablas editables
    st.write("📋 **Campos principales identificados:**")
    if st.session_state.processed_fields:
        # Crear un DataFrame con todas las columnas posibles
        all_keys = set()
        for item in st.session_state.processed_fields:
            all_keys.update(item.keys())

        # Crear un DataFrame con valores consistentes
        df_fields = pd.DataFrame(columns=list(all_keys))
        for item in st.session_state.processed_fields:
            df_row = pd.Series({k: item.get(k, "") for k in all_keys})
            df_fields = pd.concat([df_fields, df_row.to_frame().T], ignore_index=True)

        st.session_state.edited_df_fields = st.data_editor(df_fields, key="fields_editor")
    else:
        st.warning("No se encontraron campos principales en los documentos.")
        st.session_state.edited_df_fields = pd.DataFrame()

    # Mostrar tablas si existen
    if st.session_state.processed_tables:
        st.write("📊 **Tablas identificadas en los documentos:**")
        # Similar al procedimiento anterior para manejar columnas dinámicas
        all_table_keys = set()
        for item in st.session_state.processed_tables:
            all_table_keys.update(item.keys())

        df_tables = pd.DataFrame(columns=list(all_table_keys))
        for item in st.session_state.processed_tables:
            df_row = pd.Series({k: item.get(k, "") for k in all_table_keys})
            df_tables = pd.concat([df_tables, df_row.to_frame().T], ignore_index=True)

        st.session_state.edited_df_tables = st.data_editor(df_tables, key="tables_editor")
    else:
        st.info("No se identificaron tablas en los documentos.")
        st.session_state.edited_df_tables = pd.DataFrame()

    st.success("✅ Análisis completado con éxito!")

    # Guardar datos en Excel
    if st.button("📥 Guardar en Excel"):
        # Crear un buffer para guardar el archivo Excel
        buffer = BytesIO()

        # Crear un libro de Excel con formato
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Definir formatos
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4F46E5',
                'font_color': 'white',
                'border': 1
            })

            cell_format = workbook.add_format({
                'border': 1
            })

            # Hoja 1: Todos los campos en una sola hoja
            st.session_state.edited_df_fields.to_excel(writer, sheet_name='Datos_Extraídos', index=False)
            worksheet = writer.sheets['Datos_Extraídos']

            # Aplicar formato a encabezados
            for col_num, value in enumerate(st.session_state.edited_df_fields.columns.values):
                worksheet.write(0, col_num, value, header_format)
                # Ajustar ancho de columna
                worksheet.set_column(col_num, col_num, max(len(str(value)) + 2, 15))

            # Si hay tablas, agregarlas como hojas adicionales
            if not st.session_state.edited_df_tables.empty:
                # Creamos una hoja separada para cada documento con tablas
                documentos = st.session_state.edited_df_tables['Archivo PDF'].unique()

                for i, doc in enumerate(documentos):
                    doc_tables = st.session_state.edited_df_tables[
                        st.session_state.edited_df_tables['Archivo PDF'] == doc]

                    # Limitamos a 31 caracteres para nombre de hoja en Excel
                    nombre_doc = doc.replace('.pdf', '')
                    sheet_name = f"Doc{i + 1}_{nombre_doc[:20]}"
                    sheet_name = sheet_name.replace(' ', '_').replace('.', '')[:31]

                    doc_tables.to_excel(writer, index=False, sheet_name=sheet_name)
                    worksheet = writer.sheets[sheet_name]

                    # Aplicar formato a encabezados
                    for col_num, value in enumerate(doc_tables.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                        worksheet.set_column(col_num, col_num, max(len(str(value)) + 2, 15))

        # Obtener los bytes del archivo
        excel_data = buffer.getvalue()

        # Crear botón de descarga
        st.download_button(
            label="📂 Descargar Excel",
            data=excel_data,
            file_name="documentos_procesados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("✅ Archivo Excel generado con éxito!")