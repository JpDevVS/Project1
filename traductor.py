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

# Intentar importar PyMuPDF de manera segura
try:
    import fitz  # PyMuPDF para manejar PDFs
except ImportError:
    try:
        import PyMuPDF as fitz
    except ImportError:
        fitz = None
import tempfile
import shutil

# Configuración de la interfaz con Streamlit
st.set_page_config(
    page_title="Extractor de Texto de Imágenes",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🔍 Extractor de Texto de Imágenes en PDF")
st.subheader("Extrae texto de imágenes en documentos PDF o imágenes individuales")

openai_api = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
if not openai_api:
    st.info("Por favor escriba su OpenAI API key y haga click aquí para continuar.")
    st.stop()

# Configurar clave de API
OPENAI_API_KEY = openai_api

# Modo de extracción
extraction_mode = st.radio(
    "Seleccione el modo de extracción:",
    ["Tarjetas de Negocios", "Texto General"]
)

# Permitir la subida de múltiples archivos
st.html("<h3>Paso 1: Sube archivos (PDF o imágenes)</h3>")

# Mostrar advertencia de PDF si es necesario
if fitz is None:
    st.warning(
        "NOTA: La funcionalidad de PDF no está disponible porque PyMuPDF no está instalado. Solo se procesarán imágenes.")

uploaded_files = st.file_uploader(
    "Sube PDFs o imágenes",
    label_visibility="collapsed",
    type=["jpg", "png", "jpeg", "pdf"] if fitz is not None else ["jpg", "png", "jpeg"],
    accept_multiple_files=True
)


def encode_image(image_bytes):
    """Codifica una imagen en base64"""
    return base64.b64encode(image_bytes).decode('utf-8')


def extract_text_from_image(image_bytes, mode="general"):
    """Extrae texto de una imagen usando GPT-4o"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

        # Configuración de sistema según el modo
        if mode == "business_card":
            system_message = "Eres un asistente que extrae información de tarjetas de presentación."
            question = """
                        Extrae la información de la tarjeta de negocio y traducela al idioma Inglés.
                        Incluye: Nombre, Cargo, Empresa, Teléfono, Celular, Website, Correo, Dirección.
                        """
        else:  # modo texto general
            system_message = "Eres un asistente que extrae texto de imágenes con alta precisión."
            question = """
                        Extrae todo el texto visible en esta imagen.
                        Preserva el formato lo mejor posible, incluyendo párrafos, columnas y estructura.
                        Traduce el texto al idioma Inglés.
                        """
        image = encode_image(image_bytes)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    [
                        {"type": "text", "text": "{input}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                                "detail": "high",  # Usamos detalle alto para mejor OCR
                            },
                        },
                    ],
                ),
            ]
        )

        chain = prompt | llm
        #image = encode_image(image_bytes)
        response = chain.invoke({"input": question, "image": image})

        #if st.checkbox("Mostrar texto extraído sin procesar", value=False, key="1"):
        st.write(response.content)
        st.write("------------------------------------")

        return response.content
    except Exception as e:
        st.error(f"Error al extraer texto: {str(e)}")
        return "Error al procesar la imagen. Por favor, verifica tu API key de OpenAI y la conexión a internet."


def extract_images_from_pdf(pdf_bytes):
    """Extrae imágenes de un PDF y las devuelve como bytes"""
    images = []
    image_info = []

    # Verificar si PyMuPDF está disponible
    if fitz is None:
        st.error("PyMuPDF no está instalado correctamente. No se pueden procesar PDFs.")
        st.info("Por favor, instala PyMuPDF con: pip install PyMuPDF==1.23.8")
        return images, image_info

    # Crear un archivo temporal para guardar el PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf_path = temp_pdf.name

    try:
        # Abrir el PDF con PyMuPDF
        pdf_document = fitz.open(temp_pdf_path)

        # Crear directorio temporal para guardar las imágenes
        temp_dir = tempfile.mkdtemp()

        # Extraer imágenes de cada página
        for page_num, page in enumerate(pdf_document):
            # Si hay pocos o ningún elemento xref, renderizar toda la página como imagen
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolución para mejor calidad
            image_path = os.path.join(temp_dir, f"page_{page_num}.png")
            pix.save(image_path)

            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
                images.append(img_bytes)
                image_info.append(f"Página {page_num + 1} (página completa)")

            # También extraer imágenes individuales si existen
            image_list = page.get_images(full=True)
            if image_list:
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        if "image" in base_image:
                            image_bytes = base_image["image"]

                            # Guardar para referencia
                            image_path = os.path.join(temp_dir, f"page_{page_num}_img_{img_index}.png")
                            with open(image_path, "wb") as img_file:
                                img_file.write(image_bytes)

                            # Solo añadir la imagen si es suficientemente grande
                            # (evitar iconos pequeños y elementos decorativos)
                            try:
                                img_temp = Image.open(BytesIO(image_bytes))
                                width, height = img_temp.size
                                if width > 100 and height > 100:  # Filtrar imágenes pequeñas
                                    images.append(image_bytes)
                                    image_info.append(f"Página {page_num + 1}, Imagen {img_index + 1}")
                            except Exception:
                                pass  # Ignorar si no podemos determinar el tamaño
                    except Exception as e:
                        st.warning(f"No se pudo extraer una imagen de la página {page_num + 1}: {e}")

        pdf_document.close()
    except Exception as e:
        st.error(f"Error al procesar el PDF: {e}")
    finally:
        # Limpiar archivos temporales
        try:
            os.unlink(temp_pdf_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignorar errores al limpiar

    return images, image_info


def limpia_texto(texto):
    """Elimina caracteres no deseados del texto"""
    return texto.replace("*", "").replace("-", "").strip()


def parse_card_data(text):
    """Estructura los datos extraídos de la tarjeta"""
    lines = text.split("\n")

    data = {"Nombre": "", "Cargo": "", "Empresa": "", "Teléfono": "", "Celular": "", "Website": "", "Correo": "",
            "Dirección": ""}

    for line in lines:
        line = line.strip()
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().lower()
                value = limpia_texto(parts[1].strip())

                # Mapeo de campos según el key detectado
                if any(k in key for k in ["nombre", "name"]):
                    data["Nombre"] = value
                elif any(k in key for k in ["cargo", "position", "title"]):
                    data["Cargo"] = value
                elif any(k in key for k in ["empresa", "company"]):
                    data["Empresa"] = value
                elif any(k in key for k in ["teléfono", "tel", "phone", "directo", "fijo"]):
                    data["Teléfono"] = value
                elif any(k in key for k in ["celular", "cel", "móvil", "movil"]):
                    data["Celular"] = value
                elif any(k in key for k in ["web", "website", "sitio", "www", "http"]):
                    data["Website"] = value
                elif any(k in key for k in ["correo", "email", "e-mail"]):
                    data["Correo"] = value
                elif any(k in key for k in ["dirección", "direccion", "address", "location"]):
                    data["Dirección"] = value
        elif "@" in line:
            data["Correo"] = line
        elif any(x in line.lower() for x in ["http", "www"]):
            data["Website"] = line

    return data


# Procesar archivos subidos
if uploaded_files:
    # Determinar si estamos en modo tarjetas o texto general
    mode = "business_card" if extraction_mode == "Tarjetas de Negocios" else "general"

    if mode == "business_card":
        extracted_data = []
    else:
        extracted_text_results = []

    progress_bar = st.progress(0)
    file_count = len(uploaded_files)

    for file_idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()

        st.subheader(f"Procesando: {file_name}")

        # Si es un PDF, extraer imágenes primero
        if file_extension == ".pdf":
            # Verificar si PyMuPDF está disponible
            if fitz is None:
                st.error("No se puede procesar PDF. PyMuPDF no está instalado correctamente.")
                st.info("Instala PyMuPDF con: pip install PyMuPDF==1.23.8")
                st.info("O usa imágenes directamente en lugar de PDF.")
                continue

            pdf_bytes = uploaded_file.getvalue()
            images, image_info = extract_images_from_pdf(pdf_bytes)

            if not images:
                st.warning(f"No se encontraron imágenes en el PDF {file_name}")
                continue

            st.info(f"Se encontraron {len(images)} imágenes en el PDF {file_name}")

            # Procesar cada imagen del PDF
            for img_idx, (image_bytes, img_info) in enumerate(zip(images, image_info)):
                with st.spinner(f"Analizando {img_info} de {file_name}..."):
                    try:
                        # Mostrar la imagen
                        try:
                            img = Image.open(BytesIO(image_bytes))
                            st.image(img, caption=f"{img_info}", use_container_width=True)
                        except Exception as e:
                            st.warning(f"No se pudo mostrar la imagen: {e}")

                        # Extraer texto
                        extracted_text = extract_text_from_image(image_bytes, mode)

                        if mode == "business_card":
                            card_data = parse_card_data(extracted_text)
                            card_data["Archivo"] = f"{file_name} - {img_info}"
                            extracted_data.append(card_data)
                        else:
                            extracted_text_results.append({
                                "Archivo": file_name,
                                "Ubicación": img_info,
                                "Texto Extraído": extracted_text
                            })
                    except Exception as e:
                        st.error(f"Error al procesar imagen {img_idx + 1} del PDF: {e}")
        else:
            # Es una imagen directa
            with st.spinner(f"Analizando imagen {file_idx + 1} de {file_count}..."):
                try:
                    # Mostrar la imagen
                    img = Image.open(uploaded_file)
                    st.image(img, caption=file_name, use_container_width=True)

                    # Extraer texto
                    image_bytes = uploaded_file.getvalue()
                    extracted_text = extract_text_from_image(image_bytes, mode)

                    if mode == "business_card":
                        card_data = parse_card_data(extracted_text)
                        card_data["Archivo"] = file_name
                        extracted_data.append(card_data)
                    else:
                        extracted_text_results.append({
                            "Archivo": file_name,
                            "Ubicación": "Imagen completa",
                            "Texto Extraído": extracted_text
                        })
                except Exception as e:
                    st.error(f"Error al procesar la imagen {file_name}: {e}")

        # Actualizar barra de progreso
        progress_percentage = (file_idx + 1) / file_count
        progress_bar.progress(min(progress_percentage, 1.0))

    # Mostrar resultados según el modo
    st.html("<h3>Paso 2: Revisa los datos extraídos</h3>")

    if mode == "business_card" and extracted_data:
        # Mostrar datos de tarjetas en una tabla editable
        edited_df = pd.DataFrame(extracted_data)
        edited_df = st.data_editor(edited_df)

        # Guardar datos en Excel
        if st.button("🟩 Generar Archivo Excel"):
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False, sheet_name='Tarjetas')

            excel_data = buffer.getvalue()
            st.download_button(
                label="📥 Descargar Archivo Excel",
                data=excel_data,
                file_name="tarjetas_procesadas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("✅ Archivo generado correctamente!")

    elif mode == "general" and extracted_text_results:
        # Mostrar resultados de texto extraído
        text_df = pd.DataFrame(extracted_text_results)
        st.dataframe(text_df)

        # Opción para descargar como CSV
        if st.button("🟩 Generar Archivo CSV"):
            csv = text_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name="texto_extraido.csv",
                mime="text/csv",
            )
            st.success("✅ Archivo generado correctamente!")

        # Opción para descargar como TXT
        if st.button("📄 Generar Archivo de Texto"):
            txt_content = ""
            for idx, row in text_df.iterrows():
                txt_content += f"=== {row['Archivo']} - {row['Ubicación']} ===\n\n"
                txt_content += f"{row['Texto Extraído']}\n\n"
                txt_content += "=" * 50 + "\n\n"

            st.download_button(
                label="📥 Descargar Texto",
                data=txt_content.encode('utf-8'),
                file_name="texto_extraido.txt",
                mime="text/plain",
            )
            st.success("✅ Archivo de texto generado correctamente!")
else:
    st.info("Sube uno o más archivos para comenzar la extracción de texto.")