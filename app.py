import streamlit as st
import PyPDF2
import pytesseract
from PIL import Image
from googletrans import Translator
import io
import tempfile
import os

# Configuraciones específicas para OCR multiidioma
TESSERACT_CONFIGS = {
    'mixed': {
        'lang': 'ara+chi_sim+chi_tra+deu+hin+heb+spa+eng',
        'config': '--oem 3 --psm 6',
        'description': 'Detección automática de múltiples idiomas'
    }
}

# Mapeo de códigos de idioma detectados a nombres
LANGUAGE_NAMES = {
    'ar': 'Árabe 🇸🇦',
    'zh': 'Chino 🇨🇳',
    'zh-cn': 'Chino Simplificado 🇨🇳',
    'zh-tw': 'Chino Tradicional 🇹🇼',
    'de': 'Alemán 🇩🇪',
    'hi': 'Hindi 🇮🇳',
    'he': 'Hebreo 🇮🇱',
    'es': 'Español 🇪🇸',
    'en': 'Inglés 🇺🇸',
    'auto': 'Detección Automática'
}

# Configuración de la página
st.set_page_config(
    page_title="Traductor de Documentos",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Inicializar el traductor
@st.cache_resource
def init_translator():
    return Translator()


translator = init_translator()

# Lista de idiomas disponibles
IDIOMAS = {
    'Español': 'es',
    'Inglés': 'en',
    'Francés': 'fr',
    'Alemán': 'de',
    'Italiano': 'it',
    'Portugués': 'pt',
    'Ruso': 'ru',
    'Japonés': 'ja',
    'Chino (Simplificado)': 'zh-cn',
    'Chino (Tradicional)': 'zh-tw',
    'Coreano': 'ko',
    'Árabe': 'ar',
    'Hindi': 'hi',
    'Holandés': 'nl',
    'Sueco': 'sv',
    'Noruego': 'no',
    'Danés': 'da',
    'Finlandés': 'fi',
    'Polaco': 'pl',
    'Checo': 'cs',
    'Húngaro': 'hu',
    'Griego': 'el',
    'Hebreo': 'he',
    'Turco': 'tr',
    'Tailandés': 'th',
    'Vietnamita': 'vi'
}


def extract_text_from_pdf(pdf_file):
    """Extrae texto de un archivo PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error al leer el PDF: {str(e)}")
        return None


def extract_text_from_image(image_file):
    """Extrae texto de una imagen usando OCR con soporte multiidioma"""
    try:
        image = Image.open(image_file)
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Configuración de idiomas para OCR (árabe, chino, alemán, hindi, hebreo, español, inglés)
        # ara=árabe, chi_sim=chino simplificado, chi_tra=chino tradicional, deu=alemán,
        # hin=hindi, heb=hebreo, spa=español, eng=inglés
        ocr_languages = 'ara+chi_sim+chi_tra+deu+hin+heb+spa+eng'

        # Configuración personalizada de Tesseract para mejor reconocimiento
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'

        # Intentar primero con todos los idiomas
        try:
            text = pytesseract.image_to_string(image, lang=ocr_languages, config=custom_config)
        except:
            # Si falla, intentar con configuración básica
            text = pytesseract.image_to_string(image, lang=ocr_languages)

        return text.strip()
    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        return None


def translate_text(text, target_language):
    """Traduce el texto al idioma objetivo"""
    try:
        # Dividir el texto en chunks más pequeños si es muy largo
        max_chunk_size = 4000
        if len(text) <= max_chunk_size:
            result = translator.translate(text, dest=target_language)
            return result.text, result.src
        else:
            # Dividir en chunks
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            translated_chunks = []
            source_lang = None

            for chunk in chunks:
                result = translator.translate(chunk, dest=target_language)
                translated_chunks.append(result.text)
                if source_lang is None:
                    source_lang = result.src

            return '\n'.join(translated_chunks), source_lang
    except Exception as e:
        st.error(f"Error en la traducción: {str(e)}")
        return None, None


def main():
    st.title("🌐 Traductor de Documentos")
    st.markdown("### Traduce documentos PDF e imágenes a cualquier idioma")

    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")

        # Selector de idioma de destino
        target_language_name = st.selectbox(
            "Selecciona el idioma de destino:",
            list(IDIOMAS.keys()),
            index=0
        )
        target_language_code = IDIOMAS[target_language_name]

        st.markdown("---")
        st.markdown("### 📝 Instrucciones")
        st.markdown("""
        1. Sube un archivo PDF o imagen
        2. Selecciona el idioma de destino
        3. Haz clic en 'Traducir'
        4. Descarga el resultado

        **Idiomas de origen soportados:**
        🇸🇦 Árabe • 🇨🇳 Chino • 🇩🇪 Alemán • 🇮🇳 Hindi • 🇮🇱 Hebreo
        """)

        st.markdown("---")
        st.markdown("### 📋 Formatos e Idiomas soportados")
        st.markdown("""
        **PDFs:** .pdf

        **Imágenes:** .jpg, .jpeg, .png, .bmp, .tiff

        **Idiomas de origen detectados:**
        - 🇸🇦 Árabe
        - 🇨🇳 Chino (Simplificado/Tradicional)  
        - 🇩🇪 Alemán
        - 🇮🇳 Hindi
        - 🇮🇱 Hebreo
        - 🇪🇸 Español
        - 🇺🇸 Inglés
        """)

    # Área principal
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📄 Subir Documento")

        uploaded_file = st.file_uploader(
            "Arrastra y suelta tu archivo aquí o haz clic para seleccionar:",
            type=['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Formatos soportados: PDF, JPG, JPEG, PNG, BMP, TIFF"
        )

        if uploaded_file is not None:
            file_details = {
                "Nombre": uploaded_file.name,
                "Tipo": uploaded_file.type,
                "Tamaño": f"{uploaded_file.size / 1024:.2f} KB"
            }

            st.success("✅ Archivo cargado exitosamente!")

            with st.expander("📊 Detalles del archivo"):
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")

            # Vista previa para imágenes
            if uploaded_file.type.startswith('image/'):
                st.subheader("🖼️ Vista previa")
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen cargada", use_column_width=True)

    with col2:
        st.header("🔄 Resultado de la Traducción")

        if uploaded_file is not None:
            if st.button("🚀 Traducir Documento", type="primary", use_container_width=True):
                with st.spinner("Procesando documento..."):
                    # Extraer texto según el tipo de archivo
                    if uploaded_file.type == "application/pdf":
                        st.info("📖 Extrayendo texto del PDF...")
                        extracted_text = extract_text_from_pdf(uploaded_file)
                    else:
                        st.info("🔍 Extrayendo texto de la imagen con OCR multiidioma...")
                        st.caption("Detectando: Árabe, Chino, Alemán, Hindi, Hebreo, Español, Inglés")
                        extracted_text = extract_text_from_image(uploaded_file)

                    if extracted_text:
                        st.success("✅ Texto extraído exitosamente!")

                        # Mostrar texto extraído
                        with st.expander("📝 Texto extraído (original)"):
                            st.text_area("", extracted_text, height=200, disabled=True)

                        # Traducir texto
                        st.info(f"🌐 Traduciendo a {target_language_name}...")
                        translated_text, source_lang = translate_text(extracted_text, target_language_code)

                        if translated_text:
                            st.success("✅ Traducción completada!")

                            # Detectar idioma original con nombre mejorado
                            source_lang_name = LANGUAGE_NAMES.get(source_lang, f"Idioma ({source_lang})")

                            st.info(f"**Idioma detectado:** {source_lang_name}")
                            st.info(f"**Idioma de destino:** {target_language_name}")

                            # Mostrar consejos específicos si es necesario
                            if source_lang in ['ar', 'zh', 'de', 'hi', 'he']:
                                with st.expander("💡 Consejos para este idioma"):
                                    if source_lang == 'ar':
                                        st.markdown("""
                                        **Texto en Árabe:**
                                        - Se lee de derecha a izquierda
                                        - Mejor resultado con fuentes claras
                                        - Funciona bien con texto impreso
                                        """)
                                    elif source_lang == 'zh':
                                        st.markdown("""
                                        **Texto en Chino:**
                                        - Soporta caracteres simplificados y tradicionales
                                        - Mejor con imágenes de alta resolución
                                        - Funciona con texto horizontal y vertical
                                        """)
                                    elif source_lang == 'de':
                                        st.markdown("""
                                        **Texto en Alemán:**
                                        - Reconoce caracteres especiales (ä, ö, ü, ß)
                                        - Excelente con palabras compuestas
                                        - Mejor con texto impreso
                                        """)
                                    elif source_lang == 'hi':
                                        st.markdown("""
                                        **Texto en Hindi:**
                                        - Escritura Devanagari
                                        - Requiere buena calidad de imagen
                                        - Mejor con fuentes estándar
                                        """)
                                    elif source_lang == 'he':
                                        st.markdown("""
                                        **Texto en Hebreo:**
                                        - Se lee de derecha a izquierda
                                        - Mejor sin puntos vocálicos
                                        - Requiere caracteres bien definidos
                                        """)

                            # Mostrar traducción
                            st.subheader("📋 Texto Traducido")
                            st.text_area("", translated_text, height=300, key="translated_text")

                            # Botón de descarga
                            st.download_button(
                                label="📥 Descargar Traducción",
                                data=translated_text,
                                file_name=f"traduccion_{uploaded_file.name.split('.')[0]}_{target_language_name.lower()}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        else:
                            st.error("❌ Error en la traducción")
                    else:
                        st.error("❌ No se pudo extraer texto del documento")
        else:
            st.info("👆 Sube un documento para comenzar la traducción")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🚀 <strong>Traductor de Documentos</strong> | Desarrollado con Streamlit</p>
        <p><small>Soporta PDFs e imágenes con OCR • Traducción automática • Múltiples idiomas</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()