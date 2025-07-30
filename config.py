# Configuraciones específicas para OCR multiidioma
import streamlit as st

# Configuraciones de Tesseract por idioma
TESSERACT_CONFIGS = {
    'arabic': {
        'lang': 'ara',
        'config': '--oem 3 --psm 6 -c tessedit_char_whitelist=ابتثجحخدذرزسشصضطظعغفقكلمنهويىءآأؤإئة٠١٢٣٤٥٦٧٨٩',
        'description': 'Árabe - Texto de derecha a izquierda'
    },
    'chinese': {
        'lang': 'chi_sim+chi_tra',
        'config': '--oem 3 --psm 6',
        'description': 'Chino - Simplificado y Tradicional'
    },
    'german': {
        'lang': 'deu',
        'config': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß',
        'description': 'Alemán - Con caracteres especiales'
    },
    'hindi': {
        'lang': 'hin',
        'config': '--oem 3 --psm 6',
        'description': 'Hindi - Escritura Devanagari'
    },
    'hebrew': {
        'lang': 'heb',
        'config': '--oem 3 --psm 6 -c tessedit_char_whitelist=אבגדהוזחטיכלמנסעפצקרשת',
        'description': 'Hebreo - Texto de derecha a izquierda'
    },
    'mixed': {
        'lang': 'ara+chi_sim+chi_tra+deu+hin+heb+spa+eng',
        'config': '--oem 3 --psm 6',
        'description': 'Detección automática de múltiples idiomas'
    }
}

# Mapeo de códigos de idioma detectados a nombres
LANGUAGE_NAMES = {
    'ar': 'Árabe',
    'zh': 'Chino',
    'zh-cn': 'Chino Simplificado',
    'zh-tw': 'Chino Tradicional',
    'de': 'Alemán',
    'hi': 'Hindi',
    'he': 'Hebreo',
    'es': 'Español',
    'en': 'Inglés',
    'auto': 'Detección Automática'
}

# Consejos para mejorar el OCR por idioma
OCR_TIPS = {
    'ara': """
    💡 **Consejos para texto en Árabe:**
    - Asegúrate de que el texto esté bien iluminado
    - El texto árabe se lee de derecha a izquierda
    - Funciona mejor con fuentes claras y sin decoraciones
    """,
    'chi_sim': """
    💡 **Consejos para texto en Chino:**
    - Imágenes de alta resolución dan mejores resultados
    - Funciona tanto con caracteres simplificados como tradicionales
    - El texto puede ser horizontal o vertical
    """,
    'deu': """
    💡 **Consejos para texto en Alemán:**
    - Presta atención a los caracteres especiales (ä, ö, ü, ß)
    - Funciona mejor con texto impreso que manuscrito
    - Las palabras compuestas alemanas se reconocen bien
    """,
    'hin': """
    💡 **Consejos para texto en Hindi:**
    - El texto en escritura Devanagari requiere buena calidad de imagen
    - Funciona mejor con fuentes estándar
    - Los caracteres conjuntos pueden ser más difíciles de reconocer
    """,
    'heb': """
    💡 **Consejos para texto en Hebreo:**
    - El texto hebreo se lee de derecha a izquierda
    - Funciona mejor sin puntos vocálicos (nikud)
    - Asegúrate de que los caracteres estén bien definidos
    """
}

def get_language_display_name(lang_code):
    """Obtiene el nombre del idioma para mostrar"""
    return LANGUAGE_NAMES.get(lang_code, f"Idioma ({lang_code})")

def show_ocr_tips(detected_lang=None):
    """Muestra consejos específicos para el idioma detectado"""
    if detected_lang and detected_lang in OCR_TIPS:
        st.info(OCR_TIPS[detected_lang])
    else:
        st.info("""
        💡 **Consejos generales para mejor OCR:**
        - Usa imágenes de alta resolución y buena calidad
        - Asegúrate de que el texto esté bien iluminado
        - Evita imágenes borrosas o con mucho ruido
        - El texto debe tener buen contraste con el fondo
        """)