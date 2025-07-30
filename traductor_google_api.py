import glob

from deep_translator import GoogleTranslator
from llama_parse import LlamaParse
from dotenv import load_dotenv



# Cargar clave de API desde el archivo .env
load_dotenv()

def extraer_texto_pdf_llamaindex(file):
    texto_total = ""
    pdf_files = glob.glob(file)
    parser = LlamaParse(verbose=True)

    json_objs = []

    for pdf_file in pdf_files:
        # print(pdf_file)
        json_objs.extend(parser.get_json_result(pdf_file))

    #texto_total = json_objs[0]['pages'][0]['text']
    #print(texto_total)
    print(json_objs)
    return texto_total.strip()


#text = "Contrato de Compra y Venta"
#text = extraer_texto_pdf_llamaindex("data/credito.pdf")
text = extraer_texto_pdf_llamaindex("data/inv.pdf")

print(GoogleTranslator(source='auto', target='en').translate(text))


