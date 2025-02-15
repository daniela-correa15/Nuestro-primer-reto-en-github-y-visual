from fastapi import FastAPI, HTTPException  # FastAPI nos ayuda a crear la API y HTTPException maneja errores.
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import nltk  # Funciona en Python 3.11.9
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Configuración de NLTK
#nltk.data.path.append('C:\\Users\\USER\\AppData\\Roaming\\nltk_data')
#nltk.download('punkt')  # Paquete para dividir frases en palabras
#nltk.download('wordnet')  # Paquete para obtener sinónimos de palabras en inglés
#nltk.download()
# Función para cargar el dataset
def cargar_dataset():
    columnas = ['id','Area', 'H_hor', 'A_sun(w)', 'H_sun(w)', 'A_sun(s)', 'H_sun(s)']
    try:
        df = pd.read_csv("Dataset/SOLAR2.csv", usecols=columnas)
       # df.columns = df.columns.str.strip()
        df.columns = ['id','area', 'hora_solar', 'altura_solar_batio', 'hora_solar_batio', 'altura_solar_seg', 'hora_solar_seg']
        return df.fillna('').to_dict(orient='records')
    except FileNotFoundError:
        print("Error: Archivo no encontrado.")
        return []
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return []

# Cargamos el dataset al iniciar la API para evitar leer el archivo repetidamente
dataset_list = cargar_dataset()
##print(dataset_list)
# Función para obtener sinónimos de una palabra
def obtener_sinonimos(palabra):
    return {lemma.name().lower() for syn in wordnet.synsets(palabra) for lemma in syn.lemmas()}

# Inicialización de FastAPI
app = FastAPI(title="Viabilidad Cálculo de Paneles Solares", version="1.0.0")

# Ruta de inicio
@app.get("/", tags=['Home'])
def home():
    return HTMLResponse("<h1>Bienvenido a la API de cálculo de paneles solares</h1>")
 
# Ruta para obtener el dataset completo
@app.get("/Dataset", tags=['Dataset'])
def obtener_dataset():
    if dataset_list:
        return dataset_list
    raise HTTPException(status_code=500, detail="No hay datos en el dataset")

# Ruta para obtener un registro por ID
@app.get("/Dataset/{id}", tags=['Dataset'])
def obtener_por_id(id: int):
    resultado = next((m for m in dataset_list if m['id'] == id), None)
    if resultado:
        return resultado
    raise HTTPException(status_code=404, detail="Información no encontrada")

# Ruta del chatbot
@app.get("/chatbot", tags=['Chatbot'])
def chatbot(hora_solar: float):
   # query_words = word_tokenize(query.lower())
   # query_words = word_tokenize(query) #dividr texto en palabras individuales
   # sinonimos = {word for q in query_words for word in obtener_sinonimos(q)} | set(query_words)
   # resultados = [m for m in dataset_list if any(s in str(m['hora_solar']) for s in sinonimos)]
    resultado = next((m for m in dataset_list if m['hora_solar'] >= hora_solar), None)
    return JSONResponse(content={
        "respuesta": "Aquí tienes la información que buscabas" if resultado else "No se encontró la información que buscabas",
        "paneles": resultado
    })

# Ruta para buscar información por altitud específica
@app.get("/Dataset/by_area/", tags=['Dataset'])
def obtener_por_area(area: float):
    resultados = [m for m in dataset_list if m['area'] == area]
    if resultados:
        return resultados
    raise HTTPException(status_code=404, detail="No se encontraron datos para la altitud especificada")
 