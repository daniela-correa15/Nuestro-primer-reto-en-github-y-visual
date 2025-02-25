from fastapi.responses import HTMLResponse  # FastAPI nos ayuda a crear la API y HTTPException maneja errores.
from fastapi.responses import JSONResponse
import pandas as pd
import math
import nltk  # Funciona en Python 3.11.9
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from fastapi import FastAPI, HTTPException, Query
import numpy as np 
#no lee archivos locales el fastapi
# Configuración de NLTK
#nltk.data.path.append('C:\\Users\\USER\\AppData\\Roaming\\nltk_data')
#nltk.download('punkt')  # Paquete para dividir frases en palabras
#nltk.download('wordnet')  # Paquete para obtener sinónimos de palabras en inglés
#nltk.download()
# Función para cargar el dataset
def cargar_dataset():
    df = pd.read_csv("Dataset/SOLAR3.csv", delimiter=";", quotechar='"', on_bad_lines="skip") [['id','municipio','energiaMensual','horasDia']]
    return df.fillna('')


#municipios_disponibles = dataset['municipio'].unique().tolist()
    
def cargar_dataset2():
    df1 = pd.read_csv("Dataset/PANELES_CSV.csv", delimiter=";", quotechar='"', on_bad_lines="skip")[['id','panel_solar','potencia','tension','valor']]
    return df1.fillna('') #.to_dict(orient="records")

dataset = cargar_dataset()
municipios_disponibles =  dataset['municipio'].unique().tolist()#list(set(registro['municipio'] for registro in dataset))
dataset1 = cargar_dataset2()
paneles =  dataset1['panel_solar'].unique().tolist()
# Cargamos el dataset al iniciar la API para evitar leer el archivo repetidamente
#dataset_list = cargar_dataset()
##print(dataset_list)

# Función para obtener sinónimos de una palabra
#def obtener_sinonimos(palabra):
  #  return {lemma.name().lower() for syn in wordnet.synsets(palabra) for lemma in syn.lemmas()}

# Inicialización de FastAPI
app = FastAPI(title="Cálculo de Paneles Solares", version="1.0.0")

# Ruta de inicio
@app.get("/", tags=['Home'])
def home():
    return HTMLResponse("<h1>Bienvenido a la API de cálculo de paneles solares</h1>")
    #return TemplateResponse("index.html", {"request": request})
dataset_list = cargar_dataset()
# Ruta para obtener el dataset completo
@app.get("/Dataset/SOLAR3.csv", tags=['SOLAR3'])
def obtener_dataset():
    df = cargar_dataset()
   #if df.empty:
       # raise HTTPException(status_code=500, detail="No hay datos en el dataset")
   #return dataset_list
    # Convertir todos los valores NumPy a tipos nativos de Python
    dataset_dict = df.to_dict(orient="records")
    dataset_convertido = [{k: (int(v) if isinstance(v, (np.int64, np.int32)) else float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in row.items()} for row in dataset_dict]
    
    return dataset_convertido

# Ruta para obtener un registro por ID
@app.get("/Dataset/{id}", tags=['Dataset'])
def obtener_por_id(id: int):
    df = cargar_dataset()
    # Asegurar que dataset_list sea una lista de diccionarios
    dataset_list = df.to_dict(orient="records")  
    
    # Buscar el registro con el ID solicitado
    resultado = next((m for m in dataset_list if m['id'] == id), None)
    
    if resultado is None:  # Verifica si no se encontró el ID
        raise HTTPException(status_code=404, detail="Información no encontrada")
    
    return resultado
# Ruta del chatbot
#@app.get("/chatbot", tags=['Chatbot'])
#def chatbot(n_horas_dia: float, tolerance: float = 0.1):
  #  resultados = [m for m in dataset_list if abs(m['n_horas_dia'] - n_horas_dia) <= tolerance]
   # return JSONResponse(content={
   #     "respuesta": "Aquí tienes la información que buscabas" if resultados else "No se encontró la información que buscabas",
    #    "paneles": resultados
   # })

# Ruta para buscar información por municipio
@app.get("/calculo_paneles", tags=['Cálculo'])
def calcular_paneles(
    consumo: float = Query(..., description="Consumo de energía en kWh"),
    municipio: str = Query(..., description="Seleccione un municipio", enum=municipios_disponibles),
    panel: str = Query(..., description="Seleccione un tipo de panel", enum=paneles)
):
    # Obtener las horas de radiación para el municipio
    datos_municipio = dataset[dataset['municipio'] == municipio]
    if datos_municipio.empty:
        raise HTTPException(status_code=404, detail="Municipio no encontrado")
    
    n_horas_dia = float(datos_municipio.iloc[0]['horasDia'])
    
    # Obtener la potencia del panel solar
    datos_panel = dataset1[dataset1['panel_solar'] == panel]
    if datos_panel.empty:
        raise HTTPException(status_code=404, detail="Panel solar no encontrado")
    
    potw = float(datos_panel.iloc[0]['potencia'])
    
    # Calcular el número de paneles solares
    numero_paneles = ((consumo * 1000) / 30) / (n_horas_dia * potw)
    numero_paneles_redondeado = math.ceil(numero_paneles)
    
    return JSONResponse(content={
        "municipio": municipio,
        "consumo_mensual_kWh": consumo,
        "horas_radiacion_dia": n_horas_dia,
        "panel_solar": panel,
        "potencia_panel_w": potw,
        "paneles_necesarios": numero_paneles_redondeado
    })


