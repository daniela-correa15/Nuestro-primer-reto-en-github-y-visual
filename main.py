
from fastapi import FastAPI, HTTPException #FastAPI nos ayuda a crear la api y httpexception maneja errores.
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd 
import nltk #me funciona en python 3.11.9
#import nltk.download('punkt') 
from nltk.tokenize import  word_tokenize
from nltk.corpus import wordnet

#nltk.data.path.append('C:\Users\USER\AppData\Local\Programs\Python\Python311\Lib\site-packages\nltk')
#descargamos las herramientas de nltk

nltk.download('punkt') #paquete para dividir frases en palabras
nltk.download('wordnet') #paquete para traducir las palabras en ingles

#funcion para cargar el dataset 
def imc(): #leemos el archivo que contiene informacion del dataset y selecionamos las columnas mas importantes
    df = pd.read_csv("Dataset/solarradiation.csv") [['show_id','date','MIN_WAVELENGTH','MAX_WAVELENGTH','INSTRUMENT_MODE','DATA_VERSION','IRRADIANCE','IRRADIANCE_UNCERTAINTY','QUALITY']]
    
    
#renombrar las columnas para facilitar
    df.columns = ['id','date','minima longitud de onda','maxima longitud de onda','modo de instrumento','version','radiacion','irradianza','calidad']

#llenamos los espacios vacios con texto vacio y convertimos los datos en lista de diccionarios
    return df.fillna('').to_dict('records')
# cargamos el dataset al iniciar la api para no leer el archivo cada vez que alguien pregunte por ellas
dataset_list = imc()

#funcin para encontrar sinonimos de una palabra
def get_synonyms(word):
    #usamos wordnet para obtener distintas palabras que significan la misma cosa
    return{lemma.name().lower() for syn in wordnet.synsets(word)for lemma in syn.lemmas()}

#creamos la aplicacion FastAPI que sera le motor de nuestra api
#esto inicializa la api con nombre y version
app = FastAPI(title="viabilidad calculo de paneles solares", version="1.0.0")
# ruta de inicio: cuando alguien entra a la api sin especificar nada vera un mensaje de bienvenida

@app.get("/", tags=['Home'])
def home():
    #cuando entremos en el navegador veremos el mensaje de Bienvenida
    return HTMLResponse("<h1> Bienvenido a la api de paneles solares</h1>")
#obteniendo la lista del database
#creamos una ruta para obtener todas las columnas del dataset

@app.get("/Dataset", tags=['Dataset'])
def get_dataset():
    #si hay data en el dataset lo devolvemos
    return dataset_list or HTTPException(status_code=500, detail="No hay datos en el dataset")

#ruta para obtener una columna del dataset
@app.get("/Dataset/{id}", tags=['Dataset'])
def get_dataset(id: str):
    #buscamos el id en el dataset y lo devolvemos
    return next((m for m in dataset_list if m['id'] == id), {"detalle": "informacion no encontrada"})

   #ruta del chatbot 
   
@app.get("/chatbot", tags=['Chatbot'])
def chatbot(query:str):
    query_words = word_tokenize(query.lower())

#buscamos sinonimos de las palabras claves
    synonyms = {word for q in query_words for word in get_synonyms(q)} | set(query_words)
    
    #filtramos la informacion del dataset buscando conincidencias en la categoria
    results = [m for m in dataset_list if any(s in m['date'].lower() for s in synonyms)]

   #si encontramos resultados los devolvemos , sino muestra mensaje que nos e encontro coincidencias
   
    return JSONResponse (content={
       "respuesta": "Aqui tienes la informacion que buscabas" if results else "No se encontro la informacion que buscabas",
       "paneles": results 
       
   })
    
# ruta para buscar la informacion por categoria especifica

@app.get('/dataset/by_date/', tags=['Dataset']) 
def get_dataset_by_category(category: str):
    #filtramos la lista de informacion del dataset segun la categoria ingresada.
    return [m for m in dataset_list if category.lower() in m['date'].lower()]
