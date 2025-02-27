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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar datasets
def cargar_dataset():
    df = pd.read_csv("Dataset/SOLAR3.csv", delimiter=";", quotechar='"', on_bad_lines="skip")[['id', 'municipio', 'energiaMensual', 'horasDia']]
    return df.fillna('')

def cargar_dataset2():
    df1 = pd.read_csv("Dataset/PANELES_CSV.csv", delimiter=";", quotechar='"', on_bad_lines="skip")[['id', 'panel_solar', 'potencia', 'tension', 'valor']]
    return df1.fillna('')

dataset = cargar_dataset()
municipios_disponibles = dataset['municipio'].unique().tolist()
dataset1 = cargar_dataset2()
paneles = dataset1['panel_solar'].unique().tolist()

# Entrenar un modelo de regresión lineal simple
def entrenar_modelo():
    # Supongamos que queremos predecir 'energiaMensual' basado en 'horasDia'
    X = dataset[['horasDia']]
    y = dataset['energiaMensual']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

# Inicializar el modelo
modelo = entrenar_modelo()

# Inicialización de FastAPI
app = FastAPI(title="Cálculo de Paneles Solares con Machine Learning", version="1.0.0")

# Ruta de inicio
@app.get("/", tags=['Home'])
def home():
    return HTMLResponse("<h1>Bienvenido a la API de cálculo de paneles solares con Machine Learning</h1>")

# Ruta para obtener el dataset completo
@app.get("/Dataset/SOLAR3.csv", tags=['SOLAR3'])
def obtener_dataset():
    df = cargar_dataset()
    dataset_dict = df.to_dict(orient="records")
    dataset_convertido = [{k: (int(v) if isinstance(v, (np.int64, np.int32)) else float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in row.items()} for row in dataset_dict]
    return dataset_convertido

# Ruta para obtener un registro por ID
@app.get("/Dataset/{id}", tags=['Dataset'])
def obtener_por_id(id: int):
    df = cargar_dataset()
    dataset_list = df.to_dict(orient="records")
    resultado = next((m for m in dataset_list if m['id'] == id), None)
    if resultado is None:
        raise HTTPException(status_code=404, detail="Información no encontrada")
    return resultado

# Ruta para calcular el consumo estimado basado en la cantidad de paneles solares
@app.get("/calcular_consumo", tags=['Cálculo'])
def calcular_consumo(
    municipio: str = Query(..., description="Seleccione un municipio", enum=municipios_disponibles),
    panel: str = Query(..., description="Seleccione un tipo de panel", enum=paneles),
    cantidad_paneles: int = Query(..., description="Cantidad de paneles solares")
):
    # Obtener las horas de radiación para el municipio seleccionado
    datos_municipio = dataset[dataset['municipio'] == municipio]
    if datos_municipio.empty:
        raise HTTPException(status_code=404, detail="Municipio no encontrado")
    
    n_horas_dia = float(datos_municipio.iloc[0]['horasDia'])
    
    # Obtener la potencia del panel solar
    datos_panel = dataset1[dataset1['panel_solar'] == panel]
    if datos_panel.empty:
        raise HTTPException(status_code=404, detail="Panel solar no encontrado")
    
    potw = float(datos_panel.iloc[0]['potencia'])
    
    # Calcular el consumo estimado que pueden cubrir los paneles
    consumo_estimado = (cantidad_paneles * n_horas_dia * potw * 30) / 1000
    
    return JSONResponse(content={
        "municipio": municipio,
        "panel_solar": panel,
        "cantidad_paneles": cantidad_paneles,
        "horas_radiacion_dia": n_horas_dia,
        "potencia_panel_w": potw,
        "consumo_estimado_kWh": consumo_estimado
    })

# Ruta para calcular paneles solares
@app.get("/calculo_paneles", tags=['Cálculo'])
def chatbot_calcular_paneles(
    consumo: float = Query(..., description="Consumo de energía en kWh"),
    municipio: str = Query(..., description="Seleccione un municipio", enum=municipios_disponibles),
    panel: str = Query(..., description="Seleccione un tipo de panel", enum=paneles)
):
    datos_municipio = dataset[dataset['municipio'] == municipio]
    if datos_municipio.empty:
        raise HTTPException(status_code=404, detail="Municipio no encontrado")
    
    n_horas_dia = float(datos_municipio.iloc[0]['horasDia'])
    
    datos_panel = dataset1[dataset1['panel_solar'] == panel]
    if datos_panel.empty:
        raise HTTPException(status_code=404, detail="Panel solar no encontrado")
    
    potw = float(datos_panel.iloc[0]['potencia'])
    
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