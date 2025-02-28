import streamlit as st
import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cargar los datasets
def cargar_dataset():
    df = pd.read_csv("Dataset/SOLAR3.csv", delimiter=";", quotechar='"', on_bad_lines="skip")[['id', 'municipio', 'energiaMensual', 'horasDia']]
    return df.fillna('')

def cargar_dataset2():
    df1 = pd.read_csv("Dataset/PANELES_CSV.csv", delimiter=";", quotechar='"', on_bad_lines="skip")[['id', 'panel_solar', 'potencia', 'tension', 'valor']]
    return df1.fillna('')

# Cargar los datos
dataset = cargar_dataset()
dataset1 = cargar_dataset2()

# Obtener listas únicas de municipios y paneles
municipios_disponibles = dataset['municipio'].unique().tolist()
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

# Configuración de la interfaz de Streamlit
st.title("Calculadora de Paneles Solares ☀️")
st.markdown("""
Esta aplicación te ayuda a calcular el número de paneles solares necesarios para tu consumo de energía en un municipio específico.
También utiliza un modelo de machine learning para predecir el consumo de energía basado en las horas de radiación solar.
""")

# Selección de municipio
municipio = st.selectbox("Seleccione un municipio", municipios_disponibles)

# Selección de panel solar
panel = st.selectbox("Seleccione un tipo de panel solar", paneles)

# Entrada de consumo mensual
consumo = st.number_input("Ingrese su consumo mensual de energía (kWh)", min_value=0.0, format="%.2f")

# Botón para calcular
if st.button("Calcular Paneles Solares y Consumo Estimado"):
    # Obtener las horas de radiación para el municipio
    datos_municipio = dataset[dataset['municipio'] == municipio]
    if datos_municipio.empty:
        st.error("Municipio no encontrado")
    else:
        n_horas_dia = float(datos_municipio.iloc[0]['horasDia'])
        
        # Obtener la potencia del panel solar
        datos_panel = dataset1[dataset1['panel_solar'] == panel]
        if datos_panel.empty:
            st.error("Panel solar no encontrado")
        else:
            potw = float(datos_panel.iloc[0]['potencia'])
            
            # Calcular el número de paneles solares necesarios
            numero_paneles = ((consumo * 1000) / 30) / (n_horas_dia * potw)
            numero_paneles_redondeado = math.ceil(numero_paneles)
            
            # Calcular el consumo estimado que pueden cubrir los paneles
            consumo_estimado = (numero_paneles_redondeado * n_horas_dia * potw * 30) / 1000
            
            # Predecir el consumo de energía usando el modelo de machine learning
            consumo_predicho = modelo.predict([[n_horas_dia]])
            
            # Mostrar los resultados
            st.success(f"Resultados para {municipio}:")
            st.write(f"**Horas de radiación por día:** {n_horas_dia} horas")
            st.write(f"**Panel solar seleccionado:** {panel}")
            st.write(f"**Potencia del panel:** {potw} W")
            st.write(f"**Consumo mensual ingresado:** {consumo} kWh")
            st.write(f"**Número de paneles necesarios:** {numero_paneles_redondeado}")
            st.write(f"**Consumo estimado que pueden cubrir los paneles:** {consumo_estimado:.2f} kWh")
            st.write(f"**Consumo predicho usando Machine Learning:** {consumo_predicho[0]:.2f} kWh")