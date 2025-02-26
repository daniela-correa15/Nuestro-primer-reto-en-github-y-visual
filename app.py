import streamlit as st #streamlit es la libreria que nos ayuda a crear la api, donde podemos dibujar botones, cuadros de texto y mostrar resultados visuales
import pandas as pd # libreria inteligente que nos ayuda a leer y organizar datos (como los archivos csv)
import math #es como una calculadora que nos ayuda a hacer operaciones matematicas

# Cargar los datasets , aqui estamos leyendo dos archivos
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
municipios_disponibles = dataset['municipio'].unique().tolist() #lista de todos los municipios que tenemos en solar3
paneles = dataset1['panel_solar'].unique().tolist() #lista de todos los paneles que tenemos en paneles_csv

# Configuración de la interfaz de Streamlit
st.title("Chatbot:Calculadora de Paneles Solares ☀️") #titulo de la aplicacion
st.markdown("""
Esta aplicación te ayuda a calcular el número de paneles solares necesarios para tu consumo de energía en un municipio específico.
""") #es un mensaje que explica que hace la calculadora :markdown

#aqui vamos a pedir los datos al usuario para poder hacer el calculo
# Selección de municipio, el usuario escoge el municipio
municipio = st.selectbox("Seleccione un municipio", municipios_disponibles)

# Selección de panel solar, el usuario escoge el panel
panel = st.selectbox("Seleccione un tipo de panel solar", paneles)

# Entrada de consumo mensual, el usuario ingresa de su factura el consumo en kwh
consumo = st.number_input("Ingrese su consumo mensual de energía (kWh)", min_value=0.0, format="%.2f")

# Botón para calcular
if st.button("Calcular Paneles Solares"):
    # Obtener las horas de radiación para el municipio, busca cuantas horas de sol tiene el municipio que se escogio
    datos_municipio = dataset[dataset['municipio'] == municipio]
    if datos_municipio.empty:
        st.error("Municipio no encontrado")
    else:
        n_horas_dia = float(datos_municipio.iloc[0]['horasDia'])
        
        # Obtener la potencia del panel solar, busca la potencia del panel solar que eligio el usuario
        datos_panel = dataset1[dataset1['panel_solar'] == panel]
        if datos_panel.empty:
            st.error("Panel solar no encontrado")
        else:
            potw = float(datos_panel.iloc[0]['potencia'])
            
            # Calcular el número de paneles solares, 
            numero_paneles = ((consumo * 1000) / 30) / (n_horas_dia * potw)
            numero_paneles_redondeado = math.ceil(numero_paneles)
            
            # Mostrar los resultados
            st.success(f"Resultados para {municipio}:")
            st.write(f"**Horas de radiación por día:** {n_horas_dia} horas")
            st.write(f"**Panel solar seleccionado:** {panel}")
            st.write(f"**Potencia del panel:** {potw} W")
            st.write(f"**Número de paneles necesarios:** {numero_paneles_redondeado}")