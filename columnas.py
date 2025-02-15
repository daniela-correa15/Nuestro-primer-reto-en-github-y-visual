import pandas as pd

df = pd.read_csv("Dataset/SOLAR.csv", sep=";")
print(df.columns)  # Muestra los nombres de las columnas

import pandas as pd

df = pd.read_csv("Dataset/SOLAR.csv", sep=",")  
print(df.head())  # Muestra las primeras filas para verificar