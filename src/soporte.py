#%%
import pandas as pd
import numpy as np
from scipy.stats import kstest,norm
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt

#%%
def abrir_archivo (nombre):
    return pd.read_csv(nombre)

def contar_valores(df,columnas):
    for columna in columnas:
        print(df[columna].value_counts())
        print('----------')
def eliminar_col (df,col):
    return df.drop(col, axis=1,inplace=True)
def eliminar_negativos (valor):
    if valor<0:
        return np.nan
    else:
        pass
def nombres_col(df):
    nuevas_columnas = {columna: columna.lower().replace(' ','_') for columna in df.columns}
    return df.rename(columns=nuevas_columnas, inplace= True)
def datos_minus(df,columnas):
    for col in columnas:
        try:
            df[col]=df[col].str.lower()
        except:
            pass

def identificar_linealidad (dataframe, col1,col2): 
    # Realizar la prueba de normalidad
    _, p_value1 = kstest(dataframe[col1], "norm")
    _, p_value2 = kstest(dataframe[col2], "norm")

    if p_value1 > 0.05 and p_value2 > 0.05:
        return 'Las columnas tienen una relación lineal'
    else:
        return 'Las columnas no tienen una relación lineal'
    
def categorizar_grupos(educacion):
    if educacion == 'bachelor' or educacion == 'high school or below':
        return "grupo B"
    else:
        return "grupo A"
    

def ab_testing(df,columna1,columna2):
    
    df_grupoA = df[df[columna1]== "grupo A"]
    df_grupoB = df[df[columna1]== "grupo B"]
         
    # hacemos un análisis visual previo
    sns.barplot(hue=columna1, y=columna2, data=df,  palette = "cool_r")
    
    plt.set_title(columna1)
        
    # Calcular la media y la desviación estándar del grupo control
    media_A_z = df_grupoA[columna2].mean()
    std_A_z = df_grupoA[columna2].std()
    print(media_A_z)
    print(std_A_z)
    # Calcular la media y la desviación estándar del grupo control
    media_B_z = df_grupoB[columna2].mean()
    std_B_z = df_grupoB[columna2].std()
    print(media_B_z)
    print(std_B_z)

    # Calcular la cantidad de datos que tenemos en el grupo control y el test
    n_A_z = len(df_grupoA)
    print(n_A_z)
    n_B_z = len(df_grupoB)
    print(n_B_z)

    # calcular el valor de la z
    z_stat = (media_B_z - media_A_z) / np.sqrt((std_A_z**2 / n_A_z) + (std_B_z**2 / n_B_z))

    # Calcular el valor p
    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))

    # Imprimir el resultado de la prueba
    alpha = 0.05
    if p_value < alpha:
        print("Hay una diferencia significativa en el número de vuelos entre el grupo A y el grupo B.")
        print("\n ---------- \n")
        print("""
            Los resultados sugieren que si que existe una diferencia entre el número de vuelos entre el grupo A y B,
            por lo que tendremos quee optar por la nueva versión de la página web
            """)
    else:
        print("No hay evidencia de una diferencia significativa en el número de vuelos entre el grupo A y el grupo B.")
        print("\n ---------- \n")
        print(""" 
            Los resultados sugieren que no existe evidencia estadística para afirmar que las medias de las muestras son distintas,
            por lo que la nueva campaña no esta ayudando a nuestro problema. 
            """)

