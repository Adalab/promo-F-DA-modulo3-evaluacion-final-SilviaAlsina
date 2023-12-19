#%%
import pandas as pd
import numpy as np
from scipy.stats import kstest,norm,shapiro,levene
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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

def visualizacion (x,y,group,data,axes):
    sns.scatterplot(x=x,y=y, hue=group,data=data, marker = ".",s=100, palette='cool_r',ax=axes)
def identificar_linealidad (df, col1,col2): 
    X = df[[col1]]
    y = df[col2]
    modelo_regresion = LinearRegression()
    modelo_regresion.fit(X, y)
    print(f"Puntaje R^2: {modelo_regresion.score(X, y):.4f}")
def categorizar_grupos(educacion):
    if educacion == 'college' or educacion == 'high school or below':
        return "grupo B"
    else:
        return "grupo A"
    

def ab_testing(df,columna1,columna2):

    df_grupoA = df[df[columna1]== "grupo A"]
    df_grupoB = df[df[columna1]== "grupo B"]
    
    df_2=df.groupby(columna1)[columna2].sum().reset_index()
    # hacemos un análisis visual previo
    sns.barplot(hue=columna1, y=columna2, data=df_2,  palette = "cool_r")       
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
        
def normalidad(dataframe, columna):
    """
    Evalúa la normalidad de una columna de datos de un DataFrame utilizando la prueba de Shapiro-Wilk.

    Parámetros:
        dataframe (DataFrame): El DataFrame que contiene los datos.
        columna (str): El nombre de la columna en el DataFrame que se va a evaluar para la normalidad.

    Returns:
        None: Imprime un mensaje indicando si los datos siguen o no una distribución normal.
    """

    statistic, p_value = st.kstest(dataframe[columna],'norm')
    if p_value > 0.05:
        print(f"Para la columna {columna} los datos siguen una distribución normal.")
    else:
        print(f"Para la columna {columna} los datos no siguen una distribución normal.")

def homogeneidad (dataframe, columna, columna_metrica):
    
    """
    Evalúa la homogeneidad de las varianzas entre grupos para una métrica específica en un DataFrame dado.

    Parámetros:
    - dataframe (DataFrame): El DataFrame que contiene los datos.
    - columna (str): El nombre de la columna que se utilizará para dividir los datos en grupos.
    - columna_metrica (str): El nombre de la columna que se utilizará para evaluar la homogeneidad de las varianzas.

    Returns:
    No devuelve nada directamente, pero imprime en la consola si las varianzas son homogéneas o no entre los grupos.
    Se utiliza la prueba de Levene para evaluar la homogeneidad de las varianzas. Si el valor p resultante es mayor que 0.05,
    se concluye que las varianzas son homogéneas; de lo contrario, se concluye que las varianzas no son homogéneas.
    """
    
    # lo primero que tenemos que hacer es crear tantos conjuntos de datos para cada una de las categorías que tenemos, Control Campaign y Test Campaign
    valores_evaluar = []
    
    for valor in dataframe[columna].unique():
        valores_evaluar.append(dataframe[dataframe[columna]== valor][columna_metrica])

    statistic, p_value = st.levene(*valores_evaluar)
    if p_value > 0.05:
        print(f"Para la métrica {columna_metrica} las varianzas son homogéneas entre grupos.")
    else:
        print(f"Para la métrica {columna_metrica}, las varianzas no son homogéneas entre grupos.")


def test_man_whitney(dataframe,metrica, grupo_A, grupo_B, columna_grupos):

    """
    Realiza la prueba de Mann-Whitney U para comparar las medianas de las métricas entre dos grupos en un DataFrame dado.

    Parámetros:
    - dataframe (DataFrame): El DataFrame que contiene los datos.
    - columnas_metricas (list): Una lista de nombres de columnas que representan las métricas a comparar entre los grupos.
    - grupo_control (str): El nombre del grupo de control en la columna especificada por columna_grupos.
    - grupo_test (str): El nombre del grupo de test en la columna especificada por columna_grupos.
    - columna_grupos (str): El nombre de la columna que contiene la información de los grupos. Por defecto, "campaign_name".

    Returns 
    No devuelve nada directamente, pero imprime en la consola si las medianas son diferentes o iguales para cada métrica.
    Se utiliza la prueba de Mann-Whitney U para evaluar si hay diferencias significativas entre los grupos.
    """
    # filtramos el DataFrame para quedarnos solo con los datos de grupoA
    grupoA = dataframe[dataframe[columna_grupos] == grupo_A]
    
    # filtramos el DataFrame para quedarnos solo con los datos de control
    grupoB = dataframe[dataframe[columna_grupos] == grupo_B]
    
    metrica_grupoA = grupoA[metrica]
    metrica_grupoB = grupoB[metrica]
    # iteramos por las columnas de las metricas para ver si para cada una de ellas hay diferencias entre los grupos
            # aplicamos el estadístico
    u_statistic, p_value = st.mannwhitneyu(metrica_grupoA, metrica_grupoB)
    
    if p_value < 0.05:
        print(f"Las medianas entre los grupos son diferentes.")
    else:
        print(f"Las medianas entre los grupos son iguales.")
            



