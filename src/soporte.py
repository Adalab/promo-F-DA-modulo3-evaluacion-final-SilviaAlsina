#%%
import pandas as pd
from scipy.stats import kstest,levene
import scipy.stats as st
import seaborn as sns
from sklearn.linear_model import LinearRegression
#%%
def abrir_archivo (nombre):
    return pd.read_csv(nombre)

def contar_valores(df,columnas):
    for columna in columnas:
        print(df[columna].value_counts())
        print('----------')

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
    
def normalidad(dataframe, columna):
    statistic, p_value = st.kstest(dataframe[columna],'norm')
    if p_value > 0.05:
        print(f"Para la columna {columna} los datos siguen una distribución normal.")
    else:
        print(f"Para la columna {columna} los datos no siguen una distribución normal.")

def homogeneidad (dataframe, columna, columna_metrica):
    valores_evaluar = []
    
    for valor in dataframe[columna].unique():
        valores_evaluar.append(dataframe[dataframe[columna]== valor][columna_metrica])

    statistic, p_value = st.levene(*valores_evaluar)
    if p_value > 0.05:
        print(f"Para la métrica {columna_metrica} las varianzas son homogéneas entre grupos.")
    else:
        print(f"Para la métrica {columna_metrica}, las varianzas no son homogéneas entre grupos.")


def test_man_whitney(dataframe,metrica, grupo_A, grupo_B, columna_grupos):
    grupoA = dataframe[dataframe[columna_grupos] == grupo_A]
    grupoB = dataframe[dataframe[columna_grupos] == grupo_B]
    
    metrica_grupoA = grupoA[metrica]
    metrica_grupoB = grupoB[metrica]
    
    u_statistic, p_value = st.mannwhitneyu(metrica_grupoA, metrica_grupoB)
    
    if p_value < 0.05:
        print(f"Las medianas entre los grupos son diferentes.")
    else:
        print(f"Las medianas entre los grupos son iguales.")
            



