#%%
from src import soporte as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
'''
· Fase 1: Exploración y Limpieza
    1. Exploración Inicial:
        - Realiza una exploración inicial de los datos para 
          identificar posibles problemas, como valores nulos, 
          atípicos o datos faltantes en las columnas relevantes.
        - Utiliza funciones de Pandas para obtener información 
          sobre la estructura de los datos, la presencia de 
          valores nulos y estadísticas básicas de las columnas 
          involucradas.
        - Une los dos conjuntos de datos de la forma más eficiente.
    2. Limpieza de Datos:
        - Elimina o trata los valores nulos, si los hay, en las columnas clave para asegurar que los datos estén completos.
        - Verifica la consistencia y corrección de los datos para asegurarte de que los datos se presenten de forma coherente.
        - Realiza cualquier ajuste o conversión necesaria en las columnas (por ejemplo, cambiar tipos de datos) para garantizar la adecuación de los datos para el análisis estadístico.
'''

#%%
#Primero llamo a la función del soporte para abrir lso archivos
activity=sp.abrir_archivo('Customer Flight Activity.csv')
history=sp.abrir_archivo('Customer Loyalty History.csv')
#Ahora compruebo si hay filas duplicadas en ambos archivos
print(history.duplicated().sum())
print(activity.duplicated().sum())
#Como podemos ver en la bbdd de activity sí que hay filas duplicadas. Procedo a eliminarlas.
activity.drop_duplicates(inplace=True)
#A continuación, compruebo si los archivos tiene nulos
print(activity.isnull().sum())
#Como podemos observar, en este primer archivo no encontramos ningún nulo.
print('-----------')
print(history.isnull().sum())
#En este segundo archivo sí que hay algún nulo. Vamos a comprobar si es un porcentaje elevado o no.
print(history.isnull().sum()/history.shape[0]*100)
#Como podemos observar, el porcentaje de nulos en estas columnas es bastante alto. Por lo tanto, todas son firmes candidatas a ser eliminadas.
#%%
#Ahora vamos a haer un estudio general del tipo de datos de cada archivo
print(activity.info())
#En el caso de activity vemos como la matoría de los datos son int y hay un float. No hay nada raro aquí
print(history.info())
#En el caso de history podemos ver como la columna postal code es un object. Veamos si tiene sentido.
print(history['Postal Code'].unique())
#Como podemos observar, los valores tienen letras y números, por lo que el tipo de dato es correcto
# %%
#Ahora vamos a comprobar cómo son los valores de las columnas del archivo de activity
sp.contar_valores(activity,activity.columns.tolist())
'''Loyalty number: Hay un valor predominante 678205.
   Year: Sólo tenemos 2017 y 2018
   Month: No faltan meses. Todos tienen el mismo value_count
   Flights Booked: 0 es el valor predominante
   Flights with Companions: 0 es el valor predominante
   Total Flights: 0 es el valor predominante
   Distance:0 es el valor predominante
   Points Accumulated: 0 es el valor predominante
   Points Redeemed: 0 es el valor predominante
   Dollar Cost Points Redeemed: 0 es el valor predominante'''
#%%
#Haremos el mismo proceso para history
sp.contar_valores(history,history.columns.tolist())
'''Loyalty number: Todos los valores son únicos
   Country: Sólo tenemos Canada
   Province: Predomina Ontario
   City: Predomina Toronto, seguido de Vancouver
   Postal Code: Predomina V6E 3D9
   Gender: Male y Female
   Education: Predomina Bachelor
   Salary: Predomina 101933.0, además hay un valor negativo
   Marital Status: Married, Single, Divorced. Predomina el primero
   CLV: Predomina 8564.77
   Enrollment Type: Standard y 2018 Promotion. Predomina el primero
   Enrollment Year: Predomina 2018
   Enrollment Month:Predomina 5, pero por poco
   Cancellation Year: Predomina 2018, pero poco
   Cancellation Month: predominan 12,11 y 8'''
# %%
#Ahora que ya he gestionado los nulos voy a proceder a la unión de las tablas. Esta unión la voy a hacer por la columna 'Loyalty Number', ya que es el id de cada cliente. Voy a realizar un merge tipo inner ya que me interesan los viajeros que tengamos en ambas bases de datos
datos=activity.merge(history, how='inner', on='Loyalty Number')
#%%
#Ahora que ya tengo un df con todos los datos voy a estandarizar los nombres de las columnas y los datos.
sp.nombres_col(datos)
sp.datos_minus(datos,datos.columns.to_list())
print(datos.isnull().sum())
#%%
#Nuestras columnas con alto porcentaje de nulos eran Cancellation Year y Cancelation Month, esto puede ser debido a que no han cancelado la fidelidad con la empresa, por lo que voy a modificar esos nulos
datos['cancellation_year'].fillna('uncancelled',inplace=True)
datos['cancellation_month'].fillna('uncancelled',inplace=True)
print(datos.isnull().sum())
#Por lo que podemos observar, el archivo history contiene los datos de Viajeros de una compañía de viajes canadiense, mmientras que el archivo activity muestra los viajes que ha tenido cada persona con dicha compañia. (Por eso tiene valores duplicados en Loyalty number)
# %%
#Voy a eliminar los valores negativos de la columna salary
datos['salary'] = datos['salary'].apply(lambda x: np.nan if x < 0 else x)
print(datos.isnull().sum())
# %%
#AHora voy a hacer un boxplot para ver si tiene sentido cambiar los nulos por la media o la mediana.
sns.boxplot(x = 'salary', data = datos)
#%%
#Como podemos observar en el boxplot, tiene muchísimos outliers, por lo tanto, lo más óptimo será sustituir los nulos por 'unknown'
datos['salary'].fillna('unknown',inplace=True)
print(datos.isnull().sum())
# %%

# %%
'''· Fase 2: Visualización
Usando las herramientas de visualización que has aprendido durante este módulo, contesta a las siguientes preguntas usando la mejor gráfica que consideres:'''

# %%
#1. ¿Cómo se distribuye la cantidad de vuelos reservados por mes durante el año?
sns.lineplot(x='month',y='flights booked', data=datos,palette='cool_r');
'''Como podemos observar, hay un primer pico de datos alrededor del mes de marzo. Teniendo en cuenta que todos 
los clientes son Canadienses, ese pico de marzo puede deberse a una interrupción de las clases llamada 'Mid Winter
 Break'. Luego el siguiente pico se encuentra en los meses de verano y el último correspondería a Navidad.'''
# %%
#2. ¿Existe una relación entre la distancia de los vuelos y los puntos acumulados por los clientes?
sns.scatterplot(x='distance',y='points accumulated',data=datos, marker = ".",s=100, color="turquoise")
#Como podemos observar, parece que hay una relación lineal. Vamos a comprobarlo
sp.identificar_linealidad(datos,'distance','flights booked')
# %%
#3. ¿Cuál es la distribución de los clientes por provincia o estado?
plt.hist(x = 'province', data = datos, color = "turquoise",bins=11, edgecolor = "white")
plt.tick_params(axis='x', rotation=60)
#Como podemos observar, la mayoría de los clientes son de las provincias de Ontario, British Columbia y Quebec
#%%
#4. ¿Cómo se compara el salario promedio entre los diferentes niveles educativos de los clientes?
#Voy a crear un nuevo df donde descartemos las filas con salarios desconocidos
salario=datos[datos['salary']!='unknown']
salario.groupby('education')['salary'].mean().reset_index()
sns.barplot(x = 'education', y = 'salary', data=salario,
            ci = None)
# %%
#5. ¿Cuál es la proporción de clientes con diferentes tipos de tarjetas de fidelidad?
sns.histplot(x = 'Loyalty Card', data = history)

# %%
#6. ¿Cómo se distribuyen los clientes según su estado civil y género?
sns.countplot(x = 'Marital Status', data = history,hue='Gender',palette='cool_r')
# %%
'''· Fase 3: Evaluación de Diferencias en Reservas de Vuelos por Nivel Educativo
Objetivo del Ejercicio: Utilizando un conjunto de datos que hemos compartido, se busca evaluar si existen diferencias significativas en el número de vuelos reservados según el nivel educativo de los clientes. Para ello, los pasos que deberas seguir son:
    1. Preparación de Datos:
        - Filtra el conjunto de datos para incluir únicamente las columnas relevantes: 'Flights Booked' y 'Education'.
    2. Análisis Descriptivo:
        - Agrupa los datos por nivel educativo y calcula estadísticas descriptivas básicas (como el promedio, la desviación estandar, los percentiles) del número de vuelos reservados para cada grupo.
    3. Prueba Estadística:
        - Realiza una prueba de A/B testing para determinar si existe una diferencia significativa en el número de vuelos reservados entre los diferentes niveles educativos.'''
# %%
reservas=datos.groupby('education')['flights_booked'].sum().reset_index()
datos.groupby('education')['flights_booked'].mean().reset_index()
datos.groupby('education')['flights_booked'].std().reset_index()
datos.groupby('education')['flights_booked'].mean().reset_index()
# %%
np.percentile(datos.groupby('education')['flights_booked'], [25,50,75])
# %%
datos.groupby('education')['flights_booked'].quantile(0.25,0.5,0.75)
# %%
def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_{:02.0f}'.format(n*100)
    return percentile_

# %%
datos.groupby('education')['flights_booked'].agg(percentile(0.5))
# %%
datos['group']=datos['education'].apply(sp.categorizar_grupos)
#Hipotesis nula[HO]-->No hay cambios significativos entre los dos grupos
#Hipotesis alternativa [H1]-->Hay cambios significativos entre los dos grupos
# %%
sp.ab_testing(datos,'group','flights_booked');

# %%
