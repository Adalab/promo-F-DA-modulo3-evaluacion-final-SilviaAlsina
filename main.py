#%%
from src import soporte as sp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind, norm, chi2_contingency, f_oneway

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
        - Une los dos conjuntos de datos de la forma más 
        eficiente.
    2. Limpieza de Datos:
        - Elimina o trata los valores nulos, si los hay, en 
          las columnas clave para asegurar que los datos estén
          completos.
        - Verifica la consistencia y corrección de los datos
          para asegurarte de que los datos se presenten de 
          forma coherente.
        - Realiza cualquier ajuste o conversión necesaria en 
          las columnas (por ejemplo, cambiar tipos de datos) 
          para garantizar la adecuación de los datos para el 
          análisis estadístico.
'''
#%%
'''Primero llamo a la función del soporte para abrir los 
archivos'''
vuelos=sp.abrir_archivo('Customer Flight Activity.csv')
clientes=sp.abrir_archivo('Customer Loyalty History.csv')
#Ahora compruebo si hay filas duplicadas en ambos archivos
print(f'clientes tiene los siguientes duplicados: {clientes.duplicated().sum()}')
print(f'vuelos tiene los siguientes duplicados: {vuelos.duplicated().sum()}')

'''Como podemos ver en la bbdd de activity sí que hay filas 
duplicadas. Procedo a eliminarlas.'''
#%%
vuelos.drop_duplicates(inplace=True)
#A continuación, compruebo si los archivos tienen nulos
print(f'VUELOS\n{vuelos.isnull().sum()}')
'''Como podemos observar, en este primer archivo no 
encontramos ningún nulo.'''
print('-----------')
print(f'CLIENTES\n{clientes.isnull().sum()}')
'''En este segundo archivo sí que hay algún nulo. Vamos a 
comprobar si es un porcentaje elevado o no.'''
print(f'CLIENTES\n{clientes.isnull().sum()/clientes.shape[0]*100}')
'''El porcentaje de nulos en estas columnas es bastante alto. 
   Por lo tanto, todas son firmes candidatas a ser eliminadas.'''
#%%
'''Ahora vamos a hacer un estudio general del tipo de datos
 de cada archivo'''
print(vuelos.info())
'''En el caso de activity vemos como la mayoría de los 
datos son int y hay un float. No hay nada raro aquí'''
print(clientes.info())
'''En el caso de history podemos ver como la columna postal 
code es un object. Veamos si tiene sentido.'''
print(clientes['Postal Code'].unique())
'''Como podemos observar, los valores tienen letras y números,
 por lo que el tipo de dato es correcto'''
# %%
#Ahora vamos a comprobar cómo son los valores de las columnas del archivo de activity
sp.contar_valores(vuelos,vuelos.columns.tolist())
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
sp.contar_valores(clientes,clientes.columns.tolist())
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
'''Por lo que podemos observar, el archivo history contiene 
los datos de Viajeros de una compañía de viajes canadiense, 
mientras que el archivo activity muestra los viajes que ha 
tenido cada persona con dicha compañia. (Por eso tiene valores
 duplicados en Loyalty number)'''
# %%
'''Ahora que ya conozco la estructura de los datos, voy a 
proceder a la unión de las tablas. Esta unión la voy a hacer 
por la columna 'Loyalty Number', ya que es el id de cada 
cliente. Voy a realizar un merge tipo inner ya que me 
interesan los viajeros que tengamos en ambas bases de datos'''
datos=vuelos.merge(clientes, how='inner', on='Loyalty Number')
#%%
'''Ahora que ya tengo un df con todos los datos voy a 
estandarizar los nombres de las columnas y los datos.'''
sp.nombres_col(datos)
sp.datos_minus(datos,datos.columns.to_list())
#%%
'''A continuación procedo a gestionar los nulos qu nos hemos 
traído desde la bbdd de history
Nuestras columnas con alto porcentaje de nulos eran
 Cancellation Year y Cancelation Month, esto puede ser debido 
 a que no han cancelado la fidelidad con la empresa, por lo 
 que voy a modificar esos nulos'''
datos['cancellation_year'].fillna('uncancelled',inplace=True)
datos['cancellation_month'].fillna('uncancelled',inplace=True)
# %%
#Voy a eliminar los valores negativos de la columna salary
datos['salary'] = datos['salary'].apply(lambda x: np.nan if x < 0 else x)

# %%
'''AHora voy a hacer un boxplot para ver si tiene sentido 
cambiar los nulos por la media o la mediana.'''
sns.boxplot(x = 'salary', data = datos)
#%%
'''Como podemos observar en el boxplot, tiene muchísimos 
outliers, por lo tanto, lo más óptimo será sustituir los 
nulos por 'unknown'''
datos['salary'].fillna('unknown',inplace=True)
print(datos.isnull().sum())





# %%
'''· Fase 2: Visualización
Usando las herramientas de visualización que has aprendido 
durante este módulo, contesta a las siguientes preguntas 
usando la mejor gráfica que consideres:'''

# %%
'''1. ¿Cómo se distribuye la cantidad de vuelos reservados 
por mes durante el año?'''
sns.lineplot(x='month',y='flights_booked', data=datos,palette='cool_r')
plt.xlabel('mes')
plt.ylabel('vuelos reservados')
plt.title('relación vuelos/mes', fontsize = 15)
'''Como podemos observar, hay un primer pico de datos alrededor
del mes de marzo. Teniendo en cuenta que todos los clientes son
 Canadienses, ese pico de marzo puede deberse a una 
 interrupción de las clases llamada 'Mid Winter Break'. Luego 
 el siguiente pico se encuentra en los meses de verano y el 
 último correspondería a Navidad.'''
# %%
'''2. ¿Existe una relación entre la distancia de los vuelos y 
los puntos acumulados por los clientes?'''
sns.scatterplot(x='distance',y='points_accumulated',data=datos, marker = ".",s=100, palette='cool_r')
plt.xlabel('distancia')
plt.ylabel('puntos acumulados')
plt.title('relación puntos/distancia', fontsize = 15)
'''Vemos 4 líneas diferentes. Puede ser debido al tipo de 
tarjeta de fidelidad'''
#%%
sns.scatterplot(x='distance',y='points_accumulated', hue='loyalty_card',data=datos, marker = ".",s=100, palette='cool_r')
plt.xlabel('distancia')
plt.ylabel('puntos acumulados')
plt.title('relación puntos/distancia', fontsize = 15)
#plt.legend({'Aurora':'aurora','Nova':'nova','Star':'star'})
'''Aquí podemos ver como efectivamente depende del tipo de 
tarjeta, pero también de otros factores. Probemos con el año'''
#%%
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))
sp.visualizacion('distance','points_accumulated','loyalty_card',datos[datos['year']==2017],axes[0])
axes[0].set_xlabel('distancia')
axes[0].set_ylabel('puntos acumulados')
axes[0].set_title('relación puntos/distancia en 2017', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',datos[datos['year']==2018],axes[1])
axes[1].set_xlabel('distancia')
axes[1].set_ylabel('puntos acumulados')
axes[1].set_title('relación puntos/distancia en 2018', fontsize = 15)
'''Parece que en todo 2017 siguió una única tendencia lineal, pero 
en 2018 no fue así. Probemos ahora a separar por meses la gráfica 
de 2018'''
#%%
dos_18=datos[datos['year']==2018]
fig, axes = plt.subplots(nrows = 6, ncols = 2, figsize = (20, 50))
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==1],axes[0,0])
axes[0,0].set_xlabel('distancia')
axes[0,0].set_ylabel('puntos acumulados')
axes[0,0].set_title('relación puntos/distancia en enero del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==2],axes[0,1])
axes[0,1].set_xlabel('distancia')
axes[0,1].set_ylabel('puntos acumulados')
axes[0,1].set_title('relación puntos/distancia en febrero del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==3],axes[1,0])
axes[1,0].set_xlabel('distancia')
axes[1,0].set_ylabel('puntos acumulados')
axes[1,0].set_title('relación puntos/distancia en marzo del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==4],axes[1,1])
axes[1,1].set_xlabel('distancia')
axes[1,1].set_ylabel('puntos acumulados')
axes[1,1].set_title('relación puntos/distancia en abril del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==5],axes[2,0])
axes[2,0].set_xlabel('distancia')
axes[2,0].set_ylabel('puntos acumulados')
axes[2,0].set_title('relación puntos/distancia en mayo del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==6],axes[2,1])
axes[2,1].set_xlabel('distancia')
axes[2,1].set_ylabel('puntos acumulados')
axes[2,1].set_title('relación puntos/distancia en junio del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==7],axes[3,0])
axes[3,0].set_xlabel('distancia')
axes[3,0].set_ylabel('puntos acumulados')
axes[3,0].set_title('relación puntos/distancia en julio del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==8],axes[3,1])
axes[3,1].set_xlabel('distancia')
axes[3,1].set_ylabel('puntos acumulados')
axes[3,1].set_title('relación puntos/distancia en agosto del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==9],axes[4,0])
axes[4,0].set_xlabel('distancia')
axes[4,0].set_ylabel('puntos acumulados')
axes[4,0].set_title('relación puntos/distancia en septiembre del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==10],axes[4,1])
axes[4,1].set_xlabel('distancia')
axes[4,1].set_ylabel('puntos acumulados')
axes[4,1].set_title('relación puntos/distancia en octubre del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==11],axes[5,0])
axes[5,0].set_xlabel('distancia')
axes[5,0].set_ylabel('puntos acumulados')
axes[5,0].set_title('relación puntos/distancia en noviembre del 2018', fontsize = 15)
sp.visualizacion('distance','points_accumulated','loyalty_card',dos_18[dos_18['month']==12],axes[5,1])
axes[5,1].set_xlabel('distancia')
axes[5,1].set_ylabel('puntos acumulados')
axes[5,1].set_title('relación puntos/distancia en diciembre del 2018', fontsize = 15)
'''Como podemos ver, para algunos meses la tendencia lineal es 
independiente del tipo de tarjeta, pero para otros sí que es 
dependiente'''
#%%
'''Para todos los casos vistos vamos a comprobar la linealidad'''
dos_17=datos[datos['year']==2017]
print('Para el año 2017')
sp.identificar_linealidad(dos_17,'distance','points_accumulated')
dos_18=datos[datos['year']==2018]
for i in range (1,13):
    if i== 1 or i in range(5,13):
        print(f'Para el año 2018 y mes {i}')
        sp.identificar_linealidad(dos_18[dos_18['month']==i],'distance','points_accumulated')
    else:
        loyalty=dos_18['loyalty_card'].unique().tolist()
        mes=dos_18[dos_18['month']==i]
        for unique in loyalty:
            print(f'Para el año 2018, mes {i} y tarjeta tipo {unique}:')
            sp.identificar_linealidad(mes[mes['loyalty_card']==unique],'distance','points_accumulated')
# %%
'''3. ¿Cuál es la distribución de los clientes por provincia o 
estado?'''
plt.hist(x = 'Province', data =clientes, color = "turquoise", edgecolor = "white")
plt.tick_params(axis='x', rotation=60)
plt.xlabel('Provincia')
plt.ylabel('Clientes')
plt.title('distribución clientes/provincia', fontsize = 15)
'''Como podemos observar, la mayoría de los clientes son de las 
provincias de Ontario, British Columbia y Quebec'''
#%%
'''4. ¿Cómo se compara el salario promedio entre los diferentes
 niveles educativos de los clientes?'''
'''Voy a crear un nuevo df donde descartemos las filas con 
salarios desconocidos'''
#%%
clientes['Salary'].fillna('unknown',inplace=True)
salario=clientes[clientes['Salary']!='unknown']
salario=salario.groupby('Education')['Salary'].mean().reset_index()
sns.barplot(x = 'Education', y = 'Salary', data=salario, order=['High School or Below'
            ,'Bachelor','Master','Doctor'], palette='cool_r',
            ci = None)
plt.xlabel('Educación')
plt.ylabel('Salario')
plt.title('Comparación salario/educación', fontsize = 15)

# %%
'''5. ¿Cuál es la proporción de clientes con diferentes tipos de 
tarjetas de fidelidad?'''
sns.countplot(x = 'Loyalty Card', data = clientes, palette='cool_r')
plt.xlabel('Tarjeta Fidelidad')
plt.ylabel('Clientes')
plt.title('Proporción Tarjeta/Clientes', fontsize = 15)
# %%
'''6. ¿Cómo se distribuyen los clientes según su estado civil y 
género?'''
sns.countplot(x = 'Marital Status', data = clientes,hue='Gender',palette='cool_r')
plt.xlabel('Estado Civil')
plt.ylabel('Clientes')
plt.title('Proporción Clientes/Estado Civil', fontsize = 15)
# %%
'''· Fase 3: Evaluación de Diferencias en Reservas de Vuelos por 
Nivel Educativo.
Objetivo del Ejercicio: Utilizando un conjunto de datos que hemos 
compartido, se busca evaluar si existen diferencias significativas 
en el número de vuelos reservados según el nivel educativo de los 
clientes. Para ello, los pasos que deberas seguir son:
    1. Preparación de Datos:
        - Filtra el conjunto de datos para incluir únicamente las 
          columnas relevantes: 'Flights Booked' y 'Education'.
    2. Análisis Descriptivo:
        - Agrupa los datos por nivel educativo y calcula 
        estadísticas descriptivas básicas (como el promedio, 
        la desviación estandar, los percentiles) del número de 
        vuelos reservados para cada grupo.
    3. Prueba Estadística:
        - Realiza una prueba de A/B testing para determinar si 
        existe una diferencia significativa en el número de vuelos 
        reservados entre los diferentes niveles educativos.'''
# %%
datos['group']=datos['education'].apply(sp.categorizar_grupos)
# %%
vuelos_por_persona=datos.groupby('loyalty_number')['flights_booked'].sum().reset_index()
grupo_por_persona=datos[['loyalty_number','group']].drop_duplicates()
# %%
vuelos_por_persona=vuelos_por_persona.merge(grupo_por_persona, how='inner', on='loyalty_number')
# %%
sp.normalidad(vuelos_por_persona,'flights_booked')

# %%
sp.homogeneidad(vuelos_por_persona,'group','flights_booked')
#%%
vuelos_por_persona.groupby('group')['loyalty_number'].count().reset_index()
# %%
sp.test_man_whitney(vuelos_por_persona,'flights_booked','grupo A','grupo B','group')
# %%
sns.histplot(x = 'flights_booked', hue='group',data = vuelos_por_persona)
# %%
sp.ab_testing_2(vuelos_por_persona,'group','flights_booked',)
# %%
