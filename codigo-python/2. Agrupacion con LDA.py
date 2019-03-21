# Ahora se probara agrupar mediante LDA. Lo que se busca realizar aqui es asignar una distribucion de 
# probabilidad a cada documento, en donde esta distribucion representara la probabilidad de pertenecer
# a un grupo u otro. Asimismo, los grupos tendran una distribucion de probabilidad para las palabras 
# que perteneceran a este

with open("objetos.txt") as txt:
    lines = txt.read().split("\n\n")

from sklearn.feature_extraction.text import CountVectorizer

# Para este metodo, necesitamos crear la matriz de frecuencia de cada palabra.
# Aplicamos las mismas restricciones para incluir palabras que en el caso anterior, pero ademas sumamos otras que, de lo que
# fue observado, no aportan a definir el objeto de la sociedad. En particular, definimos el atributo stop_words con conectores
# que no aportan informacion relevante.

stop_w = ['de','la','a','el','que','en','los','las','con','al','sus','del','por','como','para','toda','todo']
cv = CountVectorizer(max_df=0.9,min_df=2,stop_words=stop_w)

# Creamos la matriz sparse
mtx = cv.fit_transform(lines)

# Importamos lo necesario para realizar el metodo LDA (Latent Dirichlent Allocation)
from sklearn.decomposition import LatentDirichletAllocation

k = 20

LDA = LatentDirichletAllocation(n_components=k,random_state=7)

LDA.fit(mtx)

# Observemos ahora las 10 palabras con mayor probabilidad de aparecer en cada grupo
for i, tema in enumerate(LDA.components_):
    print(f"Tema {i}:")
    print([cv.get_feature_names()[index] for index in tema.argsort()[-10:]])
    print("\n")

# Procedemos con asignar los temas a cada objeto de sociedades
temas_resultantes = LDA.transform(mtx)
import pandas as pd
df = pd.DataFrame()
df['Texto'] = lines
df['Grupo'] = temas_resultantes.argmax(axis=1)
df.head()

rep = {0:'Servicios de Salud', 1:'Comercializacion Agricola', 2:'Servicios de Construccion', 3:'Servicios Informaticos', 
       4:'Servicio Automotriz', 5:'Comercializacion de Articulos', 6:'Arriendo de Equipos de Transporte', 
       7:'Fabricacion y venta de Articulos', 8:'Servicios Generales', 9:'Venta de Insumos', 10:'Servicios de Construccion',
       11:'Servicios de Transporte', 12:'Servicios de Alimentos', 13:'Servicios Generales', 14:'Arquitectura, Ingenieria y Construccion', 
       15:'Servicios de Construccion', 16:'Servicios de Construccion', 17:'Servicios Inmobiliarios', 18:'Venta de Alimentos', 
       19:'Fabricacion y venta de Textiles'}

df['Nombre del Grupo'] = df['Grupo'].map(rep)
df.head()
# Podemos graficar para algun objeto de sociedad su probabilidad de pertenecer a alguno de los temas:
import seaborn as sns
import matplotlib.pyplot as plt
# Elegimos a modo aleatorio la entrada 1759
plt.figure(figsize=(11,6))
plt.xticks(range(0,20),range(0,20))
plt.bar(range(0,20),temas_resultantes[1759],color="#3366cc")
plt.title("Distribucion de Probabilidad para el Objeto 1759")
plt.xlabel("Grupo")
plt.ylabel("Probabilidad")
plt.show()
# Vemos que tiene mayor probabilidad de pertenecer al grupo 18, y como segunda opcion al grupo 1:
print('Grupo mas probable: '+rep[18])
print('Segundo grupo probable: '+rep[1])
print(df['Texto'].iloc[1759])
# Observemos cual de todos los grupos es el que se repite con mayor frecuencia
plt.figure(figsize=(10,6))
sns.countplot(x='Grupo',data=df,color="#3366cc")
plt.title("Frecuencia de Cada Tema")
plt.show()
# Observamos que el grupo 18 es el que aparece mas veces
print(rep[18])
# Para este caso, se definieron 16 grupos unicos. En el siguiente metodo, trataremos de identificar 15 grupos 
# de otra forma.
len(df['Nombre del Grupo'].unique())