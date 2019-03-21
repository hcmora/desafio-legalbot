# Por ultimo, probaremos agrupar los datos a traves del metodo de factorizacion de matrices.
# Basicamente, lo que buscaremos aqui es descomponer la matriz sparse que se genera con la 
# frecuencia relativa de cada palabra y aproximarla con una combinacion lineal de k temas

with open('objetos.txt') as txt:
    lines = txt.read().split('\n\n')

from sklearn.feature_extraction.text import TfidfVectorizer

# Nuevamente, agregamos las restricciones al vectorizador encontradas previamente, y sumamos otras
# observadas en el proceso de LDA que no aportan a definir un tipo de objeto de sociedad
stop_w = ['de','la','a','el','que','en','los','las','con','al','sus','del','por','como','para','toda','todo','servicios',
         'cualquier','otros','general','tipo','tipos','actividades','ya','similares','objeto','no','actividad','otra',
         'terceros','cuenta','propia','bienes','clase','ajena','act','propios','sociedad','sociedades','socios','su','sea',
         'relacionadas','otras','relacionados','especializado','especializados','nuevos','empleadores']

tfidf = TfidfVectorizer(max_df=0.9,min_df=2,stop_words=stop_w)
mtx = tfidf.fit_transform(lines)
# Ahora importamos la clase NMF (Non matrix Factorization)
from sklearn.decomposition import NMF

# Tal como fue mencionado en el metodo LDA, trataremos de definir 15 tipos de objeto
k = 15
nmf_model = NMF(n_components=k,random_state=7)
nmf_model.fit(mtx)

# Observamos las 10 palabras mas utilizadas por tipos de objeto
for i, tema in enumerate(nmf_model.components_):
    print(f"Tema {i}:")
    print([tfidf.get_feature_names()[index] for index in tema.argsort()[-10:]])
    print("\n")

# Asociamos los tipos de objetos a cada entrada
import pandas as pd
df = pd.DataFrame()
temas_resultantes = nmf_model.transform(mtx)
df['Texto'] = lines
df['Grupo'] = temas_resultantes.argmax(axis=1)
df.head()

rep = {0:'Venta de Alimentos',1:'Servicios de Construccion',2:'Servicios de Transporte',3:'Inversiones, compras y ventas',
       4:'Asesoramiento Empresarial',5:'Ingenieria y Construccion',6:'Servicio Automotriz',7:'Venta de Articulos/Alimentos',
       8:'Arriendo de Equipos, Arquitectura e Ingenieria',9:'Asesoramiento, Inversiones y Arrendamiento',10:'Bar y Restaurants',
       11:'Fabricacion de textiles y muebles',12:'Servicios Medicos',13:'Servicios de Transporte',
       14:'Servicios Informaticos y de Telecomunicaciones'}

df['Nombre del Grupo'] = df['Grupo'].map(rep)
df.head()

# De manera similar al metodo LDA, podemos ver los coeficientes que definen a que grupo pertence algun objeto.
# Elegimos nuevamente el elemento 1759
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(11,6))
plt.xticks(range(0,15),range(0,15))
plt.bar(range(0,15),temas_resultantes[1759],color="#3366cc")
plt.title("Coeficientes para el Objeto 1759")
plt.xlabel("Grupo")
plt.ylabel("Coeficiente")
plt.show()

# Vemos que el coeficiente mas relevante es el 10, seguido por el 11. Podemos observar su clasificacion
print('Grupo mas probable: '+rep[10])
print('Segundo grupo probable: '+rep[11])
print(df['Texto'].iloc[1759])

# Por ultimo, observamos el grupo mas frecuente
plt.figure(figsize=(10,6))
sns.countplot(x='Grupo',data=df,color="#3366cc")
plt.title("Frecuencia de Cada Tema")
plt.show()

# En este caso, lo mas frecuente fue dado por el grupo 3
print(rep[3])