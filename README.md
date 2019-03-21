# Desafío Legalbot
El objetivo de este desafío es construir un clasificador que sea capaz de categorizar los 3.503 objetos de sociedades entregados en el archivo "objetos.txt" en K categorías, donde K es un número definido por el usuario.

El método con el que este problema fue abordado fue a través de un análisis no supervisado de datos, en donde se buscó agrupar los objetos de cada sociedad en grupos que compartieran similares características, basándome principalmente en la frecuencia de las palabras utilizadas en cada objeto.

Para llevar a cabo esta categorización, se utilizaron 3 métodos diferentes, en donde cada uno fue contribuyendo al siguiente. En este análisis no se buscó identificar el mejor método de clasificación, sino tratar de contribuir en cada iteración a mejorar la clasificación realizada por el método previo.

## Método 1: Categorización a través de K-Means

Este método consiste en agrupar los objetos de las sociedades utilizando la similitud que exista entre las palabras utilizadas en la definición de la sociedad. Al ser el primer método utilizado, se realizó un análisis exploratorio para tratar de definir el valor de K, en donde se iteró el proceso de categorización para K=1..40 y se calculó el error (o distancia) que existía entre los valores de los objetos, y los centroides de cada categoría.

Una vez calculado el error, éste fue graficado para observar si existía algún K en donde se produjera un "salto" en la disminución del error, acorde al método del codo ([Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering))) para así elegir la cantidad de categorías. Sin embargo, no se pudo observar esto, por lo que se decidió utilizar un valor de K en donde, al agregar otra categoría, el error aumentaba.

Luego, con el valor de K = 30 seleccionado, se procedió a ejecutar el análisis y categorización utilizando el método KMeans. Es importante destacar que estos procedimientos de análisis no supervisados no asignan automáticamente una categoría, sino que agrupan los objetos en categorías que se comportan de manera similar. Por lo tanto, para definir el nombre de cada categoría, se revisaron las 10 palabras más relevantes de cada grupo designado por el método KMeans, y se definió una categoría a partir de la interpretación de estas palabras.

Finalmente, se muestra un gráfico en donde se observa la distribución de los objetos de las sociedades, en donde se observa que gran parte de estas pertenecen a un grupo generalizado.

## Método 2: Categorización a través de LDA

El método LDA (Latent Dirichlet Allocation) busca categorizar cada documento a través de la probabilidad de que éste pertenezca a cierta categoría, donde cada categoría está compuesta por palabras que poseen una probabilidad de representar a dicha categoría. Al igual que en el método anterior, el valor de K debe ser definido por el usuario, sin embargo, del primer análisis se concluyó que se podía disminuir el valor de K a 20 para tratar de agrupar, en términos más generales, los objetos de las sociedades.

Además, para este método, se asignaron palabras que no iban a ser consideradas para la categorización, utilizando el atributo "stop_words" de las funciones vectorizadoras. Esto se realizó debido a la gran presencia de conectores como palabras relevantes en los criterios de categorización. 

Al igual que en el proceso previo, una vez ejecutado el método, se procedió a definir el nombre de cada categoría observando las 10 palabras con mayor probabilidad de pertenecer a dicha categoría. Además, de manera representativa, se incluyó un gráfico donde se muestra la distribución de probabilidad de un objeto de una sociedad de pertenecer a alguna de las categorías. 

Por último, se graficó la distribución de las categorías, donde a simple vista, se observa mejor distribuida que en el método anterior.

## Método 3: Categorización a través de NMF

El método NMF (Non-Negative Matrix Factorization) busca categorizar los documentos a través de la descomposición de la matriz sparse que se genera al vectorizar la base de objetos en una combinación lineal de K dimensiones. 

Para este método, se trató de incluir una mayor cantidad de palabras al atributo "stop_words", para así tratar de categorizar utilizando palabras más representativas de los objetos de las sociedades. Además, por como se observó el comportamiento en el método LDA, se redujo el valor de K a 15.

Al igual que en los otros métodos, una vez realizada la agrupación, se observaron las 10 palabras más representativas de cada categoría para tratar de clasificar cada grupo. Una vez realizado esto, a modo comparativo con el método LDA, se graficaron los coeficientes del mismo objeto de sociedad utilizado previamente.

A diferencia del método anterior, no se obtuvo una distribución tan equilibrada como con el método LDA, sin embargo, esto puede atribuirse a la reducción de la cantidad de categorías, como también a la exclusión de palabras que no aportaban en la definición de categorías.

## Otros Métodos a Explorar

Además de estos métodos de análisis no supervisado utilizados para tratar de agrupar y categorizar los objetos de cada sociedad, otro método considerado fue clasificar manualmente un grupo de objetos elegidos aleatoriamente y así generar un modelo que permitiera predecir, para el resto de las sociedades no utilizadas, su categoría. Sin embargo, esto se descartó debido al tiempo que demandaría llevar a cabo esto pra un tamaño de muestra razonable, además de la probabilidad de que el modelo no incluyera todas las categorías.

Otro procedimiento a seguir sería tomar los objetos de sociedades que fueron clasificados de manera general, generar una nueva base de
datos con éstos y realizar nuevamente este análisis no supervisado. Probablemente, la reducción del tamaño de muestra permitiría que otras palabras que no fueron consideradas por su frecuencia original, ahora influyeran en la categorización de los objetos, permitiendo agrupar de mejor forma esta categoría.
