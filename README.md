# Predecir_respuestas_de_tweets

proyecto para la materia `Text Mining`, `2021 FAMAF UNC`

Autor: `Joaquin Feltes`

Profesora: `Laura Alonso Alemany`

# Introduccion

La idea principal es predecir cuántas respuestas va a tener un tweet. Para esto se tomo el aproach de tarea de pretexto, usando la cantidad de respuestas de un tuit para generar una tag para el entrenamiento del modelo de preddicion. Ademas, se va a utilizar el predictor de respuestas para ser agregado a un pipeline de generacion de respuestas a tuits. Este generador fue el trabajo final de Lautaro Martinez, un compañero de la materia (github.com/LMartinezEXEX/Generador_Contestaciones).

# Dataset

El dataset fue el creado por Mariano Schmidt, para su tesis de Licenciatura: Explotando características contextuales para la detección de posturas en Twitter en el marco de la vacunación del COVID-19 en Argentina. Para pedir el dataset pueden pedirmelo por privado y tambien dejo su github: https://github.com/mschmidt4.

El dataset contiene 163.180 tweets en español.

y tiene las siguientes caracteristicas:

- created_at: contiene la fecha de envio del tweet a la aplicación.
- id_str: contiene el ID especifico del tweet.
- full_text: contiene el texto generado por el usuario del tweet.
- in_reply_to_status_id: de existir, contiene el ID del tweet al cual se esta respondiendo.
- in_reply_to_user_id: de existir, contiene el ID del usuario del tweet al cual se estra respondiendo.
- user.id: contiene el ID especifico del usuario que creo el tweet.

# Consideraciones y enfoques

En principio el trabajo de predecir cuantas respuestas va a tener un tweet puede parecer sencillo, pero hay que tener en cuenta todos los distintos enfoques que son posibles para entrenar un modelo y todas las caracteristicas que pueden agregarse a este para mejorar la prediccion. Por esto se van a mostrar en este trabajo varios enfoques que han sido tomados, para el metodo de entrenamiento pero tambien para la clasificacion de los tweets.

# Preproceso

Lo primero que se hace con el dataset es la eliminacion de arrobas, hashtags y urls. No hubo mucho trabajo de ingenieria en esta parte del proyecto, ya que muchos companieros que utilizaron el mismo dataset se concentraron en esto y por lo general hicieron un preproceso similar a este.

Luego se cuenta para cada tuit la cantidad de respuestas que posee, para agregarlo como una nueva caracteristica 'ans'.

# Division del Dataset

Como ya se dijo, una parte importante del trabajo es la division en clases de las cantidades de respuestas, ya que esto es lo que se quiere predecir con el modelo final. Este proceso se llevo a cabo en el archivo `Parseo_dataset.ipynb`

Primero para observar la distribucion de respuestas de los tweets del dataset tenemos el siguiente histograma:

## ![Histograma_discreto](./images/histograma_nans.png)

Aqui se puede ver un gran problema, hay una excesiva cantidad de tuits con 0 respuestas, y vemos como baja rapidamente la cantidad ya llegando a 4. Se tiene una distribucion exponencial inversa. Esto tiene mucho sentido, ya que es lo mas comun que un tuit tenga 0 o 1 respuestas.

Por esta razon, se deben dividir en clases, para que cada clase tenga una cantidad de representantes similar a la de las otras. Mas adelante se vera que si no se hace esto, los modelos tienden a aprender solo por la cantidad de representantes y una clase sobrerepresentada va a hacer que se predigan muchos elementos para ella.

Se tomo como parametro para la cantidad 6500 representantes, ya que hay aproximadamente esa cantidad con 2 respuestas y tambien con 3 o mas. Se tomaron tambien en cuenta divisiones con mas clases, separando en 3-4, 5-7 ,7-10 , +10, pero la cantidad de representantes era muy baja, y no se generaron resultados satisfactorios para esas divisiones. Otra division interesante, que no se ha llegado a explorar demasiado es la de separar entre 0 , pocas y muchas respuestas. O incluso es interesante la division entre ninguna o alguna respuesta.

Haciendo entonces la division por clases de [0, 1, 2, +3] tenemos la siguiente distribucion de respuestas:

## ![Histograma_clases1](./images/histograma_clases_1.png)

Ahora redujimos bastante el problema, ya que no tenemos infinitas cantidades de respuestas posibles, solo lo limitamos a 4 clases pensando que los tuits de mas de 2 respuestas van a pertenecer todos al mismo grupo.

Pero todavia tenemos el problema de la sobrepoblacion de la clase del 0 y en menor medida, la del 1.

Lo que se decidio para que la cantidad de elementos sea aproximadamente la misma que las de las otras dos clases, fue dividir la clase del 0 en 18 subgrupos y la del 1 en 5 subgrupos, todos tomados al azar.

Con esto en mente, podemos ver como queda el histograma de las clases divididas exactamente con 1/18 de la clase del 0 y 1/5 de la del 1:

## ![Histograma_divididas](./images/histograma_divididas.png)

Vemos que esto deja los datos muchisimo mejor, ahora todas las clases tienen aproximadamente la misma cantidad de elementos y no deberia haber sezgo en el entrenamiento de datos.

Para la division en subgrupos, se tomaron hizo una division al azar con el framework de pandas, por lo que los resultados enseñados a continuacion no van a poder replicarse exactamente.

Por ultimo se dividio todo el dataset para tener un 80% para entrenamiento, 10% develop y 10% para testeo final.

# Aproximaciones

## Primer aproximacion utilizando Bag of words con CountVectorizer

La primera aproximacion que se tomo fue utilizando [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) de sklearn para representar los tweets con "Bag of words" (bolsa de palabras). Esto significa que cada tweet va a ser un vector, donde cada dimension es una palabra del vocabulario. Se tomo como vocabulario las 15000 palabras mas recurrentes del dataset, para mayor eficiencia y facilidad en manejo de memoria pero tambien para no incluir muchas palabras que ocurren solo una vez en todo el dataset y no van a aportar nada interesante.

La primera aproximacion que se hizo, que se puede ver en las primeras versiones de git del proyecto, fue con la division de clases [0,1,2-3,4-6,7-10,11-20,20+]. Esta fue tomada al azar, para tener una primera observacion de como se comportaba el modelo. Aqui se puede ver la Matriz de confusion de esta aproximacion.

## ![Primer_test_BOW](./images/1er_test_BOW.png)

Se puede observar que a la mayoria de elementos se les asigno la clase de 0 respuestas, esto da un buen resultado general (69% de acierto), pero haciendo el analisis de la matriz de confusion, se puede notar que la diagonal (donde se encuentran los resultados correctos) solo tiene la mayoria de la fila en la clase del 0. Aca podemos ver con claridad el sezgo por la sobrepoblacion de la clase, el cual se podria discutir que no es tan malo, si en las muestras reales de tweets la mayoria tiene esa cantidad de respuestas. Pero hay que recalcar que no se quiere conseguir un alto porcentaje de acierto prediciendo siempre que un tuit va a tener 0 respuestas, esto no nos genera ningun aprendizaje ni aporta ningun beneficio.

---

---

---

---

---

---

---

---

---

PONER LA CANTIDAD DE ELEMENTOS POR CLASE en bow discreto y en embeddings tambien, para ver como es la distribucion, y ver asi como en la matriz de confusion depende de la cantidad de elementos.

EN los embeddings con fasttext explicar que se uso lo de naza, que no puedo subir el original pero me lo pueden pedir por mi mail de contacto

---

PONER EN EL README ALGUN WARNING DE QUE LOS PICKLE OCUPAN MUCHA MEMORIA (3 GB SI GUARDAS LOS 90 + EL TRAINING)

LAUTI MARTINEZ
-AGREGAR COMO LO UNIRIA DEL TODO
-AGREGAR LOS CAMBIOS QUE HICE, CREO QUE FUERON SOLO EN MAIN, FIJARSE EN EL DRIVE
