Práctica N°2: Implementación y Evaluación de Eliminación de Stopwords con spaCy
Introducción
El procesamiento de lenguaje natural (Natural Language Processing - NLP) requiere una etapa previa de preparación de datos que permita transformar texto sin estructura en información útil para el análisis. Dentro de estas tareas de preprocesamiento, la eliminación de stopwords constituye una de las técnicas más utilizadas debido a su capacidad para reducir ruido y mejorar la representación de los documentos.
Las stopwords son palabras de alta frecuencia que suelen aportar escaso contenido semántico al texto. Artículos, preposiciones, pronombres y conectores aparecen repetidamente en la mayoría de los documentos sin contribuir significativamente a la diferenciación temática entre ellos.
El objetivo de este trabajo es implementar diferentes estrategias de eliminación de stopwords utilizando la biblioteca spaCy sobre un conjunto real de reseñas de Amazon. Además, se analizará el impacto de la limpieza sobre la distribución de términos, evaluando cómo cambia la representación textual antes y después del preprocesamiento.
Finalmente, se realizará una personalización de la lista de stopwords incorporando términos específicos del dominio estudiado para observar cómo esta decisión puede mejorar la calidad del análisis posterior.

Marco Teórico
La limpieza de texto es una etapa fundamental dentro de cualquier pipeline de procesamiento de lenguaje natural. Los algoritmos de clasificación, análisis de sentimientos y modelado temático suelen ser sensibles al ruido presente en los datos, por lo que resulta necesario aplicar técnicas que permitan conservar únicamente la información relevante.
Una de las estrategias más utilizadas consiste en eliminar stopwords. Estas palabras poseen una frecuencia muy elevada dentro de un idioma, pero generalmente contienen poca información semántica para distinguir documentos entre sí.
Entre los principales beneficios de la eliminación de stopwords se encuentran:
Reducción de dimensionalidad del vocabulario.
Disminución del ruido en los datos.
Mejora de la eficiencia computacional.
Mayor interpretabilidad de los resultados.
Incremento potencial del rendimiento de modelos predictivos.
Sin embargo, la eliminación de stopwords debe aplicarse cuidadosamente. Dependiendo del contexto de negocio, algunas palabras consideradas irrelevantes podrían contener información valiosa. Por este motivo, muchos proyectos incorporan listas personalizadas adaptadas al dominio específico bajo análisis.
La biblioteca spaCy proporciona mecanismos eficientes para la tokenización y la identificación automática de stopwords, facilitando la construcción de procesos de limpieza reproducibles y escalables.

Dataset Utilizado
Para el desarrollo de la práctica se utilizó el dataset Amazon Reviews Dataset disponible en Kaggle.
El conjunto de datos contiene más de 22.000 reseñas realizadas por usuarios de Amazon e incluye información relacionada con:
Texto de la reseña.
Calificación otorgada.
Fecha de publicación.
País del usuario.
Información complementaria sobre la experiencia de compra.
Este dataset resulta adecuado para tareas de NLP debido a la gran cantidad de texto libre generado por los usuarios.

Desarrollo de la Solución
1. Importación de librerías
Se utilizaron las siguientes librerías:
import pandas as pd
import numpy as np
import spacy

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

2. Carga del modelo de spaCy
nlp = spacy.load("en_core_web_sm")

3. Carga del dataset
df = pd.read_csv(
    "Amazon_Reviews.csv",
    encoding="utf-8",
    engine="python",
    on_bad_lines="skip"
)

4. Inspección inicial
print(df.shape)
print(df.columns)
df.head()

El dataset cargado contiene 21.214 registros y 9 columnas. 
5. Selección del texto
texts = df["Review Text"].dropna().astype(str)

Se seleccionó la columna correspondiente a las reseñas para realizar el análisis de texto.
6. Tokenización
La tokenización consiste en dividir el texto en unidades mínimas denominadas tokens.
sample_text = texts.iloc[0]

doc = nlp(sample_text)

tokens = [token.text for token in doc]

Esta etapa permite transformar texto libre en elementos individuales que posteriormente podrán ser analizados.
7. Tokenización Personalizada
Se realizó una prueba adicional de tokenización sobre una expresión compuesta relacionada con Amazon Prime para observar el comportamiento del tokenizador de spaCy. Si bien no fue necesario modificar las reglas internas del tokenizer para este ejercicio, la prueba permitió verificar la correcta segmentación de términos presentes en el dominio analizado. 

8. Eliminación de Stopwords Estándar
Se aplicó la lista de stopwords incorporada en spaCy.
def remove_stopwords(text):

    doc = nlp(text)

    return [
        token.text.lower()
        for token in doc
        if not token.is_stop
        and token.is_alpha
    ]

La utilización de la lista estándar permite eliminar automáticamente palabras funcionales del idioma inglés que presentan alta frecuencia y bajo contenido semántico. 
9. Personalización de Stopwords
Además de la lista estándar, se incorporó una lista personalizada.
custom_stopwords = {
    "amazon",
    "product",
    "item"
}

La decisión se fundamenta en que estas palabras aparecen repetidamente en las reseñas debido al contexto del dataset y no aportan información discriminante para comprender la opinión de los usuarios.
Por ejemplo, la palabra "amazon" se encuentra presente tanto en reseñas positivas como negativas, por lo que su capacidad explicativa resulta limitada.
10. Aplicación de Stopwords Personalizadas
def remove_custom_stopwords(text):

    doc = nlp(text)

    return [
        token.text.lower()
        for token in doc
        if not token.is_stop
        and token.is_alpha
        and token.text.lower() not in custom_stopwords
    ]

11. Comparación de Frecuencias
Se realizó una comparación de frecuencias antes y después del proceso de limpieza con el objetivo de evaluar el impacto de la eliminación de stopwords sobre la representación textual del corpus.
Para ello se procesó la totalidad del dataset, compuesto por 21.214 reseñas, obteniendo las palabras más frecuentes antes y después de aplicar las técnicas de limpieza implementadas.
La comparación permite identificar qué términos dominan el vocabulario antes del preprocesamiento y cuáles emergen como conceptos relevantes una vez reducido el ruido lingüístico.

12. Visualización
Se construyó un gráfico de barras utilizando la biblioteca Seaborn para visualizar las palabras más frecuentes luego de la eliminación de stopwords.
plt.figure(figsize=(10,5))
sns.barplot(
   data=after_df,
   x="Frequency",
   y="Word"
)
plt.title(
   "Palabras más frecuentes después de eliminar stopwords"
)
plt.show()
La visualización facilita la interpretación de los resultados obtenidos y permite identificar rápidamente los conceptos predominantes dentro del conjunto de reseñas analizado.




Comparación de Resultados
Palabras más frecuentes antes de la limpieza
Análisis
Antes de aplicar técnicas de limpieza textual, las palabras más frecuentes corresponden principalmente a artículos, pronombres y conectores propios del idioma inglés.
Entre los términos predominantes aparecen palabras como "i", "the", "to", "and" y "a", las cuales presentan una frecuencia muy elevada debido a su uso habitual dentro de la estructura gramatical del idioma.
La presencia dominante de estas palabras evidencia una elevada proporción de ruido lingüístico dentro del corpus, dificultando la identificación de los temas realmente relevantes presentes en las reseñas.
Si bien estos términos son necesarios para la construcción de las oraciones, aportan escasa información semántica para comprender los aspectos específicos de la experiencia de compra relatada por los usuarios.

Palabras más frecuentes después de la limpieza
Análisis de Resultados
Luego de aplicar la eliminación de stopwords estándar de spaCy y las stopwords personalizadas definidas para el dominio de Amazon, las palabras más frecuentes obtenidas fueron:
customer
service
delivery
order
time
prime
account
refund
items
day
La limpieza permitió eliminar términos gramaticales de alta frecuencia y resaltar conceptos directamente vinculados con la experiencia de compra reportada por los usuarios.
Se observa una fuerte presencia de términos asociados a atención al cliente, entregas, gestión de pedidos, tiempos de respuesta, cuentas de usuario y devoluciones. Esto indica que una parte importante de las opiniones expresadas por los usuarios se concentra en aspectos operativos y de servicio vinculados al funcionamiento de la plataforma.
Desde una perspectiva empresarial, estos resultados permiten identificar áreas críticas que impactan directamente sobre la satisfacción del cliente. Asimismo, la información obtenida podría utilizarse como insumo para futuros modelos de análisis de sentimiento, clasificación automática de reclamos o detección temprana de problemas operativos.
La comparación con las frecuencias obtenidas antes de la limpieza demuestra que el preprocesamiento mejora significativamente la capacidad de interpretar los temas predominantes dentro del corpus.

Comparación entre Técnicas Aplicadas
La tokenización estándar permitió segmentar correctamente las reseñas en palabras y signos de puntuación, constituyendo el punto de partida para las etapas posteriores del procesamiento de texto.
Posteriormente, la tokenización personalizada fue implementada utilizando la clase Tokenizer de spaCy. Aunque los resultados obtenidos fueron similares a los de la tokenización estándar debido a las características del dataset analizado, esta técnica demuestra la posibilidad de adaptar el proceso de segmentación a dominios específicos mediante reglas personalizadas.
En una segunda etapa se aplicó la eliminación de stopwords estándar incorporada en spaCy. Esta técnica permitió eliminar artículos, pronombres, preposiciones y otras palabras funcionales de alta frecuencia que aportan poco contenido semántico al análisis.
Finalmente, se incorporó una lista de stopwords personalizadas compuesta por los términos "amazon", "product" e "item". Estas palabras aparecen con elevada frecuencia debido al contexto propio del dataset y no contribuyen significativamente a diferenciar opiniones o problemáticas específicas.
Los resultados obtenidos muestran que cada técnica aplicada aporta una mejora incremental sobre la calidad del texto procesado. La combinación de tokenización, eliminación de stopwords estándar y personalización de stopwords permitió obtener una representación más limpia, interpretable y útil para futuras tareas de análisis de texto y aprendizaje automático.

Impacto del Preprocesamiento en Ciencia de Datos
La calidad del preprocesamiento tiene un impacto directo sobre el desempeño de los modelos de Machine Learning aplicados a texto.
Cuando se utilizan representaciones como Bag of Words o TF-IDF, cada palabra del vocabulario genera una característica adicional. La presencia excesiva de stopwords incrementa artificialmente la dimensionalidad del espacio de representación y puede dificultar la identificación de patrones relevantes.
La eliminación de palabras irrelevantes permite que los algoritmos asignen mayor peso a conceptos con verdadero contenido semántico, mejorando potencialmente métricas como precisión, recall y F1-score en tareas de clasificación.
Asimismo, la reducción del vocabulario disminuye el consumo de memoria y acelera los tiempos de entrenamiento, aspectos especialmente importantes cuando se trabaja con grandes volúmenes de datos.
Por este motivo, la limpieza de texto constituye una práctica estándar dentro de los proyectos profesionales de NLP.

Reflexión Crítica
La limpieza de texto produjo una reducción del vocabulario de aproximadamente 1.15%, eliminando palabras irrelevantes y términos muy frecuentes que aportaban poco valor analítico.
Antes de la limpieza predominaban artículos, pronombres y conectores como "the", "to", "and" o "of", además de términos generales como "amazon". Luego de aplicar stopwords estándar y personalizadas, emergieron conceptos más representativos del contenido de las reseñas, tales como "customer", "service", "delivery", "account" y "refund".
Estos resultados permiten identificar con mayor claridad los temas principales abordados por los usuarios, especialmente aspectos relacionados con atención al cliente, entregas y devoluciones. Desde una perspectiva de negocio, este tipo de procesamiento facilita la detección de problemáticas recurrentes y constituye una etapa fundamental para futuros análisis de sentimiento o modelos de clasificación automática de comentarios.

Conclusión
En este trabajo se implementó un proceso completo de limpieza y preprocesamiento de texto utilizando la biblioteca spaCy sobre un conjunto real de reseñas de Amazon.
El análisis fue realizado sobre la totalidad del dataset, compuesto por 21.214 registros y 9 variables, permitiendo obtener resultados representativos del comportamiento general de las reseñas analizadas.
La tokenización permitió segmentar adecuadamente los documentos, mientras que la implementación de una tokenización personalizada demostró la flexibilidad de spaCy para adaptarse a diferentes dominios de aplicación.
Posteriormente, la eliminación de stopwords estándar y personalizadas redujo significativamente el ruido presente en los datos, favoreciendo la identificación de conceptos relevantes dentro del corpus.
La comparación de frecuencias antes y después del preprocesamiento evidenció que la limpieza textual permite destacar términos estrechamente relacionados con la experiencia de los usuarios, tales como atención al cliente, entregas, gestión de pedidos y devoluciones.
Los resultados obtenidos confirman que el preprocesamiento constituye una etapa fundamental dentro de cualquier proyecto de Procesamiento de Lenguaje Natural, ya que mejora la calidad de los datos, facilita la interpretación de los resultados y contribuye potencialmente a mejorar el desempeño de futuros modelos predictivos.
Como líneas futuras de trabajo podrían incorporarse técnicas complementarias como lematización, análisis de sentimiento, representación mediante embeddings y modelos supervisados de clasificación para profundizar el estudio de las opiniones de los clientes.


