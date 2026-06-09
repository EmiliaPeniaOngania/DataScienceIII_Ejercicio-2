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

El dataset cargado contiene aproximadamente 22.000 registros y 9 columnas.
5. Selección del texto
texts = df["Review Text"].dropna().astype(str)

Se seleccionó la columna correspondiente a las reseñas para realizar el análisis de texto.
6. Tokenización
La tokenización consiste en dividir el texto en unidades mínimas denominadas tokens.
sample_text = texts.iloc[0]

doc = nlp(sample_text)

tokens = [token.text for token in doc]

Esta etapa permite transformar texto libre en elementos individuales que posteriormente podrán ser analizados.
7. Eliminación de Stopwords Estándar
Se aplicó la lista de stopwords incorporada en spaCy.
def remove_stopwords(text):

    doc = nlp(text)

    return [
        token.text.lower()
        for token in doc
        if not token.is_stop
        and token.is_alpha
    ]

La utilización de la lista estándar permite eliminar automáticamente palabras frecuentes del idioma inglés como:
the
and
to
a
it
entre otras.
8. Personalización de Stopwords
Además de la lista estándar, se incorporó una lista personalizada.
custom_stopwords = {
    "amazon",
    "product",
    "item"
}

La decisión se fundamenta en que estas palabras aparecen repetidamente en las reseñas debido al contexto del dataset y no aportan información discriminante para comprender la opinión de los usuarios.
Por ejemplo, la palabra "amazon" se encuentra presente tanto en reseñas positivas como negativas, por lo que su capacidad explicativa resulta limitada.
9. Aplicación de Stopwords Personalizadas
def remove_custom_stopwords(text):

    doc = nlp(text)

    return [
        token.text.lower()
        for token in doc
        if not token.is_stop
        and token.is_alpha
        and token.text.lower() not in custom_stopwords
    ]

10. Comparación de Frecuencias
Se realizó una comparación de frecuencias antes y después del proceso de limpieza con el objetivo de evaluar el impacto de la eliminación de stopwords sobre la representación textual del corpus. Esta comparación permite identificar qué términos dominan el vocabulario antes del preprocesamiento y cuáles emergen como conceptos relevantes una vez reducido el ruido lingüístico. 
11. Visualización
Se construyó un gráfico de barras para visualizar las palabras más frecuentes luego de la eliminación de stopwords.
La visualización permitió identificar rápidamente los conceptos predominantes dentro del conjunto de reseñas.



Comparación de Resultados
Palabras más frecuentes antes de la limpieza

Análisis
Antes de la limpieza predominan términos muy frecuentes del idioma inglés como "the", "i", "to", "and" y "for". Estas palabras aparecen constantemente en las reseñas, pero aportan poco valor semántico para comprender los temas tratados por los usuarios.

Palabras más frecuentes después de la limpieza

Análisis de Resultados
Se observa que los términos predominantes luego de la limpieza se relacionan principalmente con aspectos operativos de la experiencia de compra. Palabras como "customer", "service", "delivery" y "refund" indican que una parte significativa de las reseñas se concentra en procesos de atención al cliente, logística y gestión de devoluciones. Esto sugiere que estos factores constituyen elementos clave en la percepción del servicio por parte de los usuarios.
Desde una perspectiva empresarial, la identificación temprana de estos conceptos permitiría priorizar mejoras en los procesos con mayor impacto sobre la satisfacción del cliente. Asimismo, estos resultados podrían utilizarse como insumo para futuros modelos de clasificación automática de reclamos o análisis de sentimiento.

Impacto del Preprocesamiento en Ciencia de Datos
La calidad del preprocesamiento tiene un impacto directo sobre el desempeño de los modelos de Machine Learning aplicados a texto.
Cuando se utilizan representaciones como Bag of Words o TF-IDF, cada palabra del vocabulario genera una característica adicional. La presencia excesiva de stopwords incrementa artificialmente la dimensionalidad del espacio de representación y puede dificultar la identificación de patrones relevantes.
La eliminación de palabras irrelevantes permite que los algoritmos asignen mayor peso a conceptos con verdadero contenido semántico, mejorando potencialmente métricas como precisión, recall y F1-score en tareas de clasificación.
Asimismo, la reducción del vocabulario disminuye el consumo de memoria y acelera los tiempos de entrenamiento, aspectos especialmente importantes cuando se trabaja con grandes volúmenes de datos.
Por este motivo, la limpieza de texto constituye una práctica estándar dentro de los proyectos profesionales de NLP.

Reflexión Crítica
Si bien la eliminación de stopwords genera beneficios evidentes, no existe una estrategia universal aplicable a todos los problemas.
En determinados escenarios algunas palabras consideradas stopwords pueden aportar información relevante. Por ejemplo, términos relacionados con negaciones como "not" pueden modificar completamente el significado de una frase y afectar significativamente un análisis de sentimientos.
Por esta razón, la selección de stopwords debe responder a los objetivos específicos del proyecto y validarse empíricamente mediante experimentación.
La personalización realizada en este trabajo constituye un ejemplo de adaptación al dominio, eliminando términos frecuentes asociados a la plataforma de comercio electrónico analizada para mejorar la calidad interpretativa de los resultados.

Conclusión
En este trabajo se implementó un proceso completo de limpieza de texto utilizando spaCy sobre un conjunto real de reseñas de Amazon.
La tokenización permitió segmentar adecuadamente los documentos, mientras que la eliminación de stopwords estándar y personalizadas redujo significativamente el ruido presente en los datos.
La comparación de frecuencias antes y después del preprocesamiento demostró que la limpieza facilita la identificación de conceptos relevantes vinculados con la experiencia de los usuarios, tales como servicio al cliente, entregas y devoluciones.
Los resultados obtenidos confirman que la eliminación de stopwords constituye una etapa esencial dentro de cualquier pipeline de procesamiento de lenguaje natural, mejorando tanto la interpretabilidad de los datos como la calidad potencial de futuros modelos predictivos.
Como líneas futuras de trabajo podrían incorporarse técnicas adicionales de lematización, análisis de sentimientos, embeddings y modelos supervisados de clasificación para profundizar el estudio de las opiniones de los clientes.

