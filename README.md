#  Agente RAG de Análisis de Ventas con Mistral AI

Proyecto de análisis inteligente de datos de ventas utilizando un agente RAG (Retrieval-Augmented Generation) construido con **LangChain** y **Mistral AI**. El sistema permite hacer consultas en lenguaje natural sobre un dataset de ventas globales.


## Dataset

El dataset (`data_sample.csv`) contiene registros de ventas globales con las siguientes columnas principales:

## Tecnologías utilizadas

- **Python 3**
- **Pandas** — carga y manipulación del dataset
- **LangChain** — framework para construir el agente RAG
- **Mistral AI** — modelo de lenguaje (`mistral-small-latest`, `mistral-large-latest`) y embeddings (`mistral-embed`)
- **FAISS** — vector store para búsqueda semántica eficiente
- **Google Colab** — entorno de ejecución

---

##  Nota sobre los archivos .ipynb

Los notebooks originales (`.ipynb`) pueden mostrar un error de renderizado
en GitHub debido al peso de los outputs guardados. Por eso este repositorio
incluye los equivalentes en `.py`, que contienen exactamente el mismo código
y son compatibles con cualquier entorno Python o Google Colab.
