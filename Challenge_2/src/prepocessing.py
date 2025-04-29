# Script for loading and preprocessing data
from pathlib import Path
import re
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from typing import Optional
import spacy
from spacy.lang.en.examples import sentences # Da ejemplos de frases en español para pruebas.
from spacy.lang.en.stop_words import STOP_WORDS # Stop words en inglés
from nltk.corpus import stopwords # Lista de stopwords pero de la librería NLTK (más general).
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('punkt_tab')

# Load dataframe
def load_data_frame(path=None, num_samples:Optional[int]=None, random_seed: int = 42) -> pd.DataFrame:
    """"
    : param path: path of the csv file.
    : param num_samples: the number of samples to draw from the data frame; if None, use all samples.
    : param random_seed: the random seed to use when sampling data points
    """

    df = pd.read_csv(filepath_or_buffer=path)

    if num_samples is not None:
        df = df.sample(num_samples,
                       random_state=random_seed)
    print("Data is loaded")
    
    return df

# Prepocessing
def preprocessing_data(df, columna='pros'):
    """
    :return: datadrame with the interest column for analize
    """

    # Since there are two columns that do not provide us with information, we will eliminate them.
    df_text = df[[columna]].copy()

    # Nos quedamos con los primeros 250 registros del dataframe con la finalidad de trabajar más rápido
    df_text = df_text.head(250)
    df_text

    # Se aplica la función limpiar a la columna -pros-
    # ==============================================================================
    df_text['pros'] = df_text['pros'].apply(lambda col: limpiar(col))

    # Creamos una lista de stop words
    stop_words = list(stopwords.words('english'))

    # Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
    df_text['pros_sw'] = df_text['pros'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # Creamos una columna con las lematizaciones para las palabras sin stop words
    df_text['pros_lem'] = df_text['pros_sw'].apply(lambda x: lemmatizatizar(x))

    # Creamos los n-gramas y los graficamos
    generar_ngramas(df_text,'pros_lem', 2, 5,10)

    return df_text
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
# Definimos una función para aplicar lemmatization
def lemmatizatizar(text):

    lemmatizer_text = nlp(text)
    lemmatizer_text = [word.lemma_ for word in lemmatizer_text]
    return " ".join(lemmatizer_text)
def limpiar(texto):
    '''
    Esta función limpia y tokeniza el texto en palabras individuales.
    El orden en el que se va limpiando el texto no es arbitrario.
    El listado de signos de puntuación se ha obtenido de: print(string.punctuation)
    y re.escape(string.punctuation)
    '''
    # Se convierte todo el texto en str
    nuevo_texto = str(texto)
    # Se convierte todo el texto a minúsculas
    nuevo_texto = nuevo_texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex , ' ', nuevo_texto)
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    
    return(nuevo_texto)

def generar_ngramas(df, columna, n_min=2, n_max=5, top=20):
    """
        Genera y grafica los n-gramas más frecuentes de una columna de texto de un DataFrame.

    Parámetros:
    - df: DataFrame de entrada.
    - columna: nombre de la columna de texto.
    - n: tamaño del n-grama (por ejemplo, 2 para bigramas, 3 para trigramas).
    - top: número de n-gramas más frecuentes a graficar.
    """
    # Definimos los colores 
    colores = ['plum', 'skyblue', 'salmon', 'lightgreen']
    # Unir todos los textos en uno solo
    texto_total = ' '.join(df[columna].dropna().astype(str))

    # Tokenizar el texto
    tokens = nltk.word_tokenize(texto_total, language='english')

    for n in range(n_min, n_max + 1):
        # Generar los n-gramas para este valor de n
        n_gramas = list(ngrams(tokens, n))

        # Contar frecuencia de cada n-grama
        contador = Counter(n_gramas)

        # Tomar los 'top' más comunes
        ngramas_mas_comunes = contador.most_common(top)

        if not ngramas_mas_comunes:
            print(f"No se encontraron {n}-gramas.")
            continue

        # Separar los datos para graficar
        frases = [' '.join(grama) for grama, freq in ngramas_mas_comunes]
        frecuencias = [freq for grama, freq in ngramas_mas_comunes]

        plots_path = Path(__file__).parent.resolve() # Convierte la ruta relativa en absoluta, tenía conflicto con las diagonales
        plots_path = plots_path.parent / 'plots'
        if not os.path.exists(plots_path):
            # If it doesn't exist, it will create it
            os.makedirs(plots_path)
        # Crear el gráfico
        plt.figure(figsize=(10, 6))
        color = colores[(n - 2) % len(colores)]
        plt.barh(frases[::-1], frecuencias[::-1], color=color)
        plt.xlabel('Frecuencia')
        plt.title(f'Top {top} {n}-gramas más frecuentes')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path,f'{n}-gramas.png'))
        plt.close()