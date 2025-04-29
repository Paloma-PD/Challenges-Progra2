from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud

# Definimos la función para trabajar NLP
def nlp_process(df_text):

    ## MODELO LDA
    documents = df_text['pros_lem'].dropna().astype(str).tolist()
    # Como ya no hay que quitar stopwords, simplemente tokenizamos (split)
    texts = [doc.split() for doc in documents]
    # Crear diccionario
    dictionary = corpora.Dictionary(texts)
    # Crear corpus (bag of words)
    corpus = [dictionary.doc2bow(text) for text in texts]
    # Entrenar modelo LDA
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=4,  # Número de tópicos
        random_state=42,
        passes=10,
        iterations=50
    )
    # Mostrar los temas
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx}\nWords: {topic}\n")

    # Convirtiendo los resultados a dataframe
    # Obtener la distribución de tópicos para cada documento
    lda_corpus = lda_model[corpus]
    # Crear listas para almacenar resultados
    topic_distributions = []
    dominant_topics = []
    dominant_percents = []
    # Recorrer cada documento
    for doc_topics in lda_corpus:
        # Crear vector de tópicos inicializado en 0
        topics_probs = [0] * lda_model.num_topics
        
        # Llenar con las probabilidades reales
        for topic_num, prob in doc_topics:
            topics_probs[topic_num] = prob

        topic_distributions.append(topics_probs)
        
        # Determinar el tópico dominante y su porcentaje
        dominant_topic = max(doc_topics, key=lambda x: x[1])
        dominant_topics.append(dominant_topic[0])     # Número del topic
        dominant_percents.append(dominant_topic[1])   # Valor decimal

    # Crear el DataFrame
    df_topics = pd.DataFrame(
        topic_distributions,
        columns=[f"topic_{i}" for i in range(lda_model.num_topics)]
    )

    # Agregar Dominant_Topic y Perc_Dominant_Topic
    df_topics['Dominant_Topic'] = dominant_topics
    df_topics['Perc_Dominant_Topic'] = dominant_percents

    #Unir con df_text
    df_tt = pd.concat([df_text.reset_index(drop=True), df_topics.reset_index(drop=True)], axis=1)
   
    # WORDS CLOUDS
    # Para cada topic, hacer un WordCloud
    plots_path = Path(__file__).parent.resolve() # Convierte la ruta relativa en absoluta, tenía conflicto con las diagonales
    plots_path = plots_path.parent / 'plots'
    if not os.path.exists(plots_path):
        # If it doesn't exist, it will create it
        os.makedirs(plots_path)
    for i in range(lda_model.num_topics):
        plt.figure(figsize=(8, 6))
        plt.imshow(WordCloud(background_color='white').fit_words(dict(lda_model.show_topic(i, 30))))
        plt.axis('off')
        plt.title(f'Topic {i}')
        plt.savefig(os.path.join(plots_path,f'word_clog_topic{i}.png'))
        plt.close()
