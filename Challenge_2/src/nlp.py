import gensim
from gensim import corpora
from gensim.models import LdaModel

# Definimos la función para trabajar NLP
def nlp_process(df_text):
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
