import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def model_training(df_text):
    # Definimos las variables a trabajar en el modelo
    X = df_text['pros_lem']  # columna de texto preprocesado
    y = df_text['Flag']      # variable objetivo
    print("\n")
    print("Variaables defined")
    # It is divided into training and test sets (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("\n")
    print("Data splitting")

        # Lista de modelos
    modelos = [
        ('Naive Bayes', MultinomialNB()),
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('Random Forest', RandomForestClassifier(random_state=5)),
        ('SVM', SVC(probability=True)),
        ('Decision Tree', DecisionTreeClassifier())
    ]
    # Evaluar y guardar resultados
    resultados = []

    for nombre, modelo in modelos:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', modelo)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        resultados.append({
            'Modelo': nombre,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, pos_label='pos'),
            'Recall': recall_score(y_test, y_pred, pos_label='pos'),
            'F1-score': f1_score(y_test, y_pred, pos_label='pos')
        })
    # Convertir en dataframe
    df_modelos = pd.DataFrame(resultados)
    
    return X_train, X_test, y_train, y_test, df_modelos