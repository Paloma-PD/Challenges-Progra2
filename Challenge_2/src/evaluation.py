import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
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

def model_evaluate(modelos, X_train, y_train, X_test, y_test):
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
        
        # Evaluar el modelo
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='pos')
        recall = recall_score(y_test, y_pred, pos_label='pos')  # Sensibilidad
        f1 = f1_score(y_test, y_pred, pos_label='pos')
        report = classification_report(y_test, y_pred)

        resultados.append({
                'Modelo': nombre,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1
            })

        # Mostrar resultados
        df_resultados = pd.DataFrame(resultados)

        # Calcular la curva ROC
        y_probs = pipeline.predict_proba(X_test)[:, 1]  # Probabilidades de la clase positiva (Maligno)
        
        # Convertir etiquetas de texto a binario
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test).ravel()  # .ravel() para convertir a vector 1D
        fpr, tpr, _ = roc_curve(y_test_bin, y_probs)
        roc_auc = auc(fpr, tpr)

        plots_path = Path(__file__).parent.resolve() # Convierte la ruta relativa en absoluta, tenía conflicto con las diagonales
        plots_path = plots_path.parent / 'plots'
        if not os.path.exists(plots_path):
            # If it doesn't exist, it will create it
            os.makedirs(plots_path)

        # Graficar la curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve - {nombre} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Línea de referencia
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(f"roc_curve-{nombre}.png")
        plt.close()

        # Calcular y mostrar la matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['neg', 'pos'], yticklabels=['neg', 'pos'])
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusión - {nombre}')
        plt.savefig(f"confusion_matrix-{nombre}.png")
        plt.close()
        
        # Imprimir resultados
        print('MÉTRICAS')
        print(nombre)
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1-score: {f1:.2f}')
        print(f'Reporte de Clasificación - {nombre}:')
        print(report)
 
    return df_resultados