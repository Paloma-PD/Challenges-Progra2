import os
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

def model_evaluate(modelos, X_train, y_train, X_test, y_test, y_pred, df_modelos):
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
            'F1-score': f1_score(y_test, y_pred, pos_label='pos'),
            'y_pred': y_pred
        })
    # Convertir en dataframe
    df_modelos = pd.DataFrame(resultados)
    for nombre_modelo, modelo in modelos:
        # Calcular la curva ROC
        # Convertir etiquetas de texto a binario
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test).ravel()  # .ravel() para convertir a vector 1D
        fpr, tpr, _ = roc_curve(y_test_bin, y_probs)
        roc_auc = auc(fpr, tpr)

        # Graficar la curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve - {nombre_modelo} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Línea de referencia
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(f"roc_curve-{nombre_modelo}.png")
        plt.close()

    plots_path = Path(__file__).parent.resolve() # Convierte la ruta relativa en absoluta, tenía conflicto con las diagonales
    plots_path = plots_path.parent / 'plots'
    if not os.path.exists(plots_path):
        # If it doesn't exist, it will create it
        os.makedirs(plots_path)

    # Graph the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.savefig(os.path.join(plots_path,'confusion_matrix.png'))
    plt.close()
    
    # ROC score
    y_probs = model.predict_proba(X_test)[:, 1]  # Probabilidades de la clase positiva (Maligno)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Graph the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Línea de referencia
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(plots_path, 'ROC_curve.png'))
    plt.close()
    return accuracy, report, plots_path