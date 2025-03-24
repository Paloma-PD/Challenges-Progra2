import os
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def model_evaluate(model, X_train, y_train, X_test, y_test, y_pred):
    # METRICS
    print("\n")
    print("METRICS")
    accuracy = accuracy_score(y_test, y_pred) * 100 # to percentage
    print(f"{model}: Accuracy={accuracy:.2f} %")
    precision = precision_score(y_test, y_pred) * 100
    print(f'{model}: Precision: {precision:.2f} %')
    recall = recall_score(y_test, y_pred) * 100  # Sensibilidad
    print(f'{model}: Recall: {recall:.2f} %')
    f1 = f1_score(y_test, y_pred) * 100
    print(f'{model}:F1-score: {f1:.2f} %')
    report = classification_report(y_test, y_pred)
    print("\n")

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