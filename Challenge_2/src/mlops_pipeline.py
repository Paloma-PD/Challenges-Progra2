import mlflow
import mlflow.sklearn
import os
from pathlib import Path

# import our external modules
from nlp import nlp_process, sentimental_analysis
from evaluation import model_evaluate
from model_training import model_training
from prepocessing import load_data_frame, preprocessing_data

# Main function
def main():
    # Defines the directory of our file
    df_path = Path(__file__).parent.resolve() # Convierte la ruta relativa en absoluta, tenía conflicto con las diagonales
    df_path = df_path.parent / 'data/glassdoor_reviews.csv'
    
    # Load the data
    df = load_data_frame(path=df_path)
    # Preprocessing part
    df_text = preprocessing_data(df=df, scaling=True)
    # NLP implmentation
    nlp_process(df_text)
    print('NLP implemented')
    # Sentimental analysis
    sentimental_analysis(df_text)
    print('NSentimental analysis implemented')
    # Model training
    X_train, X_test, y_train, y_test = model_training(df_text)
    
    # Model evaluation
    df_resultados, plots_path = model_evaluate(X_train, y_train, X_test, y_test)

    # Starting an experiment in MLflow
    # Set the base folder to "challenge" (up one level from "src")
    cwd = Path(__file__).parent.resolve() # Convert relative path to absolute, had conflict with slashes
    cwd = cwd.parent / 'mlruns'
    mlflow.set_tracking_uri(cwd)
    mlflow.set_experiment("NLP")

    for i, row in df_resultados.iterrows():
        nombre_modelo = row['Modelo']
        with mlflow.start_run(run_name=nombre_modelo):
            for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
                mlflow.log_metric(metric_name, row[metric_name])
            
            report_text = row['Report']
            mlflow.log_text(report_text, f"classification_report_{nombre_modelo}.txt")
            
            # Record the ROC curve as an image
            mlflow.log_artifact(os.path.join(plots_path,f"roc_curve-{nombre_modelo}.png"))

            # # Record the confusion matrix as an image
            mlflow.log_artifact(os.path.join(plots_path,f"confusion_matrix-{nombre_modelo}.png"))
            
            # Record the model
            mlflow.sklearn.log_model(row['Model'], f"{nombre_modelo}_model")

# Main Execution Block: Code that runs when the script is executed directly
if __name__ == '__main__':
    main()
    