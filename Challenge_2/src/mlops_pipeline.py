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
    print("\n")
    # Este diccionario almacenará resultados por columna
    resultados_por_columna = {}
    # Selección de columnas a trabajar
    columnas_analisis = ['headline', 'pros', 'cons']

    for columna in columnas_analisis:
        # Preprocessing part
        df_text = preprocessing_data(df=df, columna=columna)
        # NLP implmentation
        nlp_process(df_text, columna=columna)
        print('NLP implemented')
        # Sentimental analysis
        sentimental_analysis(df_text, columna=columna)
        print('Sentimental analysis implemented')
        # Model training
        X_train, X_test, y_train, y_test = model_training(df_text)
        
        # Model evaluation
        df_resultados, plots_path = model_evaluate(X_train, y_train, X_test, y_test, columna)

        # Starting an experiment in MLflow
        # Set the base folder to "challenge" (up one level from "src")
        cwd = Path(__file__).parent.resolve() # Convert relative path to absolute, had conflict with slashes
        cwd = cwd.parent / 'mlruns'
        mlflow.set_tracking_uri(cwd)
        mlflow.set_experiment("NLP")

        for i, row in df_resultados.iterrows():
            nombre_modelo = row[f'Modelo_{columna}']
            with mlflow.start_run(run_name=nombre_modelo):
                for metric_name in [f'Accuracy_{columna}', f'Precision_{columna}', f'Recall_{columna}', f'F1-score_{columna}']:
                    mlflow.log_metric(metric_name, row[metric_name])
                
                report_text = row[f'Report_{columna}']
                mlflow.log_text(report_text, f"classification_report_{nombre_modelo}_{columna}.txt")
                
                # Record the ROC curve as an image  
                mlflow.log_artifact(os.path.join(plots_path,f"roc_curve-{nombre_modelo}-{columna}.png"))

                # # Record the confusion matrix as an image
                mlflow.log_artifact(os.path.join(plots_path,f"confusion_matrix-{nombre_modelo}-{columna}.png"))
                
                # Record the model
                mlflow.sklearn.log_model(nombre_modelo, f"{nombre_modelo}_model")
        # Guardamos resultados
        resultados_por_columna[columna] = (df_resultados, plots_path)

# Main Execution Block: Code that runs when the script is executed directly
if __name__ == '__main__':
    main()
    