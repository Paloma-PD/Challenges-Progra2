import mlflow
import mlflow.sklearn
import os
from pathlib import Path

# import our external modules
from evaluation import model_evaluate
from model_training import model_training
from prepocessing import load_data_frame, preprocessing_data

# Main function
def main():
    # Defines the directory of our file
    df_path = Path(__file__).parent.resolve() # Convierte la ruta relativa en absoluta, tenía conflicto con las diagonales
    df_path = df_path.parent / 'data/breast-cancer-wisconsin.data.csv'
    
    # Load the data
    df = load_data_frame(path=df_path)
    # Preprocessing part
    X, y = preprocessing_data(df=df, scaling=True)

    # Model training
    X_train, X_test, y_train, y_test, model, y_pred = model_training(X=X, y=y)
    
    # Model evaluation
    accuracy, report, plots_path = model_evaluate(model, X_train, y_train, X_test, y_test, y_pred)

    # Starting an experiment in MLflow
    # Set the base folder to "challenge" (up one level from "src")
    cwd = Path(__file__).parent.resolve() # Convert relative path to absolute, had conflict with slashes
    cwd = cwd.parent / 'mlruns'
    mlflow.set_tracking_uri(cwd)
    mlflow.set_experiment("Breast Cancer Wisconsin")

    with mlflow.start_run():
        
        # Record metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")
        
        # Record the ROC curve as an image
        mlflow.log_artifact(os.path.join(plots_path,"roc_curve.png"))

        # # Record the confusion matrix as an image
        mlflow.log_artifact(os.path.join(plots_path,"confusion_matrix.png"))
        
        # Record the model
        mlflow.sklearn.log_model(model, "random_forest_model")

# Main Execution Block: Code that runs when the script is executed directly
if __name__ == '__main__':
    main()
    