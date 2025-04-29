
from sklearn.model_selection import train_test_split

def model_training(df_text):
    # Definimos las variables a trabajar en el modelo
    X = df_text['pros_lem']  # columna de texto preprocesado
    y = df_text['Flag']      # variable objetivo
    print("\n")
    print("Variables defined")
    # It is divided into training and test sets (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("\n")
    print("Data splitting")
    return X_train, X_test, y_train, y_test