from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def model_training(X,y):
    # It is divided into training and test sets (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("\n")
    print("Data splitting")

    # Train a RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print('Trained model')

    # Generate predictions with the model
    y_pred = model.predict(X_test)
    
    return X_train, X_test, y_train, y_test, model, y_pred