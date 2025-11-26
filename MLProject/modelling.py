import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow. sklearn

mlflow.set_experiment("Earthquake_Tsunami_Prediction_Basic")

def load_preprocessed_data(filepath: str):
    df = pd.read_csv(filepath)
    return df

# Split data into train and test sets
def prepare_data(df: pd.DataFrame, target_col: str = 'tsunami', test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# Train model using MLflow autolog (Basic level)
def train_model_basic():
    df = load_preprocessed_data('earthquake_data_tsunami_preprocessing.csv')
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model. predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
    print("\nModel training completed with autolog!")

if __name__ == "__main__":
    train_model_basic()