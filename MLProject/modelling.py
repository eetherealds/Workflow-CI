import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
import mlflow.sklearn

mlflow. set_experiment("Earthquake_Tsunami_Prediction_Basic")

def load_preprocessed_data(filepath: str):
    df = pd.read_csv(filepath)
    return df

def prepare_data(df: pd.DataFrame, target_col: str = 'tsunami', test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_model_basic(data_path='earthquake_data_tsunami_preprocessing.csv', 
                      target_col='tsunami', test_size=0.2, random_state=42,
                      n_estimators=100, max_depth=10):
    df = load_preprocessed_data(data_path)
    X_train, X_test, y_train, y_test = prepare_data(df, target_col, test_size, random_state)
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model. predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
    print("\nModel training completed with autolog!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Earthquake Tsunami prediction model')
    parser.add_argument('--data_path', type=str, 
                        default='earthquake_data_tsunami_preprocessing.csv',
                        help='Path to preprocessed dataset')
    parser. add_argument('--target_col', type=str, default='tsunami',
                        help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size (0-1)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in Random Forest')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum depth of trees')
    
    args = parser.parse_args()
    
    train_model_basic(
        data_path=args.data_path,
        target_col=args.target_col,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )