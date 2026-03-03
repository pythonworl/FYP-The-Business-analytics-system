import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def preprocess_data(df, target_col='PurchaseStatus'):
    """Preprocesses the data: handles missing values, encoding, and scaling."""
    logging.info("Preprocessing data...")
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numerical and categorical columns
    # We explicitly define them based on our analysis to avoid including ID columns if any accidentally remain, 
    # though we are dropping columns by name in a real scenario, here we rely on the implementation plan's list.
    
    numerical_features = [
        'Age', 'AnnualIncome', 'NumberOfPurchases', 'TimeSpentOnWebsite', 
        'CustomerTenureYears', 'LastPurchaseDaysAgo', 'SessionCount',
        'CustomerSatisfaction', 'DiscountsAvailed', 'LoyaltyProgram'
    ]
    
    # Categorical features removed as per analysis (Gender, Region, etc. were low importance)
    categorical_features = []
    
    # Filter X to only include these features
    X = X[numerical_features]

    # preprocessing for numerical data: impute missing with median, scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features)
        ])
    
    return X, y, preprocessor

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Trains multiple models and evaluates them."""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gaussian NB': GaussianNB()
    }
    
    results = []
    best_model_name = None
    best_score = 0
    best_pipeline = None

    for name, model in models.items():
        logging.info(f"Training {name}...")
        
        # Create a pipeline with preprocessor and model
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        logging.info(f"{name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Select best model based on F1-Score
        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_pipeline = clf

    results_df = pd.DataFrame(results)
    logging.info("\nModel Comparison:\n" + str(results_df))
    
    return best_model_name, best_pipeline, results_df

def main():
    filepath = 'customerData_500k.csv'
    df = load_data(filepath)
    
    if df is not None:
        X, y, preprocessor = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Training models...")
        best_name, best_model, results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
        
        logging.info(f"Best Model: {best_name}")
        
        # Save best model
        model_filename = 'best_churn_model.pkl'
        joblib.dump(best_model, model_filename)
        logging.info(f"Best model saved to {model_filename}")
        
        # Save results to CSV (optional, for record)
        results_df.to_csv('model_comparison_results.csv', index=False)
        logging.info("Comparison results saved to model_comparison_results.csv")

if __name__ == "__main__":
    main()
