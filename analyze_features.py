import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def analyze():
    print("Loading data...")
    df = pd.read_csv('customerData_500k.csv')
    
    target = 'PurchaseStatus'
    
    # 1. basic correlation with target
    print("\n--- Correlation Analysis ---")
    
    # Encode categoricals for correlation analysis
    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        
    # Handle missing simple for analysis
    imputer = SimpleImputer(strategy='median')
    df_encoded_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)
    
    corr = df_encoded_imputed.corr()[target].sort_values(ascending=False)
    print(corr)
    
    # 2. Feature Importance via Random Forest
    print("\n--- Random Forest Feature Importance ---")
    X = df_encoded_imputed.drop(columns=[target])
    y = df_encoded_imputed[target]
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print(importances)
    
    # Check specific features user asked about
    print("\n--- Specific Feature Check ---")
    for feature in ['PreferredDevice', 'AnnualIncome', 'Gender']:
        if feature in importances['Feature'].values:
            rank = importances[importances['Feature'] == feature].index[0]
            val = importances[importances['Feature'] == feature]['Importance'].values[0]
            print(f"{feature}: Importance={val:.4f}")

if __name__ == "__main__":
    analyze()
