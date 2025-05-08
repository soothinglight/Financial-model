import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv('software_projects_dataset.csv')

# Data preprocessing
def preprocess_data(df):
    # Create label encoders for categorical variables
    le_dict = {}
    categorical_cols = ['project_type', 'industry', 'team_size', 'development_methodology', 
                       'risk_level', 'client_type']
    
    for col in categorical_cols:
        le_dict[col] = LabelEncoder()
        df[col] = le_dict[col].fit_transform(df[col])
    
    # Convert completion_date to numerical (days since earliest date)
    df['completion_date'] = pd.to_datetime(df['completion_date'])
    df['days_since_start'] = (df['completion_date'] - df['completion_date'].min()).dt.days
    
    # Select features for the model
    features = ['project_type', 'industry', 'team_size', 'development_methodology', 
               'risk_level', 'client_type', 'days_since_start', 'project_manager_experience',
               'team_experience', 'initial_cost_estimate', 'actual_cost', 'cost_overrun_percentage',
               'initial_time_estimate', 'actual_time', 'time_overrun_percentage',
               'requirements_stability', 'num_stakeholders', 'stakeholder_satisfaction',
               'technology_complexity', 'num_integrations', 'quality_defects',
               'code_base_size', 'velocity', 'total_defects']
    
    X = df[features]
    y = df['project_success']
    
    return X, y, le_dict

# Main execution
def train_model():
    # Preprocess the data
    X, y, le_dict = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Print model performance
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and preprocessing objects
    joblib.dump(rf_model, 'project_success_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le_dict, 'label_encoders.pkl')
    
    return rf_model, scaler, le_dict

def predict_project_success(project_data, model, scaler, le_dict):
    """
    Make predictions for new project data
    """
    # Preprocess the new data
    for col in project_data.columns:
        if col in le_dict:
            project_data[col] = le_dict[col].transform(project_data[col])
    
    # Scale the features
    project_data_scaled = scaler.transform(project_data)
    
    # Make prediction
    prediction = model.predict(project_data_scaled)
    probability = model.predict_proba(project_data_scaled)
    
    return prediction, probability

if __name__ == "__main__":
    # Train the model
    model, scaler, le_dict = train_model()