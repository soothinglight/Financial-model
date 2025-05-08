import pandas as pd
import joblib
from datetime import datetime

# Load the saved model and preprocessing objects
model = joblib.load('project_success_model.pkl')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')

# Example new project data (maintaining feature order)
new_project = pd.DataFrame({
    'project_type': ['Web Application'],
    'industry': ['Technology'],
    'team_size': ['Medium (6-15)'],
    'development_methodology': ['Agile'],
    'risk_level': ['Medium'],
    'client_type': ['External'],
    'days_since_start': [(pd.to_datetime('2024-12-31') - pd.to_datetime('2020-01-01')).days],
    'project_manager_experience': [5],
    'team_experience': [4.5],
    'initial_cost_estimate': [200000],
    'actual_cost': [220000],
    'cost_overrun_percentage': [10],
    'initial_time_estimate': [60],
    'actual_time': [65],
    'time_overrun_percentage': [8.33],
    'requirements_stability': [75],
    'num_stakeholders': [8],
    'stakeholder_satisfaction': [7.5],
    'technology_complexity': [6],
    'num_integrations': [4],
    'quality_defects': [3],
    'code_base_size': [50],
    'velocity': [2.5],
    'total_defects': [150]
})

# Make prediction
from project_success_predictor import predict_project_success
prediction, probability = predict_project_success(new_project, model, scaler, le_dict)

print("\nProject Success Prediction:", "Success" if prediction[0] else "Failure")
print("Success Probability: {:.2f}%".format(probability[0][1] * 100))
print("Failure Probability: {:.2f}%".format(probability[0][0] * 100))
print("Prediction Probability:", probability[0])