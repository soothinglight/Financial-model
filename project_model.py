import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='project_predictor.log'
)

class ProjectPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.features = None
        self.target = None
        
    def load_data(self, filepath):
        """Load and prepare the dataset"""
        try:
            self.data = pd.read_csv(filepath)
            logging.info(f"Successfully loaded data from {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data including handling missing values and encoding"""
        try:
            # Handle missing values
            self.data = self.data.fillna(self.data.mean(numeric_only=True))
            
            # Convert categorical variables
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            self.data = pd.get_dummies(self.data, columns=categorical_columns)
            
            # Separate features and target
            self.features = self.data.drop(['project_success', 'project_cost'], axis=1)
            self.target = self.data['project_success']
            
            # Scale features
            self.features_scaled = self.scaler.fit_transform(self.features)
            
            logging.info("Data preprocessing completed successfully")
            return True
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            return False
    
    def train_model(self):
        """Train the model with cross-validation"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.features_scaled, 
                self.target, 
                test_size=0.2, 
                random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            
            # Model evaluation
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"Model trained successfully. MSE: {mse:.4f}, R2: {r2:.4f}")
            logging.info(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return True
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            return False
    
    def predict(self, project_data):
        """Make predictions for new project data"""
        try:
            # Prepare input data
            project_df = pd.DataFrame([project_data])
            project_processed = self.preprocess_project_data(project_df)
            
            # Make prediction
            prediction = self.model.predict(project_processed)
            
            logging.info(f"Prediction made successfully: {prediction[0]:.2f}")
            return prediction[0]
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return None
    
    def preprocess_project_data(self, project_df):
        """Preprocess new project data"""
        try:
            # Apply same preprocessing as training data
            project_processed = self.scaler.transform(project_df)
            return project_processed
        except Exception as e:
            logging.error(f"Error in preprocessing project data: {str(e)}")
            return None
