import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from project_model import ProjectPredictor
import logging

class ModelValidator:
    def __init__(self, model_path=None):
        self.predictor = ProjectPredictor()
        self.validation_results = {}
        
    def load_validation_data(self, filepath):
        """Load validation dataset"""
        try:
            self.validation_data = pd.read_csv(filepath)
            logging.info(f"Validation data loaded successfully from {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error loading validation data: {str(e)}")
            return False
    
    def validate_model(self):
        """Perform model validation"""
        try:
            # Prepare data
            self.predictor.load_data('software_projects_dataset.csv')
            self.predictor.preprocess_data()
            self.predictor.train_model()
            
            # Make predictions on validation set
            predictions = []
            actuals = self.validation_data['project_success']
            
            for _, row in self.validation_data.iterrows():
                pred = self.predictor.predict(row)
                predictions.append(pred)
            
            # Calculate metrics
            self.validation_results['predictions'] = predictions
            self.validation_results['actuals'] = actuals
            
            # Save incorrect predictions
            self._save_incorrect_predictions(predictions, actuals)
            
            logging.info("Model validation completed successfully")
            return True
        except Exception as e:
            logging.error(f"Error in model validation: {str(e)}")
            return False
    
    def _save_incorrect_predictions(self, predictions, actuals):
        """Save incorrect predictions to CSV"""
        try:
            incorrect_mask = np.array(predictions) != np.array(actuals)
            incorrect_data = self.validation_data[incorrect_mask].copy()
            incorrect_data['predicted'] = np.array(predictions)[incorrect_mask]
            incorrect_data['actual'] = np.array(actuals)[incorrect_mask]
            
            incorrect_data.to_csv('incorrect_predictions.csv', index=False)
            logging.info("Incorrect predictions saved successfully")
        except Exception as e:
            logging.error(f"Error saving incorrect predictions: {str(e)}")
    
    def plot_results(self):
        """Generate validation result plots"""
        try:
            # Confusion Matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(
                self.validation_results['actuals'],
                self.validation_results['predictions']
            )
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig('confusion_matrix.png')
            plt.close()
            
            # Classification Report
            report = classification_report(
                self.validation_results['actuals'],
                self.validation_results['predictions']
            )
            with open('classification_report.txt', 'w') as f:
                f.write(report)
            
            logging.info("Validation plots generated successfully")
            return True
        except Exception as e:
            logging.error(f"Error generating validation plots: {str(e)}")
            return False

if __name__ == "__main__":
    results, incorrect_predictions = validate_model()
