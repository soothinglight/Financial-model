from validate_model import ModelValidator

# Initialize validator
validator = ModelValidator()

# Load validation data and run validation
validator.load_validation_data('validation_dataset.csv')
validator.validate_model()

# Generate validation plots
validator.plot_results()