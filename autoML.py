import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from pycaret.regression import *
import logging

class MLSystem:
    def __init__(self):
        pass

    def load_data(self, data):
        train = pd.read_csv(data)
        return train
    
    def preprocess_data(self, X):
        reg = setup(
            data = X,
            target = "Rings",
            numeric_features = ["Length",  "Diameter",  "Height",  "Whole weight",  "Whole weight.1",  "Whole weight.2",  "Shell weight"],
            categorical_features=["Sex"],
            normalize=True, normalize_method="zscore",
            polynomial_features=True)   
        model = create_model('lightgbm')
        return model
    
    def tuned_model(self, model, optimizer):
        tuned_dt = tune_model(model, optimize = optimizer)
        final_dt = finalize_model(tuned_dt)        
        return final_dt
    
    def evaluate_model(self, model, dataset):
        test = pd.read_csv(dataset)
        predictions = predict_model(model, data=test)
        submission = predictions[['id', 'prediction_label']]
        submission.rename(columns={'prediction_label': 'Rings'}, inplace=True)
        submission.to_csv('submission_0.csv', index=False)
        rmsle = np.sqrt(mean_squared_log_error(test["Rings"], predictions["prediction_label"]))
        return rmsle
    
    def run_entire_workflow(self, input_data_path):
        try:
            train = self.load_data(input_data_path)
            best_model = self.preprocess_data(train)
            final_dt = self.tuned_model(best_model, 'RMSLE')
            rmsle = self.evaluate_model(final_dt, input_data_path)
            return {'success': True, 'RMSLE': rmsle}
        except Exception as e:
            return {'success': False, 'message': str(e)}