from cProfile import label
import gzip
import mlflow
import mlflow.sklearn
import dagshub
import pickle
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from time import time

from src.CreditCardDefaultPrediction.exception import CustomException
from src.CreditCardDefaultPrediction.logger import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from src.CreditCardDefaultPrediction.utils.mlflow_setup import setup_mlflow_experiment
import src.CreditCardDefaultPrediction.utils.mlflow_setup as mlflow_setup


class Utils:

    def __init__(self) -> None:
        self.MODEL_REPORT = {}

    def save_object(self, file_path: str, obj):
        """
        The save_object function saves an object to a file.

        :param file_path: str: Specify the path where the object will be saved
        :param obj: obj: Pass the object to be saved
        :return: None
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with gzip.open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
                logging.info(f"File is saved at '{file_path}' successfully.")
        except Exception as e:
            logging.error("Exception occurred during saving object")
            raise CustomException(e, sys)

    def load_object(self, file_path: str):
        """
        The load_object function loads a pickled object from the file_path.

        :param file_path: str: Specify the path of the file that we want to load
        :return: None
        """
        try:
            with gzip.open(file_path, "rb") as file_obj:
                logging.info(f"File at '{file_path}' has been successfully loaded.")
                return pickle.load(file_obj)                
        except Exception as e:
            logging.error("Exception occurred during loading object")
            raise CustomException(e, sys)

    def delete_object(self, file_path: str):
        """
        The delete_object function deletes a file from the local filesystem.
        
        :param file_path: str: Specify the path of the file to be deleted
        :return: None
        """
        try:
            # Check if the file exists
            if os.path.exists(file_path):
                # Remove the file
                os.remove(file_path)
                logging.info(f"File at '{file_path}' has been successfully deleted.")
            else:
                logging.info(f"File at '{file_path}' does not exist.")
        except Exception as e:
            logging.error("Exception occurred during deleting object")
            raise CustomException(e, sys)

    def run_data_pipeline(self, data_processor, path: str = None, filename: str = None, **kwargs):
        """
        The run_data_pipeline function is a wrapper function that takes in the data_processor object and 
            calls its process_data method.
        
        :param data_processor: obj: Pass in the data processor class that will be used to process the data
        :param path: str: Specify the path to the data file
        :param filename: str: Specify the name of the file that will be used for processing
        :param **kwargs: Pass a variable number of keyword arguments to the function
        :return: The processed data
        """
        return data_processor.process_data(path, filename, **kwargs)
    
    def timer(self, start_time=None):
        """
        The timer function is a simple function that takes in an optional start_time argument. 
        If no start_time is provided, the current time will be returned. If a start_time is provided, 
        the difference between the current time and the given start_time will be printed.
        
        :param start_time: Datetime: Start the timer
        :return: None
        """
        
        if not start_time:
            start_time = datetime.now()
            return start_time
        
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            logging.info("Model took: {} hours {} minutes and {} seconds".format(thour, tmin, round(tsec, 2)))

    def predict(self, model_name, model, features, label):
        """
        The predict function predicts the labels using the model and features provided.

        :param model: Pass the model to be used for prediction
        :param features: DataFrame: Features to be used for prediction
        :param label: DataFrame: Label to be used for prediction
        :return: dict: A dictionary with the model, accuracy score, f-score, precision score and recall score
        """

        accuracy = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='accuracy').mean(), 2)
        f1 = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='f1').mean(), 2)
        precision = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='precision').mean(), 2)
        recall = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='recall').mean(), 2)
        roc_auc = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='roc_auc').mean(), 2)

        self.MODEL_REPORT[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc}
        
    def evaluate_models(self, models: dict, train_features, train_label, test_features, test_label, metric='roc_auc', verbose=0):
        """
        The evaluate_models function takes in a dictionary of models and their hyperparameters,
        train_features, train_label, test_features, and test_label. It then uses RandomizedSearchCV
        to find the best model for each model passed into it.

        :param models: dict: Models and their hyperparameters
        :param train_features: DataFrame: Training features
        :param train_label: DataFrame: Training labels
        :param test_features: DataFrame: Test features
        :param test_label: DataFrame: Test labels
        :param metric: str: Evaluation metric (default='roc_auc')
        :return: tuple: The best model
        """
        np.random.seed(42)
        self.MODEL_REPORT = {}
        TRAINING_SCORE = {}

        for model_name, (model, params) in models.items():
            logging.info("\n\n========================= {} =======================".format(model_name))

            # Perform hyperparameter tuning using RandomizedSearchCV
            random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=5, scoring=metric, cv=5, n_jobs=-1)
            random_search.fit(train_features, train_label)

            # Get the best model
            best_model = random_search.best_estimator_

            # Evaluate the best model on the test set
            best_model_pred = best_model.predict(test_features)
            if metric == 'roc_auc':
                score = roc_auc_score(test_label, best_model_pred)
            elif metric == 'accuracy':
                score = accuracy_score(test_label, best_model_pred)
            else:
                raise ValueError("Unsupported metric")

            # Record model performance in MODEL_REPORT
            self.MODEL_REPORT[model_name] = {
                'model': best_model,
                metric: score,
                'params': random_search.best_params_
            }
            TRAINING_SCORE[model_name] = random_search.best_score_

        logging.info("Model Report: {}".format(self.MODEL_REPORT))
        logging.info("Training Scores: {}".format(TRAINING_SCORE))

        # Find the best model based on the given metric
        best_model_name = max(self.MODEL_REPORT, key=lambda x: self.MODEL_REPORT[x][metric])
        best_model = self.MODEL_REPORT[best_model_name]['model']

        logging.info("BEST MODEL REPORT: {}".format(self.MODEL_REPORT[best_model_name]))

        return best_model

    def smote_balance(self, data):
        """
        The smote_balance function takes in a dataframe and returns the same dataframe with SMOTE resampling applied.
        
        :param data: DataFrame: Pass in the dataframe
        :return: DataFrame: Dataframe with the same number of rows as the original dataset, but now there are an equal number of 0s and 1s in the target column
        """
        
        target_column_name = 'default.payment.next.month'
        sm = SMOTE(sampling_strategy='minority', random_state=42)
        
        logging.info('Dataset shape prior resampling: {}'.format(data.shape[0]))
        X_resampled, y_resampled = sm.fit_resample(X=data.drop(columns=target_column_name), y=data[target_column_name])
        data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        logging.info('Dataset shape after resampling: {}'.format(data.shape[0]))
        return data


if __name__ == "__main__":
    logging.info("Demo logging activity")

    utils = Utils()
    utils.save_object(os.path.join('logs', 'utils.pkl'), utils)
    utils.load_object(os.path.join('logs', 'utils.pkl'))
    utils.delete_object(os.path.join('logs', 'utils.pkl'))
