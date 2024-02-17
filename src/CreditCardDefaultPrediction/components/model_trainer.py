# model_trainer.py

import os
import sys
import numpy as np
from dataclasses import dataclass
from src.CreditCardDefaultPrediction.logger import logging
from src.CreditCardDefaultPrediction.exception import CustomException
from src.CreditCardDefaultPrediction.utils.utils import Utils

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    """
    This is configuration class FOR Model Trainer
    """
    trained_model_obj_path: str = os.path.join("artifacts", "model.pkl.gz")
    trained_model_report_path: str = os.path.join('artifacts', 'model_report.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = Utils()

    def initiate_model_training(self, train_dataframe, test_dataframe):
        try:
            logging.info("Splitting Dependent and Independent features from train and validation & test dataset")

            X_train, y_train, X_test, y_test = (
                train_dataframe.iloc[:, :-1],
                train_dataframe.iloc[:, -1],
                test_dataframe.iloc[:, :-1],
                test_dataframe.iloc[:, -1])
            
            models = {
                     'XGBClassifier': XGBClassifier(),
                     'DecisionTree': DecisionTreeClassifier(),
                     'SVC': SVC(),
                     'LogisticRegression': LogisticRegression(),
                     'KNeighborsClassifier': KNeighborsClassifier(),
                     'GradientBoostingClassifier': GradientBoostingClassifier(),
                    'RandomForestClassifier': RandomForestClassifier()
                 }
            
            """models = {
                    'XGBClassifier': (XGBClassifier(), {
                                'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                                'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
                                'min_child_weight': [1, 3, 5, 7],
                                'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
                                'colsample_bytree': [0.3, 0.4, 0.5, 0.7]}),
                    'DecisionTreeClassifier': (DecisionTreeClassifier(), {
                                         'max_depth': [20, 30, 50, 100], 
                                         'min_samples_split': [0.1, 0.2, 0.4]
                                         }),
                    'SVC': (SVC(), {
                            'C': [2, 5, 10], 
                            'kernel': ['rbf', 'poly']}),
                    'LogisticRegression': (LogisticRegression(), {
                                            'penalty': ['l1', 'l2'], 
                                            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}),
                    'KNeighborsClassifier': (KNeighborsClassifier(), {
                                        'n_neighbors': [2, 5, 15, 30],
                                        'weights': ['uniform', 'distance'],
                                        'metric': ['minkowski', 'euclidean', 'manhattan'],
                                        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]}),
                    'GradientBoostingClassifier': (GradientBoostingClassifier(), { 
                                        'n_estimators': [5, 10, 15],
                                        'max_depth': [5, 10, 20],
                                        'min_samples_split': [10, 20, 30],
                                        'min_samples_leaf': [10, 20, 30],
                                        'max_features': [5, 10, 40]
                                        }),
                    'RandomForestClassifier': (RandomForestClassifier(),  {
                                    'n_estimators': [5, 10, 15, 30], 
                                    'max_depth': [5, 20, 50, 100]})
                    }"""

            # model evaluation without any hyper-paramter tuning            
            best_model = self.utils.evaluate_models(models, X_train, y_train, X_test, y_test, metric="roc_auc")
            
            # model evaluation along with hyper-paramter tuning
            #best_model = self.utils.evaluate_models_with_hyperparameter(models, X_train, y_train, X_test, y_test, metric="roc_auc", verbose=0)
            
            self.utils.save_object(
                 file_path=self.model_trainer_config.trained_model_obj_path,
                 obj=best_model
            )       

        except Exception as e:
            raise CustomException(e, sys)