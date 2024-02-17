from src.CreditCardDefaultPrediction.components.data_ingestion import DataIngestion
from src.CreditCardDefaultPrediction.components.data_transformation import DataTransformation
from src.CreditCardDefaultPrediction.components.model_trainer import ModelTrainer
#from src.CreditCardDefaultPrediction.components.model_evaluation import ModelEvaluation

import os
import sys
from src.CreditCardDefaultPrediction.logger import logging
from src.CreditCardDefaultPrediction.exception import CustomException
import pandas as pd

obj=DataIngestion()
train_data_path, test_data_path, val_data_path = obj.initiate_data_ingestion()


data_transformation = DataTransformation()
train_arr, test_arr, val_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path, val_data_path)


model_trainer_obj=ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr,test_arr)

#model_eval_obj = ModelEvaluation()
#model_eval_obj.initiate_model_evaluation(train_arr,test_arr)