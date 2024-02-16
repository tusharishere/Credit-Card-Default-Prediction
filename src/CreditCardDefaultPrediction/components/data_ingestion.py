# data_ingestion.py

import sys
import os
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from src.CreditCardDefaultPrediction.logger import logging
from src.CreditCardDefaultPrediction.exception import CustomException
from src.CreditCardDefaultPrediction.utils.data_processor import DBProcessor
from src.CreditCardDefaultPrediction.utils.utils import Utils

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


@dataclass
class DataIngestionConfig:
    """
    This is configuration class for Data Ingestion
    """
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    val_data_path: str = os.path.join("artifacts", "val.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """
    This class handled Data Ingestion
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.utils = Utils()
        self.db_processor = DBProcessor()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")

        try:
            # Read raw dataset from MongoDB database
            data=pd.read_csv(Path(os.path.join("notebooks/data","UCI_Credit_Card.csv")))
            logging.info(" i have read dataset as a df")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw dataset is saved in artifacts folder")

            train_data, test_data = train_test_split(data, test_size=0.40, random_state=42)
            val_data, test_data = train_test_split(test_data, test_size=0.50, random_state=42)
            logging.info("Dataset is splitted into Train & Test data")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            val_data.to_csv(self.ingestion_config.val_data_path, index=False)
            logging.info("Train & Test dataset are saved in artifacts folder")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.val_data_path
            )

        except Exception as e:
            logging.error("Exception occuring during data ingestion")
            raise CustomException(e, sys)