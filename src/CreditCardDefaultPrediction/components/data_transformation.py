# data_transformation.py

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.CreditCardDefaultPrediction.logger import logging
from src.CreditCardDefaultPrediction.exception import CustomException
from src.CreditCardDefaultPrediction.utils.utils import Utils
from src.CreditCardDefaultPrediction.utils.data_processor import CSVProcessor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, LabelEncoder

import warnings
warnings.filterwarnings("ignore")


@dataclass
class DataTransformationConfig:
    """
    This is configuration class for Data Transformation
    """
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    This class handles Data Transformation
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.utils = Utils()
        self.csv_processor = CSVProcessor()

    def transform_data(self):
        try:
            numerical_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                              'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            
            num_pipeline = Pipeline(
                steps=[
                    # ('outlier', OutlierTransformer()),
                    ('scaler', StandardScaler()),
                    
                ])

            cat_pipeline = Pipeline(
                steps=[
                    # ('onehotencoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories='auto', drop='first')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ], remainder='passthrough')

            return preprocessor
        
        except Exception as e:
            logging.error("Exception occured in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, val_path):

        def replace_categories(df):
            df = df.replace({'SEX': {1: 'MALE', 2: 'FEMALE'},
                        'EDUCATION': {1: 'graduate_school', 2: 'university', 3: 'high_school', 4: 'others'},
                        'MARRIAGE': {1: 'married', 2: 'single', 3: 'others'}})
            logging.info("Numerical categories has been converted to string values")
            return df
        
        def update_column_values(df):
            # Modify 'EDUCATION' column
            df['EDUCATION'] = df['EDUCATION'].map({0: 4, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4})            

            # Modify 'MARRIAGE' column
            df['MARRIAGE'] = df['MARRIAGE'].map({0: 3, 1: 1, 2: 2, 3: 3})

            logging.info("EDUCATION & MARRIAGE column's values are merged which has lesser counts")
            return df
    
        try:
            train_df = self.utils.run_data_pipeline(self.csv_processor, path=None, filename=train_path)
            test_df = self.utils.run_data_pipeline(self.csv_processor, path=None, filename=test_path)
            val_df = self.utils.run_data_pipeline(self.csv_processor, path=None, filename=val_path)
            
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            logging.info(f'Validation Dataframe Head : \n{val_df.head().to_string()}')

            # Handle imbalance data
            train_df = train_df.drop(columns=['ID'], axis=1)
            train_df = self.utils.smote_balance(train_df)

            test_df = test_df.drop(columns=['ID'], axis=1)
            test_df = self.utils.smote_balance(test_df)

            val_df = val_df.drop(columns=['ID'], axis=1)
            val_df = self.utils.smote_balance(val_df)
            
            # Modify column data
            train_df = update_column_values(train_df)
            test_df = update_column_values(test_df)
            val_df = update_column_values(val_df)

            # Replace categories
            # train_df = replace_categories(train_df)
            # test_df = replace_categories(test_df)
            
            target_column_name = 'default.payment.next.month'
            drop_columns = [target_column_name, 'ID']



            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)if 'ID' in train_df.columns else train_df
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)if 'ID' in test_df.columns else test_df
            target_feature_test_df = test_df[target_column_name]

            input_feature_val_df = val_df.drop(columns=drop_columns, axis=1)if 'ID' in val_df.columns else val_df
            target_feature_val_df = val_df[target_column_name]

            

            # Apply transformation
            preprocessing_obj = self.transform_data()
            preprocessing_obj.fit(input_feature_train_df)
            
            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df)
            
            input_feature_train_arr_df = pd.DataFrame(input_feature_train_arr, columns=preprocessing_obj.get_feature_names_out())
            input_feature_test_arr_df = pd.DataFrame(input_feature_test_arr, columns=preprocessing_obj.get_feature_names_out())
            input_feature_val_arr_df = pd.DataFrame(input_feature_val_arr, columns=preprocessing_obj.get_feature_names_out())

            logging.info("Applying preprocessing object on training, vdalidation and testing datasets")

            train_df = pd.concat([input_feature_train_arr_df, target_feature_train_df], axis=1)
            test_df = pd.concat([input_feature_test_arr_df, target_feature_test_df], axis=1)
            val_df = pd.concat([input_feature_val_arr_df, target_feature_val_df], axis=1)

            logging.info(f'Processed Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Processed Test Dataframe Head : \n{test_df.head().to_string()}')
            logging.info(f'Processed Validation Dataframe Head : \n{val_df.head().to_string()}')

            self.utils.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_df,
                test_df,
                val_df
            )

        except Exception as e:
            logging.error("Exception occured in Initiate Data Transformation")
            raise CustomException(e, sys)