import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
'''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer'''
from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def add_features(self, df):
        df['MA_10'] = df['Adj Close'].rolling(10).mean()
        df['MA_20'] = df['Adj Close'].rolling(20).mean() 
        df['MA_60'] = df['Adj Close'].rolling(60).mean()
        df['Daily Return'] = df['Adj Close'].pct_change()
        return df
        
    
        
    def get_data_transformer_object(self, all_stocks):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            '''numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")'''

            '''preprocessor=Pipeline([
            ('moving_averages', self.add_features, all_stocks)
            #('daily_returns', add_daily_returns)
            ])'''
            prep_ob= self.add_features(all_stocks)
            return prep_ob
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,stock_path,comp1_path, comp2_path, comp3_path):

        try:
            stock=pd.read_csv(stock_path)
            comp1=pd.read_csv(comp1_path)
            comp2=pd.read_csv(comp2_path)
            comp3=pd.read_csv(comp3_path)

            logging.info("Reading all data completed")

            logging.info("Obtaining preprocessing object")

            #preprocessing_obj=self.get_data_transformer_object()

            
            # Concatenate all the stock dataframes into one
            all_stocks = pd.concat([stock, comp1, comp2, comp3], axis=0)
            all_stocks.reset_index(drop=True, inplace=True)
            preprocessing_obj=self.get_data_transformer_object(all_stocks)

            '''input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]'''



            logging.info(
                f"Applying preprocessing object on combined dataframe"
            )

            '''kde_prepr=preprocessing_obj.(all_stocks)'''

            '''kde_arr = np.c_[
                kde_prepr, np.array(all_stocks)
            ]'''
            '''test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]'''

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                preprocessing_obj,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

   