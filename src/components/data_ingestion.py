import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import yfinance as yf
from datetime import datetime

from src.components.data_transformtion import DataTransformation, DataTransformationConfig

#from sklearn.model_selection import train_test_split
from dataclasses import dataclass

tick = 'TSLA'
comp1 = 'TM'
comp2 = 'GM'
comp3 = 'F'
start_date = datetime(2017,1,1)
end_date = datetime.today()

@dataclass
class DataIngestionConfig:
    stock_data_path: str=os.path.join('artifacts',"stock.csv")
    all_data_path: str=os.path.join('artifacts',"all.csv")
    comp1_data_path: str=os.path.join('artifacts',"comp1data.csv")
    comp2_data_path: str=os.path.join('artifacts',"comp2data.csv")
    comp3_data_path: str=os.path.join('artifacts',"comp3data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the stock data ingestion method or component")
        try:
            stock= yf.download(tick, start_date , end_date)
            stock.reset_index(inplace = True)
            stock['Company'] = tick
            stock.to_csv(self.ingestion_config.stock_data_path,index=False,header=True)
            logging.info('Read the dataset as dataframe') 

            comp1_stock= yf.download(comp1, start_date , end_date)
            comp1_stock.reset_index(inplace = True)
            comp1_stock['Company'] = comp1
            comp1_stock.to_csv(self.ingestion_config.comp1_data_path,index=False,header=True)
            logging.info('Read the 1st competitor dataset as dataframe')

            comp2_stock= yf.download(comp2, start_date , end_date)
            comp2_stock.reset_index(inplace = True)
            comp2_stock['Company'] = comp2
            comp2_stock.to_csv(self.ingestion_config.comp2_data_path,index=False,header=True)
            logging.info('Read the 2nd competitor dataset as dataframe')

            comp3_stock= yf.download(comp3, start_date , end_date)
            comp3_stock.reset_index(inplace = True)
            comp3_stock['Company'] = comp3
            comp3_stock.to_csv(self.ingestion_config.comp3_data_path,index=False,header=True)
            logging.info('Read the 3rd competitor dataset as dataframe')
            #concat call from transformation
            #os.makedirs(os.path.dirname(self.ingestion_config.all_data_path), exist_ok=True)
            #all = 
            #all.to_csv(self.ingestion_config.all_data_path,index=False,header=True)

            logging.info("Transformation Intitated")
            #train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            #train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            #test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            

            return(
                self.ingestion_config.stock_data_path,
                self.ingestion_config.comp1_data_path,
                self.ingestion_config.comp2_data_path,
                self.ingestion_config.comp3_data_path,




            )
        except Exception as e:
            raise CustomException(e,sys)
        

    def concatstocks(self, df):
        logging.info("Saving Concated Stocks Dataframe")

        try:
            os.makedirs(os.path.dirname(self.ingestion_config.all_data_path), exist_ok=True)
            df = df.to_csv(self.ingestion_config.all_data_path,index=False,header=True)
            logging.info("Ingestion complete")
            return df
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    stock, comp1, comp2, comp3 = obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    all, allp=data_transformation.initiate_data_transformation(stock, comp1, comp2, comp3)
    all = obj.concatstocks(all)