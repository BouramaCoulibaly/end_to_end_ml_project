import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    row_data_path:str = os.path.join("artifacts", "data.csv")
    
    
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data Ingestion method or component")
        
        try:
            data = pd.read_csv('archive//StudentsPerformance.csv')
            logging.info("Read the dataset as a dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            data.to_csv(self.ingestion_config.row_data_path,index = False, header = True)
            train_set,test_set=train_test_split(data,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = True, header = True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e)
        
        
if __name__=='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    