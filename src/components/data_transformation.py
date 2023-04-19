import sys
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_transformation_object(self):
        try:
            logging.info('data transformation initiated')
            
            numerical_columns=['carat','depth','table','x','y','z']
            cat_columns=['cut','color','clarity']
            
            cut_categories=['Premium','Very Good','Ideal','Good','Fair']
            color_catgories=['F','J','G','E','D','H','I']
            clarity_catgories=['VS2','SI2','VS1','SI1','IF','VVS2','VVS1','I1']
            
            logging.info('pipeline initiated')
            
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                    ])
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OrdinalEncoder(categories=[cut_categories,color_catgories,clarity_catgories])),
                ('scaler',StandardScaler())
            ])
            col_transformer=ColumnTransformer(
                transformers=[
                ('num_pipeline',num_pipeline,numerical_columns),
                ('catgorical_columns',cat_pipeline,cat_columns)
                
            ])
            return col_transformer
            logging.info('pipeline completed')
        except Exception as e:
            logging.info('Error  in data transformation ')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        logging.info("Initiate data transformation object")
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train_df and test_df as pandas DataFrame")
            logging.info('Getting Data transformation object')
            preprocessing_obj=self.get_transformation_object()
            
            target_column='price'
            drop_columns=[target_column,'id']
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column]
            print(preprocessing_obj)
            logging.info(f'Train DataFrame head:\n{input_feature_train_df.head().to_string()}')
            logging.info(f'Test DataFrame head:\n{input_feature_test_df.head().to_string()}')
            
            input_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('Applying preprocessing object on training and testing datasets')
            
            train_arr=np.c_[input_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessing_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_file_path
            )
            
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)
            
    