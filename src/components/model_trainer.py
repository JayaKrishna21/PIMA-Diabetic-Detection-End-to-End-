import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import pickle

import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        
        self.model_trainer_config = ModelTrainerConfig()
        # when initialised, this attains the path from above class

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train,y_train,X_test,y_test = (

                train_array[:,:-1], #[include all the rows,exclude the last column] ---> X_train
                train_array[:,-1], # [include all the rows, last column] ---> y_train
                test_array[:,:-1], #[include all the rows,exclude the last column] ---> X_test
                test_array[:,-1] #[include all the rows, last column] ---> y_test

            )
            #creating dictionary of models

            models = {
                'SVC': SVC(),
                'LogisticRegression': LogisticRegression(),
                'RandomForest': RandomForestClassifier(),
                'DecisionTree': DecisionTreeClassifier(),
                'KNeighbors': KNeighborsClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'GradientBoosting':GradientBoostingClassifier(),
                'NeuralNetwork': MLPClassifier(),
            }

            # Creating a function which stores accuracy scores of models mentioned above iteratively

            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)


            # To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(max(sorted(model_report.values())))
                ]
            
            # best model score

            best_model_score = max(sorted(model_report.values()))

            best_model = models[best_model_name]

            # if best_model_score < 0.6:

            #     #if the best model is having less than 0.6 score, then it is not considered to be a good model at all
            #     raise CustomException("No Best model found",sys)
            
            logging.info(f"Best model on both training and test dataset")

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            ) # dumping the file_path into model.pkl file as model_trainer_config path is given to model.pkl file

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square



        except Exception as e:
            raise CustomException(e,sys)
