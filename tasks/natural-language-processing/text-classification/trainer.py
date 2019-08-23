"""
Example of text classification with MLflow using Keras to classify various classes in the dataset.
"""

from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 
from __future__ import unicode_literals

import traceback
import numpy as np
import json
import click

import mlflow
import mlflow.keras

mlflow.keras.autolog()



from model import textcnn_model
from utils import textcnn_preprocess    
    

@click.command(help="Trains an Keras model on text dataset to perform text classification."
                    "The model and its metrics are logged with mlflow.")
@click.option("--config_file",type=click.STRING, default='./config.json', 
                help="Path of the config file required for model to execute")                     
def run(config_file):
    try:
        with open(config_file) as f:
            config = json.load(f)
    except:
        traceback.print_exc()
    
    try:
        if config['train']:

            client = mlflow.tracking.MlflowClient()
            

            training_info,model, config = do_train(config,client)

            config['model_path'] = "runs:/"+ training_info.info.run_id + "/model"
            print('Trained',config['model_path'])

            mlflow.keras.log_model( model,"model",custom_objects=config)
        else:
            
            # score = predict(config['model_path'])
            print('Testing')

    except Exception as e:
        traceback.print_exc()




def do_train(config,client):
    """
     This function is responsible for pre-processing the raw text data and further initialze the model based on the configs and execute training.
     Metrics are logged after every epoch. The logger keeps track of the best model based on the
     validation metric. At the end of the training, the best model is logged with MLflow.
    """
    try:
        with mlflow.start_run() as run:

            # setting log identifier
            client.set_tag(run.info.run_id, "experiment_name", config['experiment_name'])

            if config['model_type'] == 'textcnn':
                # Preprocessing
                X_train, X_test, y_train, y_test, vocab_size = textcnn_preprocess.pre_processing(config)
                config['vocab_size'] = vocab_size
                # Defining Keras model
                model = textcnn_model.build(config)
                # Initiate training
                history = model.fit(X_train, y_train,
                                epochs = config['epochs'],
                                verbose=True,
                                validation_data=(X_test, y_test),
                                batch_size= config['batch_size'])


    except Exception as e:
         traceback.print_exc()

    return client.get_run(run.info.run_id) ,model,config




if __name__ == '__main__':
    run()        