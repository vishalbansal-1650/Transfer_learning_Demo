from src.utils.common import read_yaml,create_directories
from src.utils.data_mgmt import get_data
from src.utils.model import create_model,save_model,save_plot

import argparse
import os
import logging
import random
import pandas as pd
import tensorflow as tf
import time


STAGE = "Base Model Creation" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'base_model.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="w"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    validation_datasize = config["params"]["validation_datasize"]
    LOSS_FUNCTION = config["params"]["base_loss_function"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["base_num_classes"]
    EPOCHS = config["params"]["epochs"]
    BATCH_SIZE = config["params"]["batch_size"]
    seed = config["params"]["seed"]
    
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)

    ## preparing data
    (x_train,y_train),(x_valid,y_valid),(x_test,y_test) = get_data(validation_datasize,seed)

    ## creating model architecture
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES,STAGE,seed)

    VALIDATION_SET = (x_valid, y_valid)

    ### defining logging dir

    logs_dir = config["logs"]["logs_dir"]
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    create_directories([model_dir_path])

    ## training the model
    logging.info("Training the model on train data set")
    start_time = time.time()

    model_history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, validation_data= VALIDATION_SET, batch_size=BATCH_SIZE)
    end_time = time.time()
   
    logging.info("Model Trained successfully")
    logging.info("Time taken in training base model:  %s seconds" %(end_time - start_time))

    ## saving model file
    model_name ='base_model.h5'
    
    save_model(model, model_name, model_dir_path)

    ## plotting model performance
    plot_name = 'base_model_plot.png'
    plots_dir = config["artifacts"]["plots_dir"]

    plots_dir_path = os.path.join(artifacts_dir, plots_dir)
    create_directories([plots_dir_path])

    save_plot(model_history, plot_name, plots_dir_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
