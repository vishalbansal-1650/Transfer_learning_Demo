from src.utils.common import read_yaml
from src.utils.data_mgmt import get_data,get_labels
from src.utils.model import save_model,save_plot,get_model_summary

import argparse
import os
import logging
import random
import pandas as pd
import tensorflow as tf
import time

## To predict given digit is less than 5 or not

STAGE = "Transfer Learning for less-than-5 classifier" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'Transfer_learning_less_than_5.log'), 
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
    NUM_CLASSES = config["params"]["TL_num_classes"]
    EPOCHS = config["params"]["epochs"]
    BATCH_SIZE = config["params"]["batch_size"]
    seed = config["params"]["seed"]
    
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)

    ## preparing data
    (x_train,y_train),(x_valid,y_valid),(x_test,y_test) = get_data(validation_datasize,seed)

    y_train_bin, y_test_bin, y_valid_bin = get_labels([y_train,y_test,y_valid],'less_than_5')

    ### defining logging dir

    logs_dir = config["logs"]["logs_dir"]
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    base_model_path = os.path.join(model_dir_path,'base_model.h5')

    ## Loading base model
    base_model = tf.keras.models.load_model(base_model_path)
    logging.info(f"Base model summary:  \n{get_model_summary(base_model)}")
    logging.info(f"Base model evaluation metrics: {base_model.evaluate(x_test,y_test)}")

    ## freezing the weights of base model
    for layer in base_model.layers[: -1]:
        logging.info(f"trainable status of before freezing weight for {layer.name}:{layer.trainable} ")
        layer.trainable = False
        logging.info(f"trainable status of after freezing weight for {layer.name}:{layer.trainable}")

    base_layer = base_model.layers[: -1]


    ## Define new model using base model weights and compile it
    new_model = tf.keras.models.Sequential(base_layer)
    new_model.add(
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax',name='output_layer')
    )
    
    logging.info(f"{STAGE} model summary: \n{get_model_summary(new_model)}")


    logging.info("Compiling the model")
    new_model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
    logging.info("Model Compiled successfully")
   
    ## training the new model
    VALIDATION_SET = (x_valid, y_valid_bin)
    logging.info("Training the model")
    start_time = time.time()

    model_history = new_model.fit(x=x_train, y=y_train_bin, epochs=EPOCHS, validation_data= VALIDATION_SET, batch_size=BATCH_SIZE)
    end_time = time.time()
   
    logging.info("Model Trained successfully")
    logging.info("Time taken in training model:  %s seconds" %(end_time - start_time))

    logging.info(f"evaluation metrics for new model {new_model.evaluate(x_test, y_test_bin)}") 

    ## saving model file
    model_name ='lt5_clf_model.h5'
    
    save_model(new_model, model_name, model_dir_path)

    ## plotting model performance
    plot_name = 'lt5_model_plot.png'
    plots_dir = config["artifacts"]["plots_dir"]

    plots_dir_path = os.path.join(artifacts_dir, plots_dir)
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
