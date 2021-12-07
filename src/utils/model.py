import os
import io
import time
import logging

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf



def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES,STAGE,seed):

    tf.random.set_seed(seed)

    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28,28],name='inputlayer1'),
        tf.keras.layers.Dense(units=300, name='hiddenlayer1'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=100, name='hiddenlayer2'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', name='outputlayer')
    ]

    logging.info(f"Defining the model architecture for {STAGE}")

    model_clf = tf.keras.models.Sequential(layers=LAYERS)

    logging.info(f"Model Summary : \n{get_model_summary(model_clf)}")

    logging.info("Compiling the model")
    model_clf.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)
    logging.info("Model Compiled successfully")

    return model_clf ## <<< untrained model

def get_model_summary(model)->str:
        """[Get standard console output of model summary and convert it into string ]

        Args:
            model ([tf.keras model]): [untrained keras model]

        Returns:
            str: [model summary in string format]
        """
        with io.StringIO() as stream:
            model.summary(print_fn = lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename


def save_model(model,model_name,model_dir):
    logging.info("Saving the trained model")
    path_to_model = os.path.join(model_dir,model_name)
    model.save(path_to_model)
    logging.info(f"Trained model saved at path : {path_to_model}")


def save_plot(model_history, plot_name, plot_dir):
    logging.info("Saving the model performance plot")
    df = pd.DataFrame(model_history.history)

    df.plot(figsize=(10,8))
    plt.grid(True)
    unique_filename = get_unique_filename(plot_name)
    path_to_plot = os.path.join(plot_dir, unique_filename)

    plt.savefig(path_to_plot)
    logging.info(f"Saved plot at path : {path_to_plot}")
