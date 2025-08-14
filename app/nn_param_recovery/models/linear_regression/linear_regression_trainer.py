# Victim network based on part 2 of tutorial here:
# https://dev.mrdbourke.com/tensorflow-deep-learning/01_neural_network_regression_in_tensorflow/

import yaml
import math
import sys
import time
import pandas as pd
from absl import app
from absl import flags
from absl import logging
from sklearn.model_selection import train_test_split

import tensorflow as tf
import os
import numpy as np
import random

SEED = 0

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)

NUM_EPOCHS = 50

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", NUM_EPOCHS, "number of epochs to train the model.")
flags.DEFINE_boolean("save_tf_model", False, "store the original unconverted tf model.")

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODELS_DIR, "models")

def load_data():
    insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
    #print(insurance.head())

    # Turn all categories into numbers
    insurance_one_hot = pd.get_dummies(insurance)
    #print(insurance_one_hot.head()) # view the converted columns

    # tflite is unhappy with float64 data...
    insurance_one_hot = insurance_one_hot.astype(np.float32)

    # Create X & y values
    X = insurance_one_hot.drop("charges", axis=1)
    y = insurance_one_hot["charges"]

    # View features
    #print(X.head())

    # Create training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X,
    #                                                    y,
    #                                                    test_size=0.2,
    #                                                    random_state=42) # set random state for reproducible splits
    return train_test_split(X, y, test_size=0.2, random_state=42) # set random state for reproducible splits

def create_model() -> tf.keras.Model:
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(100, activation='exponential'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))

    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.Adam(), # Adam works but SGD doesn't
                  metrics=['mae'])

    return model

def convert_tflite_model(model):
    """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format
        Args:
                model (tf.keras.Model): the trained hello_world Model
        Returns:
                The converted model in serialized format.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model


def save_tflite_model(tflite_model, save_dir, model_name):
    """save the converted tflite model
    Args:
            tflite_model (binary): the converted model in serialized format.
            save_dir (str): the save directory
            model_name (str): model name to be saved
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_name)
    with open(save_path, "wb") as f:
        f.write(tflite_model)
    logging.info("Tflite model saved to %s", save_dir)


def train_model(epochs):
    """Train keras hello_world model
        Args: epochs (int) : number of epochs to train the model
                x_train (numpy.array): list of the training data
                y_train (numpy.array): list of the corresponding array
        Returns:
                tf.keras.Model: A trained keras hello_world model
    """

    X_train, X_test, y_train, y_test = load_data()

    model = create_model()
    model.fit(X_train,
              y_train,
              epochs=epochs,
              verbose=2)

    print("Error stats: %s" % model.evaluate(X_test, y_test))

    save_path = os.path.join(OUTPUT_DIR, "model.keras")
    model.save(save_path)
    logging.info("TF model saved to %s" % save_path)

    return model

def write_params(trained_model):
    # Extract weights
    info = {}
    info['inputs'] = trained_model.inputs[0].shape[1]
    info['layers'] = []
    for i, layer in enumerate(trained_model.layers):
        data = {}
        data = {}
        data['layer_name'] = "layer_%s" % i
        config = layer.get_config()
        for key in ["name", "units", "activation"]:
            data[key] = config[key]
            #print("%s: %s" % (key, config[key]), end=", ")
        #print()
        weights, biases = layer.get_weights()
        data['weights'] = weights.tolist()
        data['bias'] = biases.tolist()
        info['layers'].append(data)

    # Write the params out for sanity checking!
    with open(os.path.join(OUTPUT_DIR, "params-linear_regression.yaml"), "w") as f:
        yaml.safe_dump(info, f)

def main(_):

    # make the output directory...
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    trained_model = train_model(FLAGS.epochs)

    # Write the params out for sanity checking!
    write_params(trained_model)

    # Convert and save the model to .tflite
    tflite_model = convert_tflite_model(trained_model)
    save_tflite_model(tflite_model,
                      OUTPUT_DIR,
                      model_name="linear_regression.tflite")


if __name__ == "__main__":
    app.run(main)
