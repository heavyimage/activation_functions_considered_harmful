# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""hello_world model training for sinwave recognition
"""
import math
import sys
import yaml

from absl import app
from absl import flags
from absl import logging

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
flags.DEFINE_boolean("save_tf_model", False,
                                         "store the original unconverted tf model.")
flags.DEFINE_string("activation_function", None,
                                         "The activation function to use for the first 2 layers")
flags.mark_flag_as_required('activation_function')

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODELS_DIR, "models")

def filename_generator(extension, quantized):
    if quantized:
        qstring = "_quantized"
    else:
        qstring = ""
    if extension == "yaml":
        p = "_params"
    else:
        p = ""

    model_name = "sin_with_{af}{q}{p}.{ext}".format(
            af=FLAGS.activation_function,
            q=qstring,
            p=p,
            ext=extension)
    return model_name

def get_data():
    """
    The code will generate a set of random `x` values,calculate their sine
    values.
    """
    # Generate a uniformly distributed set of random numbers in the range from
    # 0 to 2π, which covers a complete sine wave oscillation
    x_values = np.random.uniform(low=-2 * math.pi, high=2 * math.pi, size=10000).astype(np.float32)

    # Shuffle the values to guarantee they're not in order
    np.random.shuffle(x_values)

    # Calculate the corresponding sine values
    y_values = np.sin(x_values).astype(np.float32)

    return (x_values, y_values)


def create_model() -> tf.keras.Model:
    model = tf.keras.Sequential()

    activation_function = FLAGS.activation_function

    # First layer takes a scalar input and feeds it through 16 "neurons". The
    # neurons decide whether to activate based on the 'relu' activation function.
    model.add(tf.keras.layers.Dense(16, activation=activation_function, input_shape=(1, )))

    # The new second and third layer will help the network learn more complex
    # representations
    model.add(tf.keras.layers.Dense(16, activation=activation_function))

    # Final layer is a single neuron, since we want to output a single value
    model.add(tf.keras.layers.Dense(1, activation="relu"))

    # Compile the model using the standard 'adam' optimizer and the mean squared
    # error or 'mse' loss function for regression.
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def convert_tflite_model(model, quantized):
    """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format
        Args:
                model (tf.keras.Model): the trained hello_world Model
        Returns:
                The converted model in serialized format.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantized:
        banner("Generating Quantized Model")
        def representative_data_gen():
            for input_value in np.random.uniform(low=-2 * math.pi, high=2 * math.pi, size=10000).astype(np.float32):
                yield [input_value]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    else:
        banner("Generating Non-Quantized Model")
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


def train_model(epochs, x_values, y_values):
    """Train keras hello_world model
        Args: epochs (int) : number of epochs to train the model
                x_train (numpy.array): list of the training data
                y_train (numpy.array): list of the corresponding array
        Returns:
                tf.keras.Model: A trained keras hello_world model
    """
    model = create_model()
    model.summary()
    model.fit(x_values,
              y_values,
              epochs=epochs,
              validation_split=0.2,
              batch_size=64,
              verbose=2)

    filename = filename_generator("keras", False)
    save_path = os.path.join(OUTPUT_DIR, filename)
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

    filename = filename_generator("yaml", False)
    save_path = os.path.join(OUTPUT_DIR, filename)
    with open(save_path, "w") as f:
        yaml.safe_dump(info, f)

def banner(string):
    print("\n=== %s ===" % string)

def main(_):
    banner("Getting Data")
    x_values, y_values = get_data()
    banner("Training")
    trained_model = train_model(FLAGS.epochs, x_values, y_values)

    # Write the params out for sanity checking!
    banner("Writing Params")
    write_params(trained_model)

    # Convert and save the unquanitzed model to .tflite
    for is_quantized in [False, True]:
        model_name = filename_generator("tflite", is_quantized)
        tflite_model = convert_tflite_model(trained_model, is_quantized)
        banner("Writing %s" % model_name)
        save_tflite_model(tflite_model, OUTPUT_DIR, model_name=model_name)


if __name__ == "__main__":
    app.run(main)
