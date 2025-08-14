import yaml
import math
import sys
import time
import pandas as pd
from absl import app
from absl import flags
from absl import logging
import tensorflow_datasets as tfds
from collections import defaultdict

import tensorflow as tf
import os
import numpy as np
import random

SEED = 0
FIRST_LAYER_NEURON_COUNT = 128
SECOND_LAYER_NEURON_COUNT = 10
INPUT_DIMENSION = 28
FLAT_DIMENSION = INPUT_DIMENSION*INPUT_DIMENSION

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

NUM_EPOCHS = 16
BATCH_SIZE = 16

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", NUM_EPOCHS, "number of epochs to train the model.")
flags.DEFINE_boolean("save_tf_model", False, "store the original unconverted tf model.")

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODELS_DIR, "models")

def load_data():
    # https://www.tensorflow.org/datasets/keras_example

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            #shuffle_files=True,
            as_supervised=True,
            with_info=True,
            )

    # Due to an INSANE bug, calling tfds.load seems to import a root logging
    # handler that breaks my logging setup.
    import logging
    rootlogger = logging.getLogger()
    rootlogger.handlers = []

    # Training
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Testing
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test

def create_model() -> tf.keras.Model:
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input((INPUT_DIMENSION, INPUT_DIMENSION, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(FIRST_LAYER_NEURON_COUNT, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(SECOND_LAYER_NEURON_COUNT, activation='softmax'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

def convert_tflite_model(model, quantized=False):
    """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format
        Args:
                model (tf.keras.Model): the trained hello_world Model
        Returns:
                The converted model in serialized format.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantized:
        # from https://ai.google.dev/edge/litert/models/post_training_quantization:
        #
        # Dynamic range quantization provides reduced memory usage and faster
        # computation without you having to provide a representative dataset for
        # calibration. This type of quantization, statically quantizes only the
        # weights from floating point to integer at conversion time, which
        # provides 8-bits of precision:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
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

    ds_train, ds_test = load_data()

    model = create_model()
    model.fit(ds_train,
              epochs=epochs,
              verbose=2)

    print("Error stats: %s" % model.evaluate(ds_test))

    save_path = os.path.join(OUTPUT_DIR, "model.keras")
    model.save(save_path)
    logging.info("TF model saved to %s" % save_path)

    return model

def write_tflite_params(model_path):
    # Load TFLite model and allocate tensors.
    abspath = os.path.join(OUTPUT_DIR, model_path)
    interpreter = tf.lite.Interpreter(model_path=abspath, experimental_preserve_all_tensors=True)
    #interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tensor_details = interpreter.get_tensor_details()

    params = {}
    param_map = {
            (FIRST_LAYER_NEURON_COUNT, FLAT_DIMENSION): "layer_0/weights",
            (FIRST_LAYER_NEURON_COUNT,): "layer_0/bias",
            (SECOND_LAYER_NEURON_COUNT, FIRST_LAYER_NEURON_COUNT): "layer_1/weights",
            (SECOND_LAYER_NEURON_COUNT,): "layer_1/bias",
        }
    activations = {
            "layer_0": "sigmoid",
            "layer_1": "softmax"
        }
    units = {
            "layer_0": FIRST_LAYER_NEURON_COUNT,
            "layer_1": SECOND_LAYER_NEURON_COUNT,
    }
    name = {
            "layer_0": "dense",
            "layer_1": "dense_1",
    }

    # Build up the data to write to disk
    info = {}
    info['inputs'] = input_details[0]['shape'][1:].tolist()
    info['layers'] = []

    # Doing this totally procedurally was making my head spin.  Lets go manual for now.
    #
    #for details in tensor_details:
    #    idx = details['index']
    #    tensor_name = details['name']
    #    shape = details['shape']
    #    scales = details['quantization_parameters']['scales']
    #    weights = interpreter.get_tensor(idx)
    #    print(idx, tensor_name, shape, scales, weights)

    # LETS GO MANUAL MODE
    # worked out via netron....
    #QUANTIZED:
    #layer0:
    #    weights:
    #        5 tfl.pseudo_qconst1 [128 784]
    #            includes scales and weights
    #    bias:
    #        3 arith.constant2 [128]
    #            just weights (no scales)

    #layer1
    #    :weights:
    #        4 tfl.pseudo_qconst [ 10 128]
    #            includes scales and weights
    #    bias:
    #        2 arith.constant1 [10]
    #            just weights (no scales)
    #
    #NORMAL:
    #layer_0:
    #    weights:
    #        2 arith.constant1 [128 784]
    #            no scaling, obviously
    #    bias:
    #        3 arith.constant2 [128]
    #            no scaling obviously
    #layer_1:
    #    weights:
    #        1 arith.constant [ 10 128]
    #            no scaling obviously
    #    bias:
    #        4 arith.constant3 [10]
    #            no scaling obviously

    #
    # NOT QUANTIZED
    if "quantized" not in model_path:
        layer_0_weights = next(t for t in tensor_details if t['name'] == 'arith.constant1' and all(t['shape'] == (FIRST_LAYER_NEURON_COUNT, FLAT_DIMENSION)))
        layer_0_bias = next(t for t in tensor_details if t['name'] == 'arith.constant2' and all(t['shape'] == (FIRST_LAYER_NEURON_COUNT,)))
        layer_1_weights = next(t for t in tensor_details if t['name'] == 'arith.constant' and all(t['shape'] == (SECOND_LAYER_NEURON_COUNT, FIRST_LAYER_NEURON_COUNT)))
        layer_1_bias = next(t for t in tensor_details if t['name'] == 'arith.constant3' and all(t['shape'] == (SECOND_LAYER_NEURON_COUNT,)))
    else:
        layer_0_weights = next(t for t in tensor_details if t['name'] == 'tfl.pseudo_qconst1' and all(t['shape'] == (FIRST_LAYER_NEURON_COUNT, FLAT_DIMENSION)))
        layer_0_bias = next(t for t in tensor_details if t['name'] == 'arith.constant2' and all(t['shape'] == (FIRST_LAYER_NEURON_COUNT,)))
        layer_1_weights = next(t for t in tensor_details if t['name'] == 'tfl.pseudo_qconst' and all(t['shape'] == (SECOND_LAYER_NEURON_COUNT, FIRST_LAYER_NEURON_COUNT)))
        layer_1_bias = next(t for t in tensor_details if t['name'] == 'arith.constant1' and all(t['shape'] == (SECOND_LAYER_NEURON_COUNT,)))

    first_layer_weights = interpreter.get_tensor(layer_0_weights['index'])
    first_layer_bias = interpreter.get_tensor(layer_0_bias['index'])
    second_layer_weights = interpreter.get_tensor(layer_1_weights['index'])
    second_layer_bias = interpreter.get_tensor(layer_1_bias['index'])

    if "quantized" not in model_path:
        first_layer_weights = first_layer_weights.transpose()
        second_layer_weights = second_layer_weights.transpose()
    else:
        first_layer_weights = first_layer_weights.transpose() * layer_0_weights['quantization_parameters']['scales']
        second_layer_weights = second_layer_weights.transpose() * layer_1_weights['quantization_parameters']['scales']

    # first layer:
    layer_name = "layer_0"
    layer = {
            'weights': first_layer_weights.tolist(),
            'bias': first_layer_bias.tolist(),
            'layer_name': layer_name,
            'activation': activations[layer_name],
            'units': units[layer_name],
            'name': name[layer_name],
        }
    info['layers'].append(layer)

    # second layer:
    layer_name = "layer_1"
    layer = {
            'weights': second_layer_weights.tolist(),
            'bias': second_layer_bias.tolist(),
            'layer_name': layer_name,
            'activation': activations[layer_name],
            'units': units[layer_name],
            'name': name[layer_name],
        }
    info['layers'].append(layer)

    # Write the params out for sanity checking!
    with open(os.path.join(OUTPUT_DIR, f"params-{model_path}.yaml"), "w") as f:
        yaml.safe_dump(info, f)

def main(_):

    # make the output directory...
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    trained_model = train_model(FLAGS.epochs)


    #for mode in ["float", "quantized"]:
    for mode in ["float"]:

        if mode == "float":
            quantized = False
            path = "mnist.tflite"
        else:
            quantized = True
            path = "mnist-quantized.tflite"

        # Convert to tflite and save the model
        tflite_model = convert_tflite_model(trained_model, quantized=quantized)
        save_tflite_model(tflite_model,
                          OUTPUT_DIR,
                          model_name=path)

        # write the params out for sanity checking
        write_tflite_params(path)

    print("=========================================================")
    print("Don't forget to run python3 mnist_test.py!")
    print("Don't forget to run ../generate_cc_arrays.py both models!")

if __name__ == "__main__":
    app.run(main)
