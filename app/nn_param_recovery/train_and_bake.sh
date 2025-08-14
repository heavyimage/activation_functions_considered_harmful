#!/bin/bash

function cleanup {
    rm models/linear_regression/models/*.keras
    rm models/linear_regression/models/*.yaml
    rm models/linear_regression/models/*.tflite
    rm models/linear_regression/models/*.cc
    rm models/linear_regression/models/*.h

    rm models/tf-micro-lite-sin/models/*.keras
    rm models/tf-micro-lite-sin/models/*.yaml
    rm models/tf-micro-lite-sin/models/*.tflite
    rm models/tf-micro-lite-sin/models/*.cc
    rm models/tf-micro-lite-sin/models/*.h

    rm models/basic/models/*.keras
    rm models/basic/models/*.yaml
    rm models/basic/models/*.tflite
    rm models/basic/models/*.cc
    rm models/basic/models/*.h

    rm models/multiplication/models/*.keras
    rm models/multiplication/models/*.yaml
    rm models/multiplication/models/*.tflite
    rm models/multiplication/models/*.cc
    rm models/multiplication/models/*.h
}

function train {
    python3 models/linear_regression/linear_regression_trainer.py
    python3 models/tf-micro-lite-sin/train-prismatic.py --activation_function=exponential
    python3 models/tf-micro-lite-sin/train-prismatic.py --activation_function=sigmoid
    python3 models/basic/train-prismatic.py --activation_function=exponential
    python3 models/multiplication/train_multiplication.py
}

function bake {
    python3 models/generate_cc_arrays.py . models/linear_regression/models/linear_regression.tflite
    python3 models/generate_cc_arrays.py . models/tf-micro-lite-sin/models/sin_with_exponential.tflite
    python3 models/generate_cc_arrays.py . models/tf-micro-lite-sin/models/sin_with_sigmoid.tflite
    python3 models/generate_cc_arrays.py . models/basic/models/sin_with_exponential.tflite
    python3 models/generate_cc_arrays.py . models/multiplication/models/multiplication_with_sigmoid.tflite

}

cleanup
train
bake
