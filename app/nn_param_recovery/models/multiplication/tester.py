import numpy as np
import tensorflow as tf
from train_multiplication import get_data
from random import randrange


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/multiplication_with_sigmoid_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prep data
a = np.random.uniform(low=-10, high=10, size=10000).astype(np.float32)
b = np.random.uniform(low=-10, high=10, size=10000).astype(np.float32)
x_test = np.dstack((a,b))[0]

# Shuffle the values to guarantee they're not in order
np.random.shuffle(x_test)

# Calculate the corresponding multiplication value
y_test = np.multiply(x_test[:,1:], x_test[:,:1])

# Test model on random input data.
input_shape = input_details[0]['shape']

num_tests = len(x_test)

for i in range(10):
    index = randrange(num_tests)
    gt_output = y_test[index]
    print("Test %s: (index=%s)" % (i, index))
    input_data = np.array([x_test[index]], dtype=np.float32)
    print(" * ".join(str(m) for m in input_data[0]))

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data[0][0], "(guess) vs", gt_output[0], "(gt)")

    print()
