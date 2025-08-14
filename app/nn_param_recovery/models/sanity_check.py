import numpy as np
import tensorflow as tf

# sigmoid
MODEL = "/home/jesse/UoB/phd/contrib/sgx-step/app/poc_poc/Enclave/tflite-micro/tensorflow/lite/micro/examples/hello_world/models/hello_world_float_sigmoid.tflite"
#MODEL = "/home/jesse/UoB/phd/contrib/sgx-step/app/poc_poc/Enclave/tflite-micro/tensorflow/lite/micro/examples/hello_world/models/hello_world_float.tflite"

INPUT = -3.145

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array([[INPUT]], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("#"*50)
print("prediction(%s) = %s" % (INPUT, output_data))
