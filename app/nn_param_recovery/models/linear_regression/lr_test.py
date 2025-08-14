import numpy as np
import tensorflow as tf
from linear_regression_trainer import load_data
from random import randrange

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/linear_regression.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, X_test, _, y_test = load_data()

# Test model on random input data.
input_shape = input_details[0]['shape']

num_tests = len(X_test)

for i in range(10):
	index = randrange(num_tests)
	print("Test %s: (index=%s)" % (i, index))
	#import IPython
	#IPython.embed()
	input_data = np.array([X_test.iloc[index].to_numpy()])
	print("====\n%s\n====" % input_data[0])
	gt_output = y_test.iloc[index]

	print("running test!")	
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()

	# The function `get_tensor()` returns a copy of the tensor data.
	# Use `tensor()` in order to get a pointer to the tensor.
	output_data = interpreter.get_tensor(output_details[0]['index'])
	print(output_data[0][0], "vs", gt_output)
	print()
