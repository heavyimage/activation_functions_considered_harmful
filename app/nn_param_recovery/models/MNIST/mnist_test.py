import numpy as np
import tensorflow as tf
from mnist_trainer import load_data
from random import randrange

PIXEL_ASCII_MAP = " `^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

def pp(input_):
    for y, row in enumerate(input_):
        for x, c in enumerate(row):
            value = c[0]
            scaled_value = int(value * (len(PIXEL_ASCII_MAP)-1))
            print(PIXEL_ASCII_MAP[scaled_value], end="")
            print(PIXEL_ASCII_MAP[scaled_value], end="")
        print()


def test_model(model_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _, ds_test = load_data()
    ds_test = list(ds_test.unbatch())

    # Test model on random input data.
    input_shape = input_details[0]['shape'][1:]

    num_tests = len(ds_test)


    for i in range(20):
        index = randrange(num_tests)
        print("Test %s: (index=%s)" % (i, index))


        input_data = ds_test[index][0].numpy()
        gt_output = ds_test[index][1].numpy()
        #pp(input_data)

        print(gt_output)
        with open(f"/tmp/{gt_output}.txt", "w") as f:
            f.write(" ".join(str(item) for item in input_data.flatten()))

        #print("running test!")    
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_data, axis=0))
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        calculated = np.argmax(output_data)
        print(model_path, "test", (i+1), "/", 20)
        print(f"ground truth: {gt_output}")
        print(f"prediction: {calculated}")
        if calculated != gt_output:
            print("\t\t\t\t\t\t!!!MISMATCH!!!")
        print()

def main():
    test_model("models/mnist.tflite")
    test_model("models/mnist-quantized.tflite")

if __name__ == "__main__":
    main()
