# A selection of models for testing the PoC with

# Workflow
* train a model using tensorflow for python
* convert the model to a tflite model:
    ```
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model
    ```
* Make sure the output looks okay!
    * I think errors about arith.constant are okay, eg:
    ```
    * Accepted dialects: tfl, builtin, func
    * Non-Converted Ops: 6, Total Ops 14, % non-converted = 42.86 %
    * 6 ARITH ops
    - arith.constant:    6 occurrences  (f32: 6)
    ```


* Test the tflite model works / matches the original model's evaluation using the `sanity_check.py` script
* Convert the tflite file into .cc/.h file for embedding in the enclave:
    ```
    python3 generate_cc_arrays.py . /path/to/model.tflite
    ```
