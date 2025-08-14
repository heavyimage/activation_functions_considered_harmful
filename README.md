# Artifact for 'Activation Functions Considered Harmful: Recovering Neural Network Weights through Controlled Channels'

## RAID '25

## Instructions for MNIST example

* Setup sgx-step as indicated in [README-sgxstep.md](README-sgxstep.md)
	* You will need SGX-capable hardware
* build tensorflowlite-micro
    * `cd app/nn_param_recovery/Enclave/tflite-micro`
    * `make -f ./tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=release`
    * You should build './gen/linux_x86_64_release_gcc/lib/libtensorflow-microlite.a'
* Train the victim model:
    * `cd app/nn_param_recovery/models/mnist`
    * `python3 train_mnist.py`
    * `python3 test_mnist.py`
* Convert the tf-lite model into its header file representation:
    * `cd ../`
    * `python3 generate_cc_arrays.py . mnist/models/mnist.tflite`
* Start the attack (this will build the victim enclave):
    * `cd app/nn_param_recovery/`
    * python3 attack.py

## Note:
* The attack will create a .pickle file to checkpoint itself as it works.
* When all of the convergence points have been found, a table will be printed which should match the table in the paper.

