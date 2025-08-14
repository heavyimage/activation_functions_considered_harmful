# Instructions

This POC demonstrates neural network parameter recovery.  It corresponds with our paper, "Activation Functions Considered Harmful: Recovering Neural Network Weights through Controlled Channels" submitted to Euro S&P '25.

## Repo Layout

```
.
├── README.md                           # This README
├── attack                              # Python attack code
│   ├── activations.py
│   ├── consts.py
│   ├── enclave_elf.py
│   ├── __init__.py
│   ├── log.py
│   ├── network.py
│   ├── poc.py
│   ├── search.py
│   └── util.p
├── Enclave                             # enclave
│   ├── encl.config.xml
│   ├── encl.cpp                        # enlave source
│   ├── encl.edl
│   ├── encl.h
│   ├── Makefile                        # Enclave Makefile (called by main Makefile)
│   ├── model_path.h
│   ├── private_key.pem
│   ├── public_key.pem
│   └── tflite-micro                    # location of tflite checkout
├── main.c
├── Makefile                            # main Makefile
├── models                              # Location of NN model training scripts
│   ├── basic
│   ├── generate_cc_arrays.py
│   ├── linear_regression
│   ├── multiplication
│   ├── README.md
│   ├── sanity_check.py
│   └── tf-micro-lite-sin
├── poc_settings.yaml                   # Script to control which network is used for attack
├── requirements.txt                    # required python modules
│
├── run.sh                              # a helper script to run the attack with sudo
├── select_model.py                     # used at compile time to read the poc_settings.yaml file and grab the right baked network
├── start_attack.py                     # The entry point into the attack itself.
│
├── download_and_build_tf_micro.sh      # Helper script to setup tf-micro
└── train_and_bake.sh                   # helper script to train and bake all the models
```



## Setup steps
1. Setup SGX-Step as indicated in [../README.md](SGX-Step's README).
    * You will need SGX-capable hardware
    * Note that enabling SGX-Step means disabling all but one core which can make the following build steps slow.
2. build `tensorflowlite-micro`
    * run `download_and_build_tf_micro.sh`
    * this might take a while...
    * You should have built a `Enclave/tflite-micro/gen/linux_x86_64_debug/lib/libtensorflow-microlite.a`
3. Install required python libraries via `pip3 install -r requirements.txt`
4. Train the victim models:
    * run `train_and_bake.sh`
5. Start the attack (this will rebuild the victim enclave):
    * `python3 start_attack.py
    * The attack will create a .pickle file to checkpoint itself as it works.
    * When all of the convergence points have been found, a table will be printed which should match the table in the paper.
	* Sometimes the attack will error midway through (a bug where a bad choice is made for a random number value).
		* Restarting the attack should pick up from where it left off and continue normally.

## Configuration
Have a play in `attack/consts.py` if you want to experiment with some settings discussed in the paper:

```
DEPTH = 100                 # depth of the search -- anything over 100 will run the whole thing!
BONUS_EQUATIONS = 0         # bonus equations beyond the minimum number required.
```

## Aborting the Attack
A note about aborting the search: SGX-Step does not like being interupted and doing so with `ctrl+c` can cause the kernel module to enter a stuck state that is unrecoverable without a reboot.  As a backup, the two long-running search routines check for the presence of a `/tmp/abort` file and will exit between invocations of SGX-Step if this file is discovered.  Simply create it to halt the program safetly.

The checkpointing system should ensure you do not lose much data (the checkpoint is saved at the end of each Neuron)


## Choosing Networks
We include several networks for testing, but only 2 are described in the paper.

* The model training / testing scripts live in the model/ subdirectory
* Models must be baked after they are trained into byte string representations.  This is all handled by the train_and_bake.sh script
    * The baking will output a .yaml file within each model's directory that contains ground truth data extracted directly from the model for use in accuracy tests.
* You can change between networks by editing `USE_MODEL` variable in the `poc_settings.yaml` file.
* Once a new model is selected, simply re-run `python3 start_attack.py` and the model will be compiled into the enclave and used automatically.
* At the start of the attack, the name and a summary of the model architecture will be printed out as part of the attack.

### A note about training
Sometimes the training fails to converge which will work but produce strange results.  The seed values should produce working models for the current training scripts.  If you decide to change the architectures / training you might need to find some new seeds that work.

### Network Summary
Model 1: sin calculator w/ exp
	* not discussed in the paper; calculates sin using exp

Model 2: sin calculator w/ sigmoid
	* not discussed in the paper; calculates sin using sigmoid

Model 3: 11 neuron regression
	* The main example used in our code (Section 6.1); the default for USE_MODEL

Model 4: basic exponential
	 *Extremely basic simple network with exp

Model 5: multiplication network
	* The second example used in our code (Section 6.2)

##Have fun!
