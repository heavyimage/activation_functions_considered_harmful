import numpy as np

#DEPTH = 100                # depth of the search -- anything over 100 will run the whole thing!
DEPTH = 25                  # depth of the search
BONUS_EQUATIONS = 0         # bonus equations beyond the minimum number required.

# Settings -- shouldn't have to touch these.
CHECKPOINT = True   # whether to save checkpoints
PRECISION = np.float64
SMARTBOUND_START = 1000
GRIDSEARCH_BOUND = PRECISION(100000.0)
MIN_FLOAT = PRECISION(0.001)
BREAK_EARLY_COUNT = 3
EXECUTION_ATTEMPTS = 10
SETTINGS = "poc_settings.yaml"

# general constants:
SMART = 1
GRID = 2
POSITIVE = "+"
NEGATIVE = "-"
NO_SUMMARY = "<NO SUMMARY>"
NO_MODE = "<NO MODE>"
HIGH_LEVEL = "High Level"
LOW_LEVEL = "Low Level"
DOWN = "Down"
UP = "Up"

# NN Layer processing
# TODO: should probably be per layer
LAYER_SPACER_SYMBOL = "tflite::SingleArenaBufferAllocator::ResetTempAllocations()"
LAYER_BOUND_COUNT = 18

