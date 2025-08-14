from itertools import cycle, chain, product
from collections import defaultdict
import os
import warnings
from tqdm import tqdm, trange
warnings.filterwarnings("error")
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
from .consts import *
from .util import banner, abort_check

# add logger
from .logger import get_logger
logger = get_logger(__file__)

CALIBRATION_CACHE_FILE = "cc.pickle"
import pickle

def binarysearch(network, neuron, layer_idx):
    """ Do a binary search against a single neuron"""

    # Recover the signs, minvals and maxvals quickly
    logger.warning("Loading Callibration from cache")
    if os.path.exists(CALIBRATION_CACHE_FILE):
        with open(CALIBRATION_CACHE_FILE, 'rb') as handle:
            signs, minvals, maxvals = pickle.load(handle)
    else:
        logger.info("Starting Calibrating")
        signs, minvals, maxvals = smart_calibrate(network, neuron, layer_idx)
        data = [signs, minvals, maxvals]
        with open(CALIBRATION_CACHE_FILE, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # These are shared for all the equations for this neuron!
    coefficient_matrix = []
    ordinate_values = []
    depth_stats = {}
    depth_stats["layer_idx"] = layer_idx
    depth_stats["neuron"] = neuron
    depth_stats["equations"] = []

    # create a generator that emits the index of the inputs (1, 2, 3, 1, 2, 3, 1 etc) forever
    input_idxes = cycle(range(len(neuron.inputs)))

    # Generate inputs + 1 equations to solve for bias
    num_equations = len(neuron.inputs) + 1 + BONUS_EQUATIONS

    for eq in trange(num_equations, desc="equation", leave=False):

        equation_stats = []

        # pop the 'next' input index
        i = next(input_idxes)

        # If the signs are all the same, the convergence points will yield at a terrible solution
        # e.g.
        #
        # [a1, b1, c1] = 103.721
        # [a2, b2, c2] = 103.721
        # [a3, b3, c3] = 103.721 
        #
        # etc
        #
        # Solution: toggle the sign every other eq to find some other
        # convergence points!
        if len(set(signs.values())) == 1:
            maxvals[i] = maxvals[i] * -1.1 # not all overflow points are symmetrical
            if signs[i] == POSITIVE:
                signs[i] = NEGATIVE
            else:
                signs[i] = POSITIVE

        # seed input
        # NOTE: the choice of the non-searching parameters seems important but I'm not sure how to tune this ideally...
        # I think it should be close in magnitude to the eventual result but it's hard to predict that?
        #target = np.random.uniform(-1, 1, len(neuron.inputs)).astype(PRECISION)
        #target = np.array([minvals[i] * np.random.uniform(-0.5, 0.5)] * len(neuron.inputs), dtype=PRECISION)
        target = np.array([minvals[i] * np.random.uniform((-8/784), (8/764))] * len(neuron.inputs), dtype=PRECISION)

        logger.info("Finding convergence points for equation %s" % (eq+1))

        # set bounds and mid
        lower = PRECISION(0.0)

        # we should have already made sure that this underflows/overflows based on the above
        # multiply by 2 to counter first midpoint calc
        upper = maxvals[i] * 2 

        # Store progress of search for early return break
        mids = []

        # Generate the boundary (tuple of two cases we're trying to find) for
        # the given function / input sign
        boundary = neuron.func.define_boundary(signs[i])

        # binary search
        for depth in trange(DEPTH, desc="search depth", leave=False):

            # TODO: remove me!
            abort_check()

            # figure out midpoint and store it
            mid = (lower + upper) / 2.0
            mids.append(mid)

            # Adjust target
            target[i] = mid

            # generate 'insert' value to be inserted into network to
            # produce target value at the neuron in question...
            insert = network.recursive_calc(layer_idx, target)


            for _ in range(3):
                try:
                    # run the data through the network and parse the log / state 
                    # of the neuron in question.
                    #
                    # note that this may re-run / reparse the log if there are
                    # inconsistencies
                    network.predict(insert, neuron)

                    #if depth % 10 == 0:
                        #logger.info("depth %s: i:%s --> t:%s --> %s" % (depth, insert, target, neuron.mode))
                        #logger.info("depth %s: i:%s --> %s" % (depth, insert, neuron.mode))
                        #logger.info("depth %s: %s" % (depth, neuron.mode))
                    logger.info("eq %s / depth %s: var:%s vs. 783*%s--> %s" % (eq, depth, insert[i], insert[i+1], neuron.mode))

                    # Determine the direction to go based on the neuron's mode and the sign!
                    direction = neuron.func.decide_direction(signs[i], neuron.mode)
                    break
                except RuntimeError as rte:
                    print("caught %s" % str(rte))
                finally:
                    logger.critical("ruh roh")
                    import IPython
                    IPython.embed()

            if direction == DOWN:
                upper = mid # go down
            elif direction == UP:
                lower = mid # come up
            else:
                raise RuntimeError("Impossible direction")

            # Compute lhs/rhs based on where we are!
            lhs = np.append(target, [1.0]) # add the bias term to be solved!
            rhs = neuron.func.get_boundary(boundary)

            # store the info at this depth!
            # NOTE: the output is really only for debugging purposes...
            solution_info = {
                "insert": insert.copy(),
                "target": target.copy(),
                "solution": mids[-1],
                "depth": depth,
                "boundary": boundary,
                'lhs': lhs,
                'rhs': rhs,
            }
            equation_stats.append(solution_info)

            # if the last BREAK_EARLY_COUNT values in the search are the same, abort early!
            if len(mids) > BREAK_EARLY_COUNT and all(x == mids[-1] for x in mids[-BREAK_EARLY_COUNT:]):
                #logger.warning("Breaking after %s / %s iterations" % (depth, DEPTH))
                break

        # after we finish the quation, store the results from that scan
        depth_stats["equations"].append(equation_stats)
  
    # Now that we're done, build a set of coefficient_matrix, ordinate_values per depth
    # This'll let us compute accuracy at any given depth
    eqs_by_depth = {}
    for depth, pair in enumerate(zip(*depth_stats['equations'])):
        matrix = [e['lhs'] for e in pair]
        values = [e['rhs'] for e in pair]
        eqs_by_depth[depth] = (matrix, [values])

    return eqs_by_depth


def smart_calibrate(network, neuron, layer_idx):
    """Force large positive input into a exp-based neuron to force 
    overflow/underflow, determining signs
    """

    signs = {}
    minvals = {}
    maxvals = {}
    count = 0

    for i in trange(len(neuron.inputs), desc="calibration by input", leave=False):
        signs[i] = None
        maxvals[i] = SMARTBOUND_START
        minvals[i] = PRECISION(0.1)
        #if i % 10 == 0:
        #    banner("Finding sign for input %s / %s" % (i, len(neuron.inputs)))

        goes = 0
        while not signs[i]:

            # TODO: remove me!
            abort_check()

            if goes > 10:
                raise RuntimeError("SAFEGUARD")
            goes += 1

            # create the target array -- all but index i are minval!
            target = [minvals[i]] * len(neuron.inputs)
            target[i] = maxvals[i]

            # turn target into a numpy array
            target = np.array(target, dtype=PRECISION)

            # generate the insert value to achieve this wherever we are
            insert = network.recursive_calc(layer_idx, target)
            if np.isnan(insert).any():
                raise RuntimeError("NaN detected in recursive calc")
            count += 1

            # Run the network
            network.predict(insert, neuron)

            #logger.debug("i:%s --> t:%s --> (%s = mode %s)" % (insert, target, neuron, neuron.mode))

            # try to convert the current mode into a sign
            sign = neuron.func.mode_to_sign(neuron.mode)
            if sign:
                signs[i] = sign

            # If we can't, the magnitude isn't big enough yet!
            else:
                # Multiplying by 10 ought to grow the value in case the weight is
                # so small, we're still in a normal case.
                #
                # TODO: this does not work if there is a low ratio of inputs to
                # neurons as in:
                #
                # t[1.e+03 1.e-01 ... 1.e-01 ] --> i[-0.12593848] 
                # ...
                # t[1.e+12 1.e-01 ... 1.e-01 ] --> i[-0.10710279]
                #
                # Eventually, t becomes [1.e+322 1.e-01 ... ] which overflows
                # without actually producing and input that has caused an overflow!
                #
                # This might only be a problem in simple toy networks with few
                # imputs; leaving this for now.
                #
                maxvals[i] *= 10

                #logger.debug("input %s: i%s/t%s failed to over/underflow (mode=%s); 10x ing maxval" % (i, insert, target, neuron.mode))
                #logger.debug("input %s: failed to over/underflow (mode=%s); 10x ing maxval" % (i, neuron.mode))


    pp_signs = " ".join([v for _, v in sorted(signs.items(), key=lambda x: x[0])])
    pp_maxvals = " ".join([str(v) for _, v in sorted(maxvals.items(), key=lambda x: x[0])]) 
    pp_minvals = " ".join([str(v) for _, v in sorted(minvals.items(), key=lambda x: x[0])]) 
    logger.info("count: %s" % count)
    logger.info("recovered minvals: %s" % pp_minvals)
    logger.info("recovered signs: %s" % pp_signs)
    logger.info("recovered maxvals: %s" % pp_maxvals)

    return signs, minvals, maxvals


def gridsearch(network, neuron, layer_idx):

    coefficient_matrix = []
    ordinate_values = []

    # set to low level...
    neuron.func.attack_mode = LOW_LEVEL

    for i in trange(len(neuron.inputs), desc="gridsearch by input", leave=False):

        banner("INPUT %s" % i)

        # for the desired number of equations...
        for eq_num in trange(len(neuron.inputs) + 1 + BONUS_EQUATIONS, desc="equation", leave=False):

            logger.info("Finding convergence points for equation %s" % (eq_num))

            # initialize target with smallish values
            target = np.random.rand(len(neuron.inputs)) * 0.01

            last_mode = None
            last_target = None
            convergence_temp = []

            # TODO: rather than hardcode 1000, recursively deepen the search?
            # replace with a breadth first search of depth 9 to try to find boundaries...
            #outputs = []
            #for val in tqdm(np.linspace(0.0, 1.0, num=10000)):

            # Generate a non-linear gridsearch clustered around 0 (where a lot
            # of the INNER zones are) to try to find edges
            #
            # TODO: dynamically adjust this if we're not getting enough equations...?

            NUM = 100
            positive_space = np.geomspace(0.00001, GRIDSEARCH_BOUND, num=NUM)
            negative_space = -positive_space.copy()
            full_range = np.concatenate((np.flip(negative_space), positive_space))

            logger.debug("SCANNING")

            for c, val in enumerate(full_range):

                # TODO: remove me!
                abort_check()

                target[i] = val # bias towards larger values, at least for sigmoid!
                target_arr = np.array(target, dtype=PRECISION)
                try:
                    insert = network.recursive_calc(layer_idx, target_arr)
                except RuntimeWarning as rtw:
                    #logger.warning(str(rtw))
                    #logger.warning("skipping impossible target: %s" % target_arr)
                    continue

                # run the network
                network.predict(insert, neuron)

                this_mode = neuron.mode

                #neuro_state = [n.mode for n in network.layers[layer_idx]]
                #logger.debug("works:", c, "/", NUM*2, insert, target_arr, neuro_state)
                #logger.debug("works:", c, "/", NUM*2, insert, " ".join(["%s %s" % p for p in neuro_state]))

                if c % 10 == 0:
                    logger.info(c, "/", NUM*2, insert)

                # if we hit a state change
                if this_mode != last_mode and last_target is not None: # and 1 not in set([this_mode, last_mode]):

                    # handle complex types as in sigmoid
                    if type(this_mode) == type(last_mode) == type(()):
                        if this_mode[0] != last_mode[0]:
                            logger.warning("Don't know how to handle high level boundaries in gridsearch yet")
                            continue

                        if this_mode[1] is None or last_mode[1] is None:
                            #logger.warning("skipping strange 'None' state!")
                            continue

                    # Do a binary search between this gridsearch points for increased accuracy
                    result = _sub_search(network, neuron, layer_idx, i, last_target, last_mode[1], target_arr, this_mode[1])
                    logger.info("Did search ~%s --> (%s, %s) --> %s" % (insert, this_mode[1], last_mode[1], result))

                    # NOTE: boundaries between 6/7 and 5/6 are symetric so we don't know what the sign of the boundary point is....
                    lhs = np.append(result, [1.0]) # add the bias term to be solved!
                    #rhs = neuron.func.get_boundary((last_mode, this_mode))
                    rhs = ((last_mode[1], this_mode[1]))

                    coefficient_matrix.append(lhs)
                    ordinate_values.append(rhs)

                # storing last results
                last_mode = this_mode
                last_target = target_arr

        # finished equation finding for this input
        logger.info("Finished equation finding for input %s!" % i)

    logger.info("Finished whole run!")

    logger.info("Printing partially discovered covergence points for this neuron!")
    for pair in zip(coefficient_matrix, ordinate_values):
        logger.info(pair[0], pair[1])
    logger.critical("exiting early; Integration with solving architecture not implemented.")
    import sys
    sys.exit()

    ## TODO: move this into the other search...
    #else:
    #    for signset in product([1, -1], repeat=len(ordinate_values)):
    #        candidate = []
    #        for i, o in enumerate(sorted(ordinate_values)):
    #            for v in ordinate_values[o]:
    #                candidate.append(v * signset[i])
    #        candidates.append(candidate)
    # Discard solutions where the mangitude of the weights is extremely low thus 
    # placing everything into the bias.  This is an indicator that the weights
    # are wrong!
    #if len(candidate_results) > 1:
    #    import IPython
    #    IPython.embed()
    #candidate_results = dict([(c, r) for c, r in candidate_results.items() if np.all(abs(r)>1e-6)])

    # TODO: remove duplicates from these!
    return coefficient_matrix, ordinate_values

def _sub_search(network, neuron, layer_idx, i, target0, mode0, target1, mode1):
    """ A mini binary search to enhance our grid search! """

    def get_mode(target):
        insert = network.recursive_calc(layer_idx, target)
        network.predict(insert, neuron)
        return neuron.mode

    low = target0[i]
    high = target1[i]
    mid = 0 

    mids = [mid]

    while mid != low or mid != high:
        mid = (high + low) / 2.0
        mids.append(mid)
        test_target = target0.copy()
        test_target[i] = mid
        result = get_mode(test_target)

        # TODO: need to drive this from data in the activation func!
        if result == mode0: # go up
            low = mid

        elif result == mode1: # go down
            high = mid
        if len(mids) > 5 and len(set(mids[-5:])) == 1:
            break

    final = target0.copy()
    final[i] = mids[-1]
    return final
