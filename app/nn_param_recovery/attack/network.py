import re
import yaml
import numpy as np
from time import sleep
from graphviz import Digraph
from numpy.linalg import solve, lstsq, inv
import subprocess
from collections import defaultdict
import traceback

from .consts import *
from .util import banner, chunks
from .log import Log
from .activations import TFActivation

# add logger
from .logger import get_logger
logger = get_logger(__file__)

class Neuron(object):
    def __init__(self, name, depth, tffunc):
        self.name = name
        self.depth = depth
        self.func = TFActivation.get_instance(tffunc)
        self.data = 0
        self.mode = NO_MODE
        self.summary = NO_SUMMARY

    def __repr__(self):
        return "< %s >" % self.name

    def clear(self):
        self.summary = NO_SUMMARY
        self.mode = NO_MODE

class HiddenNeuron(Neuron):
    def __init__(self, name, depth, tffunc, gt_bias):
        super().__init__(name, depth, tffunc)
        self.inputs = []

        # for accuracy testing
        self.gt_bias = gt_bias
        self.gt_weights = []

        self.solved_weights = []
        self.solved_bias = None

    def add_parent(self, node, weight):
        assert not isinstance(weight, list)
        self.inputs.append(node)
        self.gt_weights.append(weight)

class Network(object):

    # TODO: add method to instatiate with gt values!
    def __init__(self, models, model_index, enclave_elf):

        self.model_index = model_index

        # stats
        self.executions = 0
        self.incomplete_logs = 0
        self.non_deterministic_steps = 0
        self.neuron_check_failures = 0

        self.inputs = []
        self.layers = []
        self.neuron_count = 0
        self.enclave_elf = enclave_elf
        self.symbols = models[self.model_index]['symbols']
        self.start_offset = models[self.model_index].get('start_offset', 0)
        if self.start_offset != 0:
            logger.warning("Per start_offset setting, discarding first %s log lines" % self.start_offset)

        # Load the net_desc's
        self.description = models[self.model_index]['description']

        # don't keep spamming messages about unparsable activation functions
        self.cant_parse = []

        # Load correct answers!
        with open(models[self.model_index]['ground_truth']) as f:
            ground_truth = yaml.load(f, Loader=yaml.FullLoader)

        # Find the page who's 18 step count accesses indicate the start and end of a layer!
        self.spacer_symbol_page = self.enclave_elf.get_page_for_symbol(LAYER_SPACER_SYMBOL)

        # Load and display architecture
        banner("Loading victim network: '%s'" % self.description)

        # ...inputs
        for i in range(ground_truth['inputs']):
            name = "input.node%s" % i
            n = Neuron(name, 0, "tflinear")
            self.inputs.append(n)
        banner("%s input(s)" % ground_truth['inputs'])

        # ...other layers
        last_layer = self.inputs

        num_layers = len(ground_truth['layers'])
        for layer_no, layer in enumerate(range(num_layers), 1):
            nodes = []
            l = ground_truth['layers'][layer]
            banner("%s has %s %s neuron(s)" % (l['layer_name'], l['units'], l['activation']))
            for node in range(l['units']):
                name = "L%s.N%s" % (layer, node)
                func_name = "tf%s" % l['activation']
                n = HiddenNeuron(name, layer_no, func_name, l['bias'][node])
                # NOTE: this might be the wrong way around...
                for i, weight in enumerate(l['weights']):
                    n.add_parent(last_layer[i], weight[node])
                self.neuron_count += 1
                nodes.append(n)
            self.layers.append(nodes)
            last_layer = self.layers[-1]

    def visualize(self):
        # instantiating object
        graph = Digraph(comment=self.description)
        graph.attr(rankdir='LR', splines='line')

        # plot inputs
        with graph.subgraph(name='cluster_0') as c:
            c.attr(rank='same')
            c.attr(color='blue')
            c.node_attr['style'] = 'filled'
            c.attr(label='layer 0')
            for i, node in enumerate(self.inputs):
                label = node.name + "\n" + node.summary
                c.node(node.name, label)

        # ...other layers
        prev_nodes = self.inputs
        for l, layer in enumerate(self.layers, 1):
            with graph.subgraph(name='cluster_%s' % l) as c:
                c.attr(rank='same')
                c.attr(color='blue')
                c.node_attr['style'] = 'filled'
                c.attr(label='layer %s' % l)
                for node in layer:
                    # create the node
                    bias = str(round(node.gt_bias, 3))
                    label = "%s\nbias: %s\n%s\n%s" % (node.name, bias,
                            node.func.llfunc.name, node.summary)
                    c.node(node.name, label)

                    # create the bias
                    #bias_id = "%s.b" % node.name
                    #c.node(bias_id, str(round(node.bias, 3)))
                    #c.edge(bias_id, node.name)

                    # link to previous nodes

                    for i, pnode in enumerate(prev_nodes):
                        c.edge(pnode.name, node.name, label=str(round(node.gt_weights[i], 3)))
            prev_nodes = self.layers[l-1]

        # saving source code
        graph.format = 'png'
        graph.render('Graph')

    def clear(self):
        for n in self.inputs:
            n.clear()
        for l in self.layers:
            for node in l:
                node.clear()

    def predict(self, input_arr, checkNeuron):
        # TODO: could possibe cache results from neurons I'm not examining and see if those work...

        # clear the network
        self.clear()

        # Re-run the input through the network till we get a complete log
        for attempt in range(EXECUTION_ATTEMPTS):

            # Execute the network with the input and create a log instance
            output = self.execute_victim_with_input(input_arr)

            # discard everything before start_offset to speed up log processing
            if self.start_offset != 0:
                #output = truncate1(output, self.start_offset)
                output = truncate2(output, self.start_offset)

            log = Log(output)

            # ensure the log is complete
            if len(log.pages) == 0 or not log.complete:
                #logger.warning("Incomplete log detected; retrying!")
                self.incomplete_logs += 1
                sleep(attempt/EXECUTION_ATTEMPTS)
                continue

            # Ensure we can read the log
            elif not self.parse_log(log):
                #logger.warning("Error parsing log; retrying!")
                self.non_deterministic_steps += 1
                sleep(attempt/EXECUTION_ATTEMPTS)
                continue

            # If the neuron we're checking on has a bad state, re-run!
            elif checkNeuron.mode in [None, NO_MODE]:
                #logger.warning("Neuron failed check!")
                self.neuron_check_failures += 1
                sleep(attempt/EXECUTION_ATTEMPTS)
                continue

            else:
                break

        else:
            raise RuntimeError("Aborting due to %s failures" % EXECUTION_ATTEMPTS)


    def parse_log(self, log):
        """ Given an execution log, split it into layers and send the runs of
            neurons to their activation functions for conversion into states.
            """


        layer_borders = [index for index, count in 
                log.find_runs_for_page(self.spacer_symbol_page) 
                        if count == LAYER_BOUND_COUNT]


        # HACK: for now, trim the last layer border for certain networks explcitly.
        # TODO: generate this procedurally based on the number of layers we care about
        if self.model_index in [1, 2, 4, 5]:
            layer_borders = layer_borders[:-1] # There always seems to be an extra one at the end!

        # for mnist skip first pair, and last hanging one
        # actually, do this with giant initial start_offset
        #elif self.model_index in [6]:
        #    layer_borders = layer_borders[1:]

        # for each layer
        for layer_idx, layer_bounds in enumerate(chunks(layer_borders, 2)):

            # bad scan of boundaries -- happens rarely!
            if len(set(layer_bounds)) == 1:
                return False

            neurons = self.layers[layer_idx]
            layer_func = neurons[0].func

            try:
                # Get the states from the activation function's parse_layer() function
                states = layer_func.parse_layer(log, layer_bounds, self.enclave_elf)
            except RuntimeError as e:
                #logger.warning("error parsing layers %s: %s" % (layer_idx, e))
                return False

            # for layers that aren't implemented yet
            if states is None:
                continue

            # Make sure we recovered all the neurons for this layer;
            # returning false casues a re-run of the input
            if len(states) != len(neurons):
                #logger.warning("weird number of states: %s (vs %s neurons)" % (len(states), len(neurons)))
                return False

            # Load into the neuron
            for node_idx, state in enumerate(states):
                self.layers[layer_idx][node_idx].mode = state

        return True

    def execute_victim_with_input(self, guess):
        self.executions += 1
        args = " ".join(str(g) for g in guess)
        cmd = ["./run.sh", args]

        #sleep(0.001)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        # wait for the process to terminate
        out, _ = process.communicate()
        #sleep(0.001)

        # Make sure the command suceeded
        assert process.returncode == 0

        try:
            return out.decode("ascii")
        except UnicodeDecodeError as ude:
            raise(ude)

    def stats(self):
        logger.info("Recovery complete in %s executions" % self.executions)
        logger.info("%s incomplete logs" % self.incomplete_logs)
        logger.info("%s non-deterministic steps" % self.non_deterministic_steps)

    def recursive_calc(self, layer_idx, target, depth=0):

        # TODO: check if desired values are possible given the output range of the function!
        
        # base case -- we're at the input layer!
        if layer_idx == 0:
            return target

        # otherwise, compute the target matrix recursively
        else:
            # Assume all neurons in the same layer have the same activation function
            layer_func = self.layers[layer_idx][0].func.llfunc
            prev_layer_neurons = self.layers[layer_idx-1]
            # for example, if act is exp(), act_inv is np.log()
            act_inv = layer_func.inverse_func
            b = np.array([n.solved_bias for n in prev_layer_neurons], dtype=PRECISION)
            W = np.array([n.solved_weights for n in prev_layer_neurons], dtype=PRECISION)

            result = lstsq(W, act_inv(target)-b, rcond=None)[0]

            return self.recursive_calc(layer_idx-1, result, depth=depth+1)

