from .consts import *
import numpy as np

# For higher level TF Activation functions

class TFActivation(object):
    def __init__(self):
        self.last_total = 0
        self.last_output = 0
        self.last_mode = 0 
        self.attack_mode = None

    @classmethod
    def get_instance(cls, subcls_name):

        def get_all_subclasses(cls, results={}):
            results[cls.__name__.lower()] = cls
            for scls in cls.__subclasses__():
                get_all_subclasses(scls, results=results)
            return results


        subclasses = get_all_subclasses(cls)
        return subclasses[subcls_name]()

    def is_solvable(self):
        return False

    def get_boundary(self, boundary):
        if self.attack_mode == HIGH_LEVEL:
            return self.BOUNDARY_POINTS[boundary]
        elif self.attack_mode == LOW_LEVEL:
            return self.llfunc.BOUNDARY_POINTS[boundary]
        else:
            raise RuntimeError("Impossible attack mode!")

    def generate_mini_log(self, log, layer_bounds, af_page):
        start, end = layer_bounds

        # find only the events in the log that are af hits
        mini_log = []
        for run_idx in range(start, end+1):
            offset, count = log.runs[run_idx]
            if offset == af_page:
                mini_log.append(run_idx)

        return mini_log

    def define_boundary(self, sign):
        raise NotImplementedError("need to define boundaries for this activation function!")

# Stub -- can't attack yet!
class TFSoftMax(TFActivation):
    def __init__(self):
        super().__init__()
    def parse_layer(self, log, layer_bounds, e):
        return None

class TFLinear(TFActivation):
    def __init__(self):
        super().__init__()
        self.llfunc = Linear()

class TFRelu(TFActivation):
    def __init__(self):
        super().__init__()
        self.llfunc = Relu()

    def parse_layer(self, log, layer_bounds, e):
        return None

class TFSigmoid(TFActivation):
    CUTOFF_UPPER = PRECISION(16.619047164916992188)
    CUTOFF_LOWER = PRECISION(-9.0)

    ONE = "One"
    SIG = "Sigmoid"
    EXP = "Exp"

    BOUNDARY_POINTS = {
        (ONE, SIG): CUTOFF_UPPER,
        (SIG, EXP): CUTOFF_LOWER,
    }

    @classmethod
    def sigmoid_mode(cls, value):
        if value > cls.CUTOFF_UPPER:
            return cls.ONE
        elif value < cls.CUTOFF_LOWER:
            return cls.EXP
        else:
            return cls.SIG

    @classmethod
    def tf_sigmoid_decode(cls, sequence):
        """ Given a sequence of instruction count "gaps" between a tf sigmoid
            activation, work out what the neuron modes must have been.
            """

        # Position names
        START = "starting"
        MIDDLE = "middle"
        END = "ending"

        # The basic states, without the complication of ONE states
        RULES = {
            START: {
                25: cls. EXP, # startup
                26: cls.SIG, # startup
            },
            MIDDLE: {
                13: cls.EXP, # middle
                19: cls.SIG, # middle
            },
            END: {
                17: cls.EXP, # end
                22: cls.SIG, # end
            }
        }

        # Store the special subtraction values used in different modes
        SUBTRACTION  = {
            START: 25,
            MIDDLE: 13,
        }

        def run_of_ONEs(mode, quotient, remainder):
            """ Given a quotient and remainder, determine the number of ones to add
                alongside whichever other symbols.
                """

            # reminder can only be 5 or 6 in the middle but should never occur at the
            # start so we should be okay!
            if remainder in [0, 5]:
                return [cls.ONE] * quotient + [cls.EXP]

            if remainder in [1, 6]:
                return [cls.ONE] * quotient + [cls.SIG]

            raise RuntimeError(f"Impossible {mode} remainder: %s!" % remainder)

        output = []

        # Iterate over the sequence
        for i, val in enumerate(sequence):

            # We handle start and middle tokens in almost the same way
            if i != len(sequence)-1:

                # determine start or middle mode
                mode = START if i == 0 else MIDDLE

                # basic token
                token = RULES[mode].get(val, None)
                if token is not None:
                    addition = [token]

                # handle sequences of ONEs
                else:
                    # subtraction value is different at start vs middle
                    check_val = val - SUBTRACTION[mode]
                    quotient, remainder = divmod(check_val, 7)
                    addition = run_of_ONEs(mode, quotient, remainder)

            # Handle the last token a bit differently
            else:
                token = RULES[END].get(val, None)
                if token is not None:
                    # if it's just a normal value, use it as a check
                    if output[-1] != token:
                        raise RuntimeError("Check Failed!")

                # handle trailing ONEs
                else:
                    # here the subtraction value depends on the previous item
                    if len(output) == 0: # all zeros!
                        sub = 29
                    elif output[-1] == cls.EXP:
                        sub = 17
                    elif output[-1] == cls.SIG:
                        sub = 19
                    check_val = val - sub
                    quotient = check_val // 7
                    run = [cls.ONE] * quotient
                    output.extend(run)

                # now, return the string
                return output

            # keep extending output
            output.extend(addition)

    def __init__(self):
        super().__init__()
        self.llfunc = Sigmoid()

        # sigmoid can be attack at a "high level" eg against the tensorflow
        # timing or at a lower level eg against the underlying math library.
        self.attack_mode = HIGH_LEVEL

    def is_solvable(self):
        return True

    def mode_to_sign(self, mode):
        if self.attack_mode == HIGH_LEVEL:
            if mode == self.ONE:
                return POSITIVE
            elif mode == self.EXP:
                return NEGATIVE
            else:
                return False
        else:
            raise NotImplementedError("need to implement mode_to_sign() for low level attack")

    def define_boundary(self, sign):
        if self.attack_mode == HIGH_LEVEL:
            if sign == POSITIVE:
                return (self.ONE, self.SIG)
            elif sign == NEGATIVE:
                return (self.SIG, self.EXP)
            else:
                raise RuntimeError("Impossible Sign")
        else:
            raise NotImplementedError("need to implement define_boundary() for low level attack")

    def decide_direction(self, sign, mode):
        if self.attack_mode == HIGH_LEVEL:
            if sign == POSITIVE and mode == self.ONE:
                return DOWN
            elif sign == NEGATIVE and mode == self.EXP:
                return DOWN
            elif mode in self.SIG:
                return UP
            else:
                raise RuntimeError("in case %s; signs=%s" % (mode, sign))
        else:
            raise NotImplementedError("need to implement decide_direction() for low level attack")

    def parse_layer(self, log, layer_bounds, e):

        # calculate the page for this activation function
        af_page = e.get_page_for_symbol(self.llfunc.symbol)

        # Generate the mini_log
        mini_log = self.generate_mini_log(log, layer_bounds, af_page)

        neuron_states = []
        padding = []

        # Special case -- no hits, must be N ONE states
        if len(mini_log) == 0:
            # Find LogisticEval area where no neurons are firing...
            _, end = layer_bounds
            _, count = log.runs[end-3]
            padding = [count]

        # Otherwise, at least one activation function call...
        else:
            # use the mini log to find just events just before to just
            # after the run of af calls
            first_index = min(mini_log)-1
            last_index = max(mini_log)+1

            # read the mini-log into states + padding
            for run_idx in range(first_index, last_index+1):
                offset, count = log.runs[run_idx]
                if offset != af_page:
                    padding.append(count)

                # Used for low-level attack
                else:
                    neuron_state = self.llfunc.SGX_STEP_STATES.get(count, None)
                    neuron_states.append(neuron_state)

        # convert padding into high level tfsigmoid states!
        tf_states = TFSigmoid.tf_sigmoid_decode(padding)

        if self.attack_mode == HIGH_LEVEL:
            return tf_states
        else:
            # Marry the neuron state and decoded padding into actionable information
            result = []
            try:
                for function_mode in tf_states:
                    if function_mode == "1":
                        state = ONE
                    else:
                        state = neuron_states.pop(0)
                    result.append((function_mode, state))
            except IndexError as e:
                raise RuntimeError("Bad list")

            return result

class TFExponential(TFActivation):
    def __init__(self):
        super().__init__()
        self.llfunc = Exp()
        self.attack_mode = LOW_LEVEL

    def is_solvable(self):
        return True

    def mode_to_sign(self, mode):
        if mode == self.llfunc.OVERFLOW:
            return POSITIVE
        elif mode == self.llfunc.UNDERFLOW:
            return NEGATIVE
        else:
            return False

    def define_boundary(self, sign):
        if sign == POSITIVE:
            return (self.llfunc.NORMAL_POS, self.llfunc.OVERFLOW)
        elif sign == NEGATIVE:
            return (self.llfunc.UNDERFLOW, self.llfunc.SPECIAL_2)
        else:
            raise RuntimeError("Impossible Sign")

    def decide_direction(self, sign, mode):
        if sign == POSITIVE and mode == self.llfunc.OVERFLOW:
            return DOWN
        elif sign == NEGATIVE and mode == self.llfunc.UNDERFLOW:
            return DOWN
        elif mode in self.llfunc.ALL_NORMAL:
            return UP
        else:
            raise RuntimeError("in case %s; signs=%s" % (mode, sign))

    def parse_layer(self, log, layer_bounds, e):

        # calculate the page for this activation function
        af_page = e.get_page_for_symbol(self.llfunc.symbol)

        # Generate the mini_log
        mini_log = self.generate_mini_log(log, layer_bounds, af_page)
        if len(mini_log) == 0:
            raise RuntimeError("Empty mini_log")

        # use the mini log to find just events just before to just
        # after the run of af calls
        first_index = min(mini_log)-1
        last_index = max(mini_log)+1

        # read the mini-log into counts
        counts = []
        for run_idx in range(first_index, last_index+1):
            offset, count = log.runs[run_idx]
            if offset == af_page:
                counts.append(count)

        neuron_states = [self.llfunc.SGX_STEP_STATES.get(count, None) for count in counts]

        return neuron_states


class TFTanh(TFActivation):
    def __init__(self):
        super().__init__()
        self.llfunc = Tanh()



# For lower level mathematical functions...
# from https://en.wikipedia.org/wiki/Activation_function
class LLMathFunc(object):
    def __init__(self):
        self.name = ""
        self.symbol = None
        self.bounds = []

    @classmethod
    def get_instance(cls, subcls_name):

        def get_all_subclasses(cls, results={}):
            results[cls.__name__.lower()] = cls
            for scls in cls.__subclasses__():
                get_all_subclasses(scls, results=results)
            return results


        subclasses = get_all_subclasses(cls)
        return subclasses[subcls_name]()

    
class Linear(LLMathFunc):
    def __init__(self):
        super().__init__()
        self.bounds = [-np.inf, np.inf]
        self.name = "linear"

class Relu(LLMathFunc):
    def __init__(self):
        super().__init__()
        self.bounds = [0, np.inf]
        self.name = "relu"




class ExpBase(LLMathFunc):

    # NOTE: all of this was derived using the measurement_and_graphing branch

    # first, define names for states
    NAN = "NaN"
    UNDERFLOW = "Underflow"
    SPECIAL_2 = "Special #2"
    SPECIAL_1 = "Special #1"
    NORMAL_NEG = "Normal (-)"
    INNER3_NEG = "Inner3 (-)"
    INNER2_NEG = "Inner2 (-)"
    INNER1 = "Inner1"
    INNER2_POS = "Inner2 (+)"
    INNER3_POS = "Inner3 (+)"
    NORMAL_POS = "Normal (+)"
    OVERFLOW = "Overflow"

    # Create a name for the combined NORMAL, INNER2, and INNER3
    NORMAL = "Normal"
    INNER2 = "Inner2"
    INNER3 = "Inner3"

    # create an alias for all 'normal' modes
    ALL_NORMAL = [SPECIAL_2, SPECIAL_1, NORMAL, INNER3, INNER2, INNER1]

    # Now give them cycle counts
    GDB_STATES = {
        # Error states!
        13: NAN,

        # TODO: include +inf/-inf?

        # The useful states
        # Note that there is no way to differentiate 
        #   pos and neg variants since the counts are the same :-(
        17: OVERFLOW,
        18: UNDERFLOW,
        20: INNER1,
        34: INNER2,
        54: INNER3,
        59: NORMAL,
        61: SPECIAL_1,
        66: SPECIAL_2,
    }
    assert len(GDB_STATES) == 9, "State Collision"

    # Now give them cycle counts
    SGX_STEP_STATES = {
        # Error states!
        11: NAN, # I think this is correct

        # TODO: include +inf/-inf?

        # The useful states
        # Note that there is no way to differentiate 
        #   pos and neg variants since the counts are the same :-(
        15: OVERFLOW,
        16: UNDERFLOW,
        17: INNER1,
        31: INNER2,
        49: INNER3,
        54: NORMAL,
        56: SPECIAL_1,
        62: SPECIAL_2,
    }
    assert len(SGX_STEP_STATES) == 9, "State Collision"


    B1 = 88.72283554077148
    B2 = 1.0397207140922546
    B3 = 0.346573606133461
    B4 = 3.7252901874396116e-09


    BOUNDARY_POINTS = {
        # negative:
        (UNDERFLOW, SPECIAL_2): -103.97208786010742,
        (SPECIAL_2, SPECIAL_1): PRECISION(-B1),
        (SPECIAL_1, NORMAL_NEG): -86.98997116088867,
        (NORMAL_NEG, INNER3_NEG): PRECISION(-B2),
        (INNER3_NEG, INNER2_NEG): PRECISION(-B3),
        (INNER2_NEG, INNER1): PRECISION(-B4),

        # positive:
        (INNER1, INNER2_POS): PRECISION(B4),
        (INNER2_POS, INNER3_POS): PRECISION(B3),
        (INNER3_POS, NORMAL_POS): PRECISION(B2),
        (NORMAL_POS, OVERFLOW): PRECISION(B1),
    }

class Exp(ExpBase):
    def __init__(self):
        super().__init__()
        self.name = "exp"
        self.symbol = "expf"
        self.bounds = [0.0, np.inf]
        self.exp_direction = POSITIVE
        self.inverse_func = np.log

class Sigmoid(ExpBase):
    def __init__(self):
        super().__init__()
        self.name = "sigmoid"
        self.symbol = "expf"
        self.bounds = [0.0, 1.0]
        self.exp_direction = NEGATIVE
        self.inverse_func = np.vectorize(lambda x: np.log(np.divide(x, 1 - x)))

class Tanh(ExpBase):
    def __init__(self):
        super().__init__()
        self.name = "tanh"
        self.symbol = "tanh"
        self.bounds = [-1.0, 1.0]
        # tanh has both but we return the positive one for now
        self.exp_direction = POSITIVE
        self.inverse_func = np.arctanh

