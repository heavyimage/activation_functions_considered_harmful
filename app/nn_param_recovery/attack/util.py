import os 
import re
import yaml
from collections import defaultdict
from .consts import *

# add logger
from .logger import get_logger
logger = get_logger(__file__)

# https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict/16439720#16439720
def dd():
    return defaultdict(dict)

def short_func(symbol):
    return re.sub("\([^\)]+\)$", "()", symbol).split("::")[-1]

def is_square(m):
    return len(m) == len(m[0])

def round_down(m, n):
    return m // n * n

def accuracy(guess, target):
    return 100 - abs((abs(target - guess) * 100.0) / target)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def banner(s):
    logger.info("===[ %s ]===" % (s))

def get_meta():
    """ A utility function to load the poc settings from the yaml file """
    attack_module = os.path.dirname(__file__)
    with open(os.path.join(attack_module, "..", SETTINGS)) as f:
        meta = yaml.safe_load(f)
        return meta

def truncate1(text, offset):
    return "\n".join(text.split("\n")[offset:])

def truncate2(text, offset):
    AVG_LENGTH = 27
    idx = offset * AVG_LENGTH
    text = text[idx:]
    text = text[text.index("\n")+1:]
    return text

def convert_to_preferred_format(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02dh %02dm %02ds" % (hour, min, sec) 

def abort_check():
    """ safely exit the program """
    if os.path.exists("/tmp/abort"):
        logger.critical("*** Aborting due to /tmp/abort! ***")
        import sys
        sys.exit()

