import logging
from colorlog import ColoredFormatter

# constants
LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s"

# setup the stream
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(
        LOGFORMAT,
        datefmt="%m-%d %H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
            },
        )
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)

def get_logger(identifier):
    log = logging.getLogger(identifier)
    log.setLevel(LOG_LEVEL)
    log.addHandler(stream)
    return log

# USAGE:
#
# from .logger import get_logger
# logger = get_logger(__file__)
#
# logger.debug("A quirky message only developers care about")
# logger.info("Curious users might want to know this")
# logger.warning("Something is wrong and any user should be informed")
# logger.error("Serious stuff, this is red for a reason")
# logger.critical("OH NO everything is on fire")
