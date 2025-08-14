import matplotlib.pyplot as plt
from itertools import groupby, count
from .util import get_meta, short_func

# add logger
from .logger import get_logger
logger = get_logger(__file__)

class Log(object):
    """ A class for interacting with poc's log """

    def __init__(self, log_text):

        # initialize some vars
        self.steps = []
        self.pages = []
        self.runs = []
        # NOTE: erips are not available in production mode --
        #       only used for graphing purposes :-)
        self.erips = []

        self.complete = True

        # process only the lines that are page accesses or errors!
        lines = []
        for line in log_text.split("\n"):
            if line.startswith("[main.c] 0x"):
                lines.append(line.strip())
            elif "excessive interrupt rate detected" in line:
                self.complete = False
                return

        for i, line in enumerate(lines[6:]):
            _, erip, page = line.split(" ")
            self.steps.append(i)
            self.erips.append(int(erip, 16))
            self.pages.append(int(page, 16))

        # Calculate runs
        self.calculate_runs()

    def stats(self):
        logger.debug("Log Stats:")
        logger.debug("%s total events" % len(self.pages))
        logger.debug("%s total page runs" % len(self.runs))
        logger.debug("")

    def calculate_runs(self):
        """ Once a log is parsed, we can calculate the runs of a given page """

        # Format is:
        #
        #   [
        #       (71680, 23),
        #       (149504, 14),
        #       (63488, 20),
        #       ...
        #   ]
        #
        #   where the first value is the page and the second is the number of
        #   steps spent there.

        self.runs = [(k, len(list(g))) for k, g in groupby(self.pages)]

    # TODO: expand this to work for patterns of multiple pages
    # TODO: document that it also includes the indicies of the finding
    def find_runs_for_page(self, poi, start=0):
        return [(i, length) for i, (page, length)
                in enumerate(self.runs[start:], start)
                        if page == poi]


    def graph_log(self, symbols, enclave_elf):
        fig, axs = plt.subplots(1, 1)

        # Figure out all the accessed pages
        accessed_pages = set(p//4096 for p in self.pages)

        # set ylim
        axs.set_ylim((min(accessed_pages))*4096, (max(accessed_pages)+1)*4096)

        # Plot the execution
        axs.plot(self.steps, self.erips, alpha=0.5, color="red", linestyle="dotted", label="Precise Enclave RIP Value\n(Debug mode only)")

        # draw "pages" in the middle of the page
        offset_pages = [p+2048 for p in self.pages]
        axs.plot(self.steps, offset_pages, alpha=0.5, color="black", label="Page Table Accesses")

        # try to plot runs?
        x = 0
        for run in self.runs:
            y, run_length = run
            axs.annotate(str(run_length), xy=(x - 10 + run_length/2.0, y+3000))
            x += run_length

        # pretty print the pages / the pages of interest
        poi = {}
        for symbol in symbols:
            page = enclave_elf.symboldict[symbol]
            symbol = short_func(symbol)
            if page not in poi:
                poi[page] = "Page containing %s" % symbol
            else:
                poi[page] = "%s, %s" % (poi[page], symbol)

        # Generate a palette with the correct number of pois
        discrete_cmap = plt.cm.get_cmap('viridis', len(poi))

        yrange = [y*4096 for y in range(min(accessed_pages), max(accessed_pages)+1)]
        sorted_pois = sorted(poi.keys())
        for page in yrange:
            if page in poi:
                color = discrete_cmap.colors[sorted_pois.index(page)]
                alpha = 0.5
                label = poi[page]
            else:
                color = "grey"
                alpha = 0.1
                label = None
            axs.axhspan(page, page+4096, alpha=alpha, color=color, label=label)

        # Create a ytick for each page
        axs.set_yticks(yrange)

        # setting label for y tick
        #axs.set_yticklabels("%s:%s" % (p, str(hex(p))) for p in yrange)
        axs.set_yticklabels("%s" % (str(hex(p))) for p in yrange)


        axs.set_ylabel("Enclave RIP")
        axs.set_xlabel("CPU Step Count\n(starting from performInference())")

        plt.suptitle("Visualizing Enclaved Neural Network Execution\nStep Count by Enclave RIP / Page")
        plt.legend()
        plt.show()

