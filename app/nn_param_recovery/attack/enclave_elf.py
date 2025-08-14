import os
import re
import subprocess
from collections import defaultdict
from .util import round_down

NM_REGEX = br"([\w]+) \w ([^$]+)$"

class Enclave_ELF(object):

    def __init__(self):

        self.all_symbols = {}
        self.symboldict = []

        self._load_symbols()

    def _load_symbols(self):
        poc_root = os.path.join(os.path.dirname(__file__), "..")
        path = os.path.join(poc_root, "Enclave", "encl.so")

        # Extract all the pages from the enclave
        pages = defaultdict(dict)
        symboldict = {}
        cmd = f"nm -C {path}"
        output = subprocess.check_output(cmd, shell=True)
        for line in output.split(b"\n"):
            m = re.match(NM_REGEX, line)
            if m:
                offset, symbol = m.groups()
                offset = int(offset.decode("ascii"), 16)
                symbol = symbol.decode("ascii")
                self.all_symbols[offset] = symbol

                page = round_down(offset, 4096)
                symboldict[symbol] = page
        self.symboldict = symboldict

    def get_page_for_symbol(self, symbol):
        return self.symboldict[symbol]
