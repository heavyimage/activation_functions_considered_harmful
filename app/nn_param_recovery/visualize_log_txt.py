from attack.poc import PoC
from attack.util import get_meta
from attack.enclave_elf import Enclave_ELF
from attack.log import Log

def main():
    #poc = PoC()
    e = Enclave_ELF()
    e._load_symbols()

    m = get_meta()
    poc_idx = m['USE_MODEL']
    symbols = m['models'][poc_idx]['symbols']
    offset = m['models'][poc_idx].get("start_offset", 0)

    # open the log
    with open("log.txt") as f:
        contents = [l.strip() for l in f.readlines()]
        contents = "\n".join(contents[offset:])
    log = Log(contents)


    log.graph_log(symbols, e)

if __name__ == "__main__":
    main()
