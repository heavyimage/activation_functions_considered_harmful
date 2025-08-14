from attack.poc import PoC
import signal
import sys
from time import sleep

def signal_handler(sig, frame):
    print("ctrl+c'ing the process will cause SGX-step to hang.")
    print("Instead, `touch /tmp/abort` and wait for the process to abort.")

def main():
    signal.signal(signal.SIGINT, signal_handler)
    poc = PoC()
    #poc.network.visualize()
    poc.attack()


if __name__ == "__main__":
    main()
