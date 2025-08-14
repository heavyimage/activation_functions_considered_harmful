#!/usr/bin/python
import os
from attack.util import get_meta

def main():
    meta = get_meta()
    model_index = meta['USE_MODEL']
    model_settings = meta["models"][model_index]
    model_path = model_settings['model_path']

    poc_root = os.path.dirname(__file__)
    path = os.path.join(poc_root, "Enclave", "model_path.h")

    print("Using %s" % model_path)

    with open(path, "w") as f:
        f.write("#include \"%s\"" % model_path)
        f.write("\n")

if __name__ == "__main__":
    main()
