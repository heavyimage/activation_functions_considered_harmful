# Modified from generate_cc_arrays.py in tensorflow
#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for converting .tflite, .bmp and .wav files to cc arrays."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import struct
import wave
import numpy as np

def generate_file(out_fname, array_type, array_contents, size):
    """Write an array of values to a CC or header file."""
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    print("\twriting %s ..." % out_fname, end="")

    if out_fname.endswith('.cc'):
        out_cc_file = open(out_fname, 'w')
        out_cc_file.write('#include <cstdint>\n\n')
        out_cc_file.write('#include "{}"\n\n'.format(
                os.path.basename(out_fname.split('genfiles/')[-1]).replace('.cc', '.h')))
        # Define a constant if the model is quantized or not
        #if "quantized" in out_fname:
        #    out_cc_file.write('#define QUANTIZED 1\n\n')
        #else:
        #    out_cc_file.write('#define QUANTIZED 0\n\n')

        out_cc_file.write('alignas(16) const {} {}[] = {{'.format(
                array_type, "model_data"))
        out_cc_file.write(array_contents)
        out_cc_file.write('};\n')
        out_cc_file.close()
    elif out_fname.endswith('.h'):
        out_hdr_file = open(out_fname, 'w')
        out_hdr_file.write('#include <cstdint>\n\n')
        out_hdr_file.write('constexpr unsigned int {}_size = {};\n'.format(
                "model_data", str(size)))
        out_hdr_file.write('extern const {} {}[];\n'.format(
                array_type, "model_data"))
        out_hdr_file.close()
    else:
        raise ValueError('generated file must be end with .cc or .h')
    print("done!")


def bytes_to_hexstring(buffer):
    """Convert a byte array to a hex string."""
    hex_values = [hex(buffer[i]) for i in range(len(buffer))]
    out_string = ','.join(hex_values)
    return out_string


def generate_array(input_fname):
    """Return array size and array of data from the input file."""
    if input_fname.endswith('.tflite'):
        with open(input_fname, 'rb') as input_file:
            buffer = input_file.read()
        size = len(buffer)
        out_string = bytes_to_hexstring(buffer)
        return [size, out_string]

    else:
        raise ValueError('input file must be .tflite, .bmp, .wav or .csv')


def get_array_name(input_fname):
    # Normalize potential relative path to remove additional dot.
    abs_fname = os.path.abspath(input_fname)
    base_array_name = 'g_' + abs_fname.split('.')[-2].split('/')[-1]
    if input_fname.endswith('.tflite'):
        return [base_array_name + '_model_data', 'unsigned char']


def main():
    """Create cc sources with c arrays with data from each .tflite or .bmp."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'output',
            help='base directory for all outputs or a cc or header to generate.')
    parser.add_argument(
            'inputs',
            nargs='+',
            help='input wav, bmp or tflite files to convert. '
            'If output is a cc or header only one input may be specified.')
    args = parser.parse_args()

    # Deduplicate inputs to prevent duplicate generated files (ODR issue).
    for input_file in list(dict.fromkeys(args.inputs)):
        print("Processing %s..." % os.path.basename(input_file))
        output_base_fname = os.path.join(args.output, os.path.splitext(input_file)[0])
        if input_file.endswith('.tflite'):
            output_base_fname = output_base_fname + '_model_data'
        else:
            raise ValueError("Don't know how to convert input file!")

        output_cc_fname = output_base_fname + '.cc'
        # Print output cc filename for Make to include it in the build.
        output_hdr_fname = output_base_fname + '.h'
        size, cc_array = generate_array(input_file)
        _, array_type = get_array_name(input_file)
        generate_file(output_cc_fname, array_type, cc_array, size)
        generate_file(output_hdr_fname, array_type, cc_array, size)


if __name__ == '__main__':
    main()
