/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include <string.h>
#include <sgx_trts.h>

#include "encl.h"

#include <stdio.h> /* vsnprintf */
#include <string.h>

#define TF_LITE_STATIC_MEMORY
#include "tensorflow/lite/core/c/common.h"
#include "model_path.h" // generated at compile time; loads model data!
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

static void __text_dummy1(void) __attribute__((__used__, __aligned__(4096)));
static void __text_dummy1(void) {}

namespace {
    // #1: Adjust this number ----------------------------------v
    using HelloWorldOpResolver = tflite::MicroMutableOpResolver<4>;

    // #2: be sure to add a resolver for each layer per netron
    TfLiteStatus RegisterOps(HelloWorldOpResolver& op_resolver) {
        TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
        TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
		TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
		TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
        return kTfLiteOk;
    }
    // #3: need to run through standalone to see error messages!
}

static void __text_dummy2(void) __attribute__((__used__, __aligned__(4096)));
static void __text_dummy2(void) {}


TfLiteStatus performInference(tflite::MicroInterpreter interpreter,
							  float* inputs,
							  size_t num_inputs,
                              int& predicted_class) {

    // Obtain a pointer to the model's input tensor
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // manually fill
    for (int k=0; k<num_inputs; k++){
        input->data.f[k] = *(inputs+k);;
    }

    // Perform inference
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());

    // argmax on output probs
    float maxval = -1.0;
    int idx = -1;
    for (int i=0; i<10; i++){
        if (output->data.f[i] > maxval){
            maxval = output->data.f[i];
            idx = i;
        }
    }
    //printf("maxval = %f\n", maxval);
    //printf("class idx / prediction = %d\n", idx);
    predicted_class = idx;

    return kTfLiteOk;
}

static void __text_dummy3(void) __attribute__((__used__, __aligned__(4096)));
static void __text_dummy3(void) {}

int entry_point(int num_runs, float* inputs, size_t num_inputs){

    tflite::InitializeTarget();

    // Build the model
    const tflite::Model* model = ::tflite::GetModel(model_data);
    TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

    // Make an op resolver
    HelloWorldOpResolver op_resolver;
    TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

    // Arena size just a round number. The exact arena usage can be determined
    // using the RecordingMicroInterpreter.
    //
    // this seems like the largest power of 2 I can use...
    constexpr int kTensorArenaSize = 128 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];

    // Make the interpreter
    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);

    // ensure the tensors are allocated
    TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

    int predicted_class = 0;
    //for (int i=0; i < num_runs; i++){
    TF_LITE_ENSURE_STATUS(performInference(interpreter, inputs, num_inputs, predicted_class));
    //}
    return predicted_class;
}

static void __text_dummy4(void) __attribute__((__used__, __aligned__(4096)));
static void __text_dummy4(void) {}

// from: https://github.com/intel/linux-sgx/issues/468
/*
 * printf:
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */
int printf(const char* fmt, ...) {
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}
