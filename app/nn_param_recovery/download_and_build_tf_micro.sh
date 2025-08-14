# clone the library
git clone git@github.com:heavyimage/tflite-micro.git Enclave/tflite-micro

# build the library (this will take a while)
cd Enclave/tflite-micro
make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=release
