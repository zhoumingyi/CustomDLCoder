/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdio>
#include <iostream>
// #include <sys/time.h>
// #include <time.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
// #include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/optimized/multithreaded_conv.h"
#include "tensorflow/lite/kernels/eigen_support.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
// const int num = 150528;

using namespace tflite;

extern "C" const float* tflite_minimal(char* path, int* input_v, int num_input, int num_output) {
// int main(int argc, char* argv[]) {
  // srand(time(NULL));
  const char* filename = path;
  // std::cout << "TFLite model: " << filename << std::endl;

  // const int num = atoi(argv[2]);
  // std::cout << "runs in here #0 " << std::endl;
  // float input_v[num*num*3]={0};
  // for(int i = 0; i < (num*num*3); i++)
  // {
  //   input_v[i] = rand() % 256;
  // }
  // timeval t_start, t_end;
  // gettimeofday( &t_start, NULL);
  // for(int i = 0; i < 100; i++)
  // {
    // std::cout << "runs in here #1 " << std::endl;
    // Load model
    // timeval t_start, t_end;
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(filename);
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    // gettimeofday( &t_start, NULL);
    interpreter->AllocateTensors();
    // std::cout << "runs in here #3 " << std::endl;
    int* input = interpreter->typed_input_tensor<int>(0);
    memcpy(input, input_v, 1*num_input*sizeof(int));
    // std::cout << "runs in here #4 " << std::endl;
    interpreter->Invoke();
    float* output = interpreter->typed_output_tensor<float>(0);
    float* output_f = new float[num_output];
    memcpy(output_f, output, 1*num_output*sizeof(float));
  // }
  // std::cout << "runs in here #5 " << std::endl;
  // gettimeofday( &t_end, NULL);
  // double delta_t = (t_end.tv_sec-t_start.tv_sec)*1000.0 +
  //                 (t_end.tv_usec-t_start.tv_usec)/1000.0;
  return output_f;
  // return 0;
}
