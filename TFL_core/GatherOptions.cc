/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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


#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

using namespace tflite;

namespace randomname {

input_raw=;

input_tensor_data=input_raw;

int axis=;
int batch_dims=;

const int positions_dims_size=;
const int32_t positions_dims_raw=;

const int input_dims_size=;
const int32_t input_dims_raw=;

const int output_dims_size=;
int32_t output_dims_raw=;
const int32_t output_num=;


auto* randomname(int* indexes) {
  bool indices_has_only_positive_elements = true;
  // if (positions_type == kTfLiteInt32){
  //   const size_t num_indices = positions->bytes / sizeof(PositionsT);
  // }
  // for (size_t i = 0; i < num_indices; i++) {
  //   if (indexes[i] < 0) {
  //     indices_has_only_positive_elements = false;
  //     break;
  //   }
  // }
  // // TF_LITE_ENSURE(context, indices_has_only_positive_elements);

  tflite::GatherParams op_params;
  op_params.axis = axis;
  op_params.batch_dims = batch_dims;

  float* output_data = (float*)malloc(sizeof(float) * output_num);

  optimized_ops::Gather(op_params, RuntimeShape(input_dims_size, input_dims_raw),
                        input_tensor_data, RuntimeShape(positions_dims_size, positions_dims_raw),
                        indexes,
                        RuntimeShape(output_dims_size, output_dims_raw), output_data);

  return output_data;
}
}
