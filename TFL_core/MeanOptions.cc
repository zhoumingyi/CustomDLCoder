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
#include "tensorflow/lite/kernels/internal/reference/reduce.h"

#include <stddef.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
using namespace tflite;

namespace randomname {

const int32_t num_axis=;
int32_t axis=;
const bool keep_dims=;

// const int input_dims_size=;
// const int32_t input_dims_raw=;
const int input_dims_size=;
const int32_t input_dims_raw=;


const int output_dims_size=;
const int32_t output_dims_raw=;
const int32_t output_num=;


void ResolveAxis(const int* axis_data, int axis_count,
                 tflite::MeanParams* op_params) {
  int i = 0;
  for (; i < axis_count; ++i) {
    op_params->axis[i] = static_cast<int16>(axis_data[i]);
  }
  for (; i < 4; ++i) {
    op_params->axis[i] = 1;
  }
}


// auto* randomname(float* input_v_0, float* input_v_1) {
auto* randomname(float* input_v) {

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);

  tflite::MeanParams op_params;
  op_params.axis_count = num_axis;
  ResolveAxis(axis, num_axis, &op_params);
  // const TfLiteTensor* input = op_context.input;

  if (keep_dims && input_dims_size == 4 &&
      op_params.axis_count == 2 &&
      ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
        (op_params.axis[0] == 2 && op_params.axis[1] == 1))) {
    reference_ops::Mean(op_params, RuntimeShape(input_dims_size, input_dims_raw), input_v,
                        RuntimeShape(output_dims_size, output_dims_raw),
                        output_data);
}
  return output_data;
}  // namespace randomname