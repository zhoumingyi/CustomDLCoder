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
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/add.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
using namespace tflite;

namespace randomname {


const TfLiteFusedActivation activation=;
const bool pot_scale_int16=;

const int input_0_dims_size=;
const int32_t input_0_dims_raw=;

input_0_raw=;
input_v_0=input_0_raw;

const int input_1_dims_size=;
const int32_t input_1_dims_raw=;

input_1_raw=;
input_v_1=input_1_raw;

const int output_dims_size=;
const int32_t output_dims_raw=;
const int32_t output_num=;
const float scale_output=;
const int32_t zero_point_output=;
const TfLiteType output_type=;


// auto* randomname(float* input_v_0, float* input_v_1) {
auto* randomname(auguments_placeholder) {

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);

  tflite::ArithmeticParams op_params;
  const bool need_broadcast = optimized_ops::ProcessBroadcastShapes(
      RuntimeShape(input_0_dims_size, input_0_dims_raw), RuntimeShape(input_1_dims_size, input_1_dims_raw), &op_params);
  float output_activation_min, output_activation_max;                \
  CalculateActivationRange(activation, &output_activation_min,   \
                           &output_activation_max);                      \
  SetActivationParams(output_activation_min, output_activation_max,      \
                      &op_params);
  if (need_broadcast) {
    optimized_ops::BroadcastAddDispatch(op_params, RuntimeShape(input_0_dims_size, input_0_dims_raw),
               input_v_0, RuntimeShape(input_1_dims_size, input_1_dims_raw),
               input_v_1, RuntimeShape(output_dims_size, output_dims_raw),
               output_data);
  } else {
    optimized_ops::Add(op_params, RuntimeShape(input_0_dims_size, input_0_dims_raw),
               input_v_0, RuntimeShape(input_1_dims_size, input_1_dims_raw),
               input_v_1, RuntimeShape(output_dims_size, output_dims_raw),
               output_data);
  }


  return output_data;
}
}  // namespace randomname