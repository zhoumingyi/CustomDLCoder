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
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/examples/coder/include/common_funcs.h"

using namespace tflite;
// using namespace optimized_ops;
namespace randomname {

const TfLiteFusedActivation activation=;
const TfLitePadding paddings=;
const int stride_height=;
const int stride_width=;
const int filter_height=;
const int filter_width=;
const int padding_values_height=;
const int padding_values_width=;

const int input_dims_size=;
const int32_t input_dims_raw=;
const float scale_input=;
const int32_t zero_point_input=;
const TfLiteType input_type=;


const int output_dims_size=;
const int32_t output_dims_raw=;
const int32_t output_num=;
const float scale_output=;
const int32_t zero_point_output=;
const TfLiteType output_type=;


auto* randomname(float* input_v) {

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);

  float activation_min, activation_max;
  CalculateActivationRange(activation, &activation_min,
                           &activation_max);
  tflite::PoolParams op_params;               
  op_params.stride_height = stride_height;    
  op_params.stride_width = stride_width;         
  op_params.filter_height = filter_height;  
  op_params.filter_width = filter_width;   
  op_params.padding_values.height = padding_values_height;  
  op_params.padding_values.width = padding_values_width;   
  op_params.float_activation_min = activation_min;     
  op_params.float_activation_max = activation_max;   
  optimized_ops::AveragePool(op_params, RuntimeShape(input_dims_size, input_dims_raw), input_v,
                RuntimeShape(output_dims_size, output_dims_raw), output_data);

  return output_data;
}
}  // namespace randomname