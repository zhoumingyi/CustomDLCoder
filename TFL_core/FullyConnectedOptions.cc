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

#include "tensorflow/lite/kernels/internal/optimized/integer_ops/fully_connected.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/optimized/sparse_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/reference/sparse_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/examples/coder/include/common_funcs.h"

using namespace tflite;

namespace randomname{

filter_raw=;

bias_raw=;

filter_tensor_data=filter_raw;
bias_tensor_data=bias_raw;

bool has_conv_bias=;
const TfLiteFusedActivation activation=;
const TfLiteFullyConnectedWeightsFormat weights_format=;
const bool keep_num_dims=;
const bool asymmetric_quantize_inputs=;
const TfLiteType filter_type=;
const TfLiteType bias_type=;
const int32_t filter_dims_raw=;
const int filter_dims_size=;
const int32_t bias_dims_raw=;
const int bias_dims_size=;

const float scale_filter=;
const int32_t zero_point_filter=;
const float scale_bias=;
const int32_t zero_point_bias=;

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


void ExtractFullyConnectedParams(
                               TfLiteFullyConnectedWeightsFormat weights_format,
                               bool keep_num_dims, bool asymmetric_quantize_inputs,
                               TfLiteFusedActivation activation,
                               TfLiteFullyConnectedParams* data_params) {
  // TfLiteFullyConnectedParams data_params;
  data_params->weights_format = weights_format;
  data_params->keep_num_dims = keep_num_dims;
  data_params->asymmetric_quantize_inputs = asymmetric_quantize_inputs;
  data_params->activation = activation;
  // return data_params;
}



auto* randomname(float* input_v,input_2_placeholder,CpuBackendContext* cpu_backend_context) {
  //------------------------------------------------------------------------------
  // define params
  //------------------------------------------------------------------------------
  TfLiteFullyConnectedParams data_params;
  ExtractFullyConnectedParams(weights_format, keep_num_dims,
                              asymmetric_quantize_inputs, activation, &data_params);
  TfLiteFullyConnectedParams* params = &data_params;

  //--------------------------------------------------------------------------------
  // define output
  //--------------------------------------------------------------------------------

  // int output_num = 1;
  // for (int i = 0; i < output_dims_size; ++i) {
  //   output_num *= output_dims_raw[i];
  // }
  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);


  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  op_params.lhs_cacheable=true;
  op_params.rhs_cacheable=false;

  // const TfLiteTensor* bias = nullptr;

  optimized_ops::FullyConnected(op_params, \
      RuntimeShape(input_dims_size, input_dims_raw), input_v, \
      RuntimeShape(filter_dims_size, filter_dims_raw), filter_tensor_data, \
      RuntimeShape(bias_dims_size, bias_dims_raw), bias_tensor_data, \
      RuntimeShape(output_dims_size, output_dims_raw), output_data, \
      cpu_backend_context);
  return output_data;
}
}
