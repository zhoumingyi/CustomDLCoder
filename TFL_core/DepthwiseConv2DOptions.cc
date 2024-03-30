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


#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_multithread.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/examples/coder/include/common_funcs.h"


using namespace tflite;
using namespace optimized_ops;

namespace randomname {

filter_raw=;

bias_raw=;

filter_tensor_data=filter_raw;
bias_tensor_data=bias_raw;

bool has_conv_bias=;
const int stride_width=;
const int stride_height=;
const int padding_values_width=;
const int padding_values_height=;
const TfLiteFusedActivation activation=;
const int dilation_width_factor=;
const int dilation_height_factor=;
const int filter_dims_size=;
const int32_t filter_dims_raw=;
const int bias_dims_size=;
const int32_t bias_dims_raw=;
const TfLitePadding paddings=;
const TfLiteType filter_type=;
const TfLiteType bias_type=;
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
int32_t output_dims_raw=;
const int32_t output_num=;
const float scale_output=;
const int32_t zero_point_output=;
const TfLiteType output_type=;

auto* randomname(float* input_v, CpuBackendContext* cpu_backend_context) {
  //------------------------------------------------------------------------------
  // define params
  //------------------------------------------------------------------------------

  TfLiteDepthwiseConvParams data_params;
  ExtractDepthConvParams(paddings, stride_width, stride_height, dilation_width_factor, dilation_height_factor, activation, &data_params);
  TfLiteDepthwiseConvParams* params = &data_params;

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);

  //--------------------------------------------------------------------------------
  // calculate params
  //--------------------------------------------------------------------------------
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  TfLitePaddingValues padding;
  padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, input_dims_raw[1],
      input_dims_raw[2], filter_dims_raw[1], filter_dims_raw[2], paddings, &output_dims_raw[1], &output_dims_raw[2]);

  DepthwiseParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = padding.width;
  op_params.padding_values.height = padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  op_params.depth_multiplier = filter_dims_raw[3] / input_dims_raw[3];

  // auto* cpu_backend_context = new CpuBackendContext();
  // cpu_backend_context->SetMaxNumThreads(-1);

  optimized_ops::DepthwiseConv<float, float>(op_params, 
      RuntimeShape(input_dims_size, input_dims_raw), input_v,
      RuntimeShape(filter_dims_size, filter_dims_raw), filter_tensor_data,
      RuntimeShape(bias_dims_size, bias_dims_raw), bias_tensor_data,
      RuntimeShape(output_dims_size, output_dims_raw), output_data,
      cpu_backend_context);

  return output_data;
}
}
