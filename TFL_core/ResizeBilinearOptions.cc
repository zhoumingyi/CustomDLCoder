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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
// clang-format off: Clang-format thinks this header is paired.
#include "tensorflow/lite/kernels/internal/optimized/resize_bilinear.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
using namespace tflite;
namespace randomname {

int32 size_raw=;
int32 size_dims_size=;
int32_t size_dims_raw=;
bool align_corners=;
bool half_pixel_centers=;
int32_t new_height=;
int32_t new_width=;

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

  const int32_t* size_dims_data;
  size_dims_data = size_dims_raw;

  const int32* size_data;
  size_data = size_raw;

  ResizeBilinearParams op_params;                            \
  op_params.align_corners = align_corners;                   \
  op_params.half_pixel_centers = half_pixel_centers;         \
  optimized_ops::ResizeBilinear(op_params, RuntimeShape(input_dims_size, input_dims_raw),                     \
               input_v, RuntimeShape(size_dims_size, size_dims_data), \
               size_data, RuntimeShape(output_dims_size, output_dims_raw),   \
               output_data);

  return output_data;
}
}  // namespace randomname