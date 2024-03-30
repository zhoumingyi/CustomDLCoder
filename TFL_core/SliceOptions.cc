#include <stdint.h>

// #include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
// #include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_type.h"
using namespace tflite;
namespace randomname {

std::vector<int> begins=;
std::vector<int> sizes=;

const int input_dims_size=;
const int32_t input_dims_raw=;


const int output_dims_size=;
const int32_t output_dims_raw=;
const int32_t output_num=;
const float scale_output=;
const int32_t zero_point_output=;
const TfLiteType output_type=;

const int kMaxDim = 5;

auto* randomname(float* input_v) {

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);


  // std::vector<int> begins;
  // begins.reserve(kMaxDim);
  // std::vector<int> sizes;
  // sizes.reserve(kMaxDim);

  // for (int i = NumDimensions(input); i < kMaxDim; ++i) {
  //   begins.push_back(0);
  //   sizes.push_back(1);
  // }

  tflite::SliceParams op_params;
  op_params.begin_count = kMaxDim;
  op_params.size_count = kMaxDim;
  for (int i = 0; i < kMaxDim; ++i) {
    op_params.begin[i] = begins[i];
    op_params.size[i] = sizes[i];
  }

  // if (kernel_type == kGenericOptimized) {
  optimized_ops::Slice<float>(op_params, RuntimeShape(input_dims_size, input_dims_raw), input_v,
                              RuntimeShape(output_dims_size, output_dims_raw), output_data);
  // } else {
  //   reference_ops::Slice<float>(op_params, GetTensorShape(input), input,
  //                                   GetTensorShape(output), output);
  // }

  return output_data;
}
}  // namespace randomname