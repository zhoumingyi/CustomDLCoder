#include <stdint.h>
#include <stdlib.h>

#include <cmath>
#include <limits>

#include "tensorflow/lite/c/common.h"
// #include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

// #if __aarch64__ && __clang__
// #include <arm_neon.h>
// #endif
using namespace tflite;
namespace randomname {
const int input_dims_size=;
const int32_t input_dims_raw=;
// const float scale_input=;
// const int32_t zero_point_input=;
// const TfLiteType input_type=;


const int output_dims_size=;
const int32_t output_dims_raw=;
const int32_t output_num=;
// const float scale_output=;
// const int32_t zero_point_output=;
const TfLiteType output_type=;

template <typename T>
void EvalImpl(const T* input_v,
                             T* output_data,
                             std::function<T(T)> func) {
  const int64_t num_elements = output_num;
  for (int64_t i = 0; i < num_elements; ++i) {
    output_data[i] = func(input_v[i]);
  }
//   return kTfLiteOk;
}


auto* randomname(float* input_v) {

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);
  switch (output_type) {
    case kTfLiteFloat32:
          EvalImpl<float>(input_v, output_data, [](float f) { return 1.f / std::sqrt(f); });
  }
  return output_data;
}
}  // namespace randomname