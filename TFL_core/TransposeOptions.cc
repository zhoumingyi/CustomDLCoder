#include <stdint.h>

#include "tensorflow/lite/c/common.h"
// #include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
using namespace tflite;
namespace randomname {

std::vector<int32_t> perm=;
int32_t perm_size=;

const int input_dims_size=;
const int32_t input_dims_raw=;


const int output_dims_size=;
const int32_t output_dims_raw=;
const int32_t output_num=;
// const float scale_output=;
// const int32_t zero_point_output=;
const TfLiteType output_type=;



auto* randomname(float* input_v) {

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);

  // const int* perm_data = perm;
  const int size = perm_size;
  TransposeParams params;
  params.perm_count = size;
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm[i];
  }

  optimized_ops::Transpose(params, RuntimeShape(input_dims_size, input_dims_raw), \
                  input_v,  \
                  RuntimeShape(output_dims_size, output_dims_raw),        \
                  output_data);

  return output_data;
}
}  // namespace randomname