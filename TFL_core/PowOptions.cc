#include <stddef.h>
// #include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
// #include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

using namespace tflite;

namespace randomname {

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
const TfLiteType output_type=;



auto* randomname(auguments_placeholder) {

  float* output_data = (float*)malloc(sizeof(float) * output_num);

  bool requires_broadcast;
  requires_broadcast = false;
  if (input_0_dims_size != input_1_dims_size) {
    requires_broadcast = true;
  } else {
    for (int i = 0; i < input_0_dims_size; ++i) {
      if (input_0_dims_raw[i] != input_1_dims_raw[i]) {
        requires_broadcast = true;
        break;
      }
    }
  }

  if (requires_broadcast) {
    optimized_ops::BroadcastPow4D(
        RuntimeShape(input_0_dims_size, input_0_dims_raw), input_v_0,
        RuntimeShape(input_1_dims_size, input_1_dims_raw), input_v_1,
        RuntimeShape(output_dims_size, output_dims_raw), output_data);
  } else {
    reference_ops::Pow(RuntimeShape(input_0_dims_size, input_0_dims_raw), input_v_0,
                       RuntimeShape(input_1_dims_size, input_1_dims_raw), input_v_1,
                       RuntimeShape(output_dims_size, output_dims_raw), output_data);
  }
  return output_data;
}
}