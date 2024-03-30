#include "tensorflow/lite/kernels/internal/reference/sub.h"

// #include <stddef.h>
// #include <stdint.h>

#include <algorithm>
// #include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
// #include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
// #include "tensorflow/lite/kernels/internal/optimized/integer_ops/sub.h"
// #include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
// #include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
// #include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
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



auto* randomname(auguments_placeholder) {

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);

  bool requires_broadcast = false;
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

  float output_activation_min, output_activation_max;
  CalculateActivationRange(activation, &output_activation_min,
                           &output_activation_max);
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);

  if (requires_broadcast) {
    reference_ops::BroadcastSubSlow(
        op_params, RuntimeShape(input_0_dims_size, input_0_dims_raw), input_v_0,
        RuntimeShape(input_1_dims_size, input_1_dims_raw), input_v_1,
        RuntimeShape(output_dims_size, output_dims_raw), output_data);
  } else {
    reference_ops::SubWithActivation(
        op_params, RuntimeShape(input_0_dims_size, input_0_dims_raw), input_v_0,
        RuntimeShape(input_1_dims_size, input_1_dims_raw), input_v_1,
        RuntimeShape(output_dims_size, output_dims_raw), output_data);
  }
  return output_data;
}
}  // namespace randomname