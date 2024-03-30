// #include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
// #include "tensorflow/lite/kernels/internal/quantization_util.h"
// #include "tensorflow/lite/kernels/internal/reference/binary_function.h"
// #include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
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

template <typename T>
T SquaredDifference(T input1, T input2) {
  const T difference = input1 - input2;
  return difference * difference;
}

template <typename T>
void EvalSquaredDifference(T* input_v_0, T* input_v_1, T* output_data, bool requires_broadcast) {
  if (requires_broadcast) {
    reference_ops::BroadcastBinaryFunction4DSlow<T, T, T>(
        RuntimeShape(input_0_dims_size, input_0_dims_raw), input_v_0,
        RuntimeShape(input_1_dims_size, input_1_dims_raw), input_v_1,
        RuntimeShape(output_dims_size, output_dims_raw), output_data, SquaredDifference<T>);
  } else {
    reference_ops::BinaryFunction<T, T, T>(
        RuntimeShape(input_0_dims_size, input_0_dims_raw), input_v_0,
        RuntimeShape(input_1_dims_size, input_1_dims_raw), input_v_1,
        RuntimeShape(output_dims_size, output_dims_raw), output_data, SquaredDifference<T>);
  }
}

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

  if (output_type == kTfLiteFloat32) {
    EvalSquaredDifference<float>(input_v_0, input_v_1, output_data, requires_broadcast);
  }
//   else if (output_type == kTfLiteInt32) {
//     EvalSquaredDifference<int32_t>(output_data, requires_broadcast);
//   }
// //   else if (output->type == kTfLiteInt8) {
// //     EvalQuantizedSquaredDifference<int8_t>(output_data);
// //   }
//   else {
//     context->ReportError(
//         context,
//         "SquaredDifference only supports FLOAT32 and INT32 now, got %d.",
//         output->type);
//     return kTfLiteError;
//   }
  return output_data;
}
}