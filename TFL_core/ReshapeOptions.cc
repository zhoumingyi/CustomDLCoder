#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
using namespace tflite;
namespace randomname {

std::vector<int32_t> shape=;
int32_t shape_size=;

const int input_dims_size=;
const int32_t input_dims_raw=;


const int output_dims_size=;
const int32_t output_dims_raw=;
const int32_t output_num=;
const float scale_output=;
const int32_t zero_point_output=;
const TfLiteType output_type=;



auto* randomname(float* input_v) {

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);

  // auto* cpu_backend_context = new CpuBackendContext();
  // cpu_backend_context->SetMaxNumThreads(-1);   
  memcpy(output_data, input_v, sizeof(float) * output_num);

  return output_data;
}
}  // namespace randomname