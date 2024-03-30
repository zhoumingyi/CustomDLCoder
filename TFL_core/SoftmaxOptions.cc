#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

using namespace tflite;
// using namespace optimized_ops;
namespace randomname {

const float beta=;

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



auto* randomname(float* input_v, CpuBackendContext* cpu_backend_context) {

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);

  tflite::SoftmaxParams op_params;        
  op_params.beta = beta;
  // auto* cpu_backend_context = new CpuBackendContext();
  // cpu_backend_context->SetMaxNumThreads(-1);   
  optimized_ops::Softmax(op_params, RuntimeShape(input_dims_size, input_dims_raw),
                          input_v, RuntimeShape(output_dims_size, output_dims_raw), output_data,
                          cpu_backend_context);

  return output_data;
}
}  // namespace randomname