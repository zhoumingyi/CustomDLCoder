#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
using namespace tflite;
namespace randomname {


int values_count=;
int axis=;

//support maximal 10 input tensors
const int input_0_dims_size=;
const int32_t input_0_dims_raw=;

const int input_1_dims_size=;
const int32_t input_1_dims_raw=;

const int input_2_dims_size=;
const int32_t input_2_dims_raw=;

const int input_3_dims_size=;
const int32_t input_3_dims_raw=;

const int input_4_dims_size=;
const int32_t input_4_dims_raw=;

const int input_5_dims_size=;
const int32_t input_5_dims_raw=;

const int input_6_dims_size=;
const int32_t input_6_dims_raw=;

const int input_7_dims_size=;
const int32_t input_7_dims_raw=;

const int input_8_dims_size=;
const int32_t input_8_dims_raw=;

const int input_9_dims_size=;
const int32_t input_9_dims_raw=;

const int input_10_dims_size=;
const int32_t input_10_dims_raw=;

const int input_11_dims_size=;
const int32_t input_11_dims_raw=;


const int tensor_num=;


const int output_dims_size=;
const int32_t output_dims_raw=;
const int32_t output_num=;
// const float scale_output=;
// const int32_t zero_point_output=;
const TfLiteType output_type=;

auto* randomname(list_input_v) {

  // float* output_data = new float[output_num];
  float* output_data = (float*)malloc(sizeof(float) * output_num);

  std::vector<float*> all_data_;
  std::vector<RuntimeShape> all_shape_;
  std::vector<RuntimeShape*> all_shape_ptr_;

  all_data_.reserve(tensor_num);
  all_shape_.reserve(tensor_num);
  all_shape_ptr_.reserve(tensor_num);

  all_data_input_v_0;
  all_data_input_v_1;
  all_data_input_v_2;
  all_data_input_v_3;
  all_data_input_v_4;
  all_data_input_v_5;
  all_data_input_v_6;
  all_data_input_v_7;
  all_data_input_v_8;
  all_data_input_v_9;
  all_data_input_v_10;
  all_data_input_v_11;

  RuntimeShape_input_0;
  RuntimeShape_input_1;
  RuntimeShape_input_2;
  RuntimeShape_input_3;
  RuntimeShape_input_4;
  RuntimeShape_input_5;
  RuntimeShape_input_6;
  RuntimeShape_input_7;
  RuntimeShape_input_8;
  RuntimeShape_input_9;
  RuntimeShape_input_10;
  RuntimeShape_input_11;

  // Taking the pointer from inside a std::vector is only OK if the vector is
  // never modified, so we populate all_shape in the previous loop and then we
  // are free to grab iterators here.
  for (int i = 0; i < tensor_num; ++i) {
    all_shape_ptr_.push_back(&all_shape_[i]);
  }

  tflite::PackParams op_params;
  op_params.axis = axis;
  op_params.inputs_count = values_count;

  reference_ops::Pack<float>(op_params, all_shape_ptr_.data(), all_data_.data(),
                         RuntimeShape(output_dims_size, output_dims_raw), output_data);

  return output_data;
}
}
