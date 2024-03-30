#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
using namespace tflite;

namespace randomname {

const int num_splits=;
int32_t axis=;

const int input_dims_size=;
const int32_t input_dims_raw=;


const int output_0_dims_size=;
const int32_t output_0_dims_raw=;
const int32_t output_0_num=;

const int output_1_dims_size=;
const int32_t output_1_dims_raw=;
const int32_t output_1_num=;

const int output_2_dims_size=;
const int32_t output_2_dims_raw=;
const int32_t output_2_num=;

// template <typename T>
// class VectorOfOutputs {
//  public:
//   // Build with the tensors in 'tensor_list'.
//   VectorOfOutputs(const TfLiteContext& context,
//                   const TfLiteIntArray& tensor_list) {
//     int num_tensors = tensor_list.size;

//     all_data_.reserve(num_tensors);
//     all_shape_.reserve(num_tensors);
//     all_shape_ptr_.reserve(num_tensors);

//     for (int i = 0; i < num_tensors; ++i) {
//       TfLiteTensor* t = &context.tensors[tensor_list.data[i]];
//       all_data_.push_back(GetTensorData<T>(t));
//       all_shape_.push_back(GetTensorShape(t));
//     }

//     // Taking the pointer from inside a std::vector is only OK if the vector is
//     // never modified, so we populate all_shape in the previous loop and then we
//     // are free to grab iterators here.
//     for (int i = 0; i < num_tensors; ++i) {
//       all_shape_ptr_.push_back(&all_shape_[i]);
//     }
//   }
//   // Return a pointer to the data pointers of all tensors in the list. For
//   // example:
//   //   float* const* f = v.data();
//   //   f[0][1] is the second element of the first tensor.
//   T* const* data() const { return all_data_.data(); }

//   // Return a pointer the shape pointers of all tensors in the list. For
//   // example:
//   //   const RuntimeShape* const* d = v.dims();
//   //   dims[1] are the dimensions of the second tensor in the list.
//   const RuntimeShape* const* shapes() const { return all_shape_ptr_.data(); }

//  private:
//   std::vector<T*> all_data_;
//   std::vector<RuntimeShape> all_shape_;
//   std::vector<RuntimeShape*> all_shape_ptr_;
// };


// auto* randomname(float* input_v_0, float* input_v_1) {
std::tuple<type_placeholder> randomname(float* input_v) {

  // float* output_data = new float[output_num];
  float* output_0_data = (float*)malloc(sizeof(float) * output_0_num);
  float* output_1_data = (float*)malloc(sizeof(float) * output_1_num);
  float* output_2_data = (float*)malloc(sizeof(float) * output_2_num);

  std::vector<float*> all_data_;
  std::vector<RuntimeShape> all_shape_;
  std::vector<RuntimeShape*> all_shape_ptr_;

  all_data_.reserve(num_splits);
  all_shape_.reserve(num_splits);
  all_shape_ptr_.reserve(num_splits);

  all_data_output_v_0;
  all_data_output_v_1;
  all_data_output_v_2;
  RuntimeShape_output_0;
  RuntimeShape_output_1;
  RuntimeShape_output_2;

  // for (int i = 0; i < num_splits; ++i) {
  //   all_data_.push_back(GetTensorData<T>(t));
  //   all_shape_.push_back(GetTensorShape(t));
  // }

  // Taking the pointer from inside a std::vector is only OK if the vector is
  // never modified, so we populate all_shape in the previous loop and then we
  // are free to grab iterators here.
  for (int i = 0; i < num_splits; ++i) {
    all_shape_ptr_.push_back(&all_shape_[i]);
  }

  int axis_value = axis[0];
  if (axis_value < 0) {
    axis_value += input_dims_size;
  }

  tflite::SplitParams op_params;                                    \
  op_params.num_split = num_splits;                           \
  op_params.axis = axis_value;                                      \
  reference_ops::Split(op_params, RuntimeShape(input_dims_size, input_dims_raw), \
                       input_v,     \
                       all_shape_ptr_.data(), all_data_.data());

  return {outputs_placeholder};
}
}  // namespace randomname