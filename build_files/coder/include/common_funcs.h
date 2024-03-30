# ifndef COMMON_FUNCS_H
# define COMMON_FUNCS_H
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
// #include "tensorflow/lite/kernels/eigen_support.cc"
using namespace tflite;

void ExtractConvParams(TfLitePadding padding, int stride_width, int stride_height, 
                               int dilation_width_factor, int dilation_height_factor,
                               TfLiteFusedActivation activation,
                               TfLiteConvParams* data_params) {
  // TfLiteConvParams data_params;
  data_params->padding = padding;
  data_params->stride_width = stride_width;
  data_params->stride_height = stride_height;
  data_params->dilation_width_factor = dilation_width_factor;
  data_params->dilation_height_factor = dilation_height_factor;
  data_params->activation = activation;
  // return data_params;
}

void ExtractDepthConvParams(TfLitePadding padding, int stride_width, int stride_height,
                               int dilation_width_factor, int dilation_height_factor,
                               TfLiteFusedActivation activation,
                               TfLiteDepthwiseConvParams* data_params) {
  // TfLiteDepthwiseConvParams data_params;
  data_params->padding = padding;
  data_params->stride_width = stride_width;
  data_params->stride_height = stride_height;
  data_params->dilation_width_factor = dilation_width_factor;
  data_params->dilation_height_factor = dilation_height_factor;
  data_params->activation = activation;
  // return data_params;
}

void GetTensor(TfLiteType type, char* name, TfLiteIntArray* tensor_dims_data, 
                       TfLiteQuantizationParams quant_params,
                       char* tensor_data, TfLiteAffineQuantization* quant_struct,
                       size_t bytes_size, TfLiteTensor* tensor) {
  tensor->type = type;
  tensor->name = name;
  tensor->dims = tensor_dims_data;
  tensor->params = quant_params;
  // tensor->data.raw = reinterpret_cast<char*>(tensor_data);
  if (strcmp(name, "output") == 0){
    tensor->data.raw = (char*)malloc(bytes_size);
  }
  else{
    tensor->data.raw = tensor_data;
  }
  // tensor->data.raw = tensor_data;
  tensor->bytes = bytes_size;
  tensor->allocation_type = kTfLiteMemNone;
  // data_0.allocation = allocation;
//   if (name != "output"){
//     tensor->is_variable = false;
//   }
//   else{
//     tensor->is_variable = true;
//   }
  tensor->is_variable = false;
  if (type != kTfLiteFloat32) {
    tensor->quantization.type = kTfLiteAffineQuantization;
    tensor->quantization.params = quant_struct;
  } else {
    tensor->quantization.type = kTfLiteNoQuantization;
    tensor->quantization.params = nullptr;
  }
  tensor->sparsity = nullptr;
}

void ResetTensor(TfLiteTensor* tensor, TfLiteTensor* src_tensor) {
//   TfLiteTensorFree(tensor);
  tensor->type = src_tensor->type;
  tensor->name = src_tensor->name;
  tensor->dims = src_tensor->dims;
  tensor->params = src_tensor->params;
  tensor->data.raw = reinterpret_cast<char*>(GetTensorData<float>(src_tensor));
  tensor->bytes = src_tensor->bytes;
  tensor->allocation_type = src_tensor->allocation_type;
  tensor->allocation = src_tensor->allocation;
  tensor->is_variable = src_tensor->is_variable;

  if (src_tensor->type != kTfLiteFloat32) {
    tensor->quantization.type = src_tensor->quantization.type;
    tensor->quantization.params = src_tensor->quantization.params;
  } else {
    tensor->quantization.type = kTfLiteNoQuantization;
    tensor->quantization.params = nullptr;
  }
}
// tflite::eigen_support::LazyEigenThreadPoolHolder* getThreadPoolHolder(const int num_threads){
//     return new tflite::eigen_support::LazyEigenThreadPoolHolder(num_threads);
// }

# endif