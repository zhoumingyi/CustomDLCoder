# ctypes_test.py
import time
import ctypes
from ctypes import *
# import onnx
import tensorflow as tf
import numpy as np
from numpy.ctypeslib import ndpointer
import gc
from utils.utils import *
import argparse
size_batch = 1
input_dtype = "float32"
# from tvm.contrib import graph_executor as runtime

def eval(model_path, not_gpt):

    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    # print(input_details)
    input_tensor = input_details[0]["name"]
    input_shape = input_details[0]["shape"]
    input_shape[0] = size_batch

    num_input = 1
    for i in range(len(input_shape)):
        num_input *= input_shape[i]

    if not_gpt:
        inputs = np.random.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
    else:
        print("this is the gpt2 model")
        inputs = np.random.randint(0, 255, size=tuple(input_shape)).astype(np.int32)
    input_shape = tuple(input_shape)

    output_details = interpreter.get_output_details()
    # print(output_details)
    output_tensor = output_details[0]["name"]
    output_shape = output_details[0]["shape"]
    output_shape[0] = size_batch

    num_output = 1
    for i in range(len(output_shape)):
        num_output *= output_shape[i]

    output_shape = tuple(output_shape)
    del interpreter
    gc.collect()
    output_ori = model_inference(model_path, inputs)

    C_inputs = inputs
    if not C_inputs.flags['C_CONTIGUOUS']:
        C_inputs = np.ascontiguousarray(C_inputs, dtype=C_inputs.dtype)
    inputs_ctypes_ptr = cast(C_inputs.ctypes.data, POINTER(c_float))

    time_start=time.time()
    out_tflite = tflite_inference(model_path, "minimal_x86_build/libminimal.so", output_shape, inputs_ctypes_ptr, num_input, num_output)
    time_end=time.time()
    print("tflite time: ", time_end-time_start)
    # print("tflite_cmake error: ", np.max(np.abs(out_tflite - output_ori)))
    time_start=time.time()
    out_coder = CustomDLCoder_inference("coder_x86_build/libcoder.so", output_shape, inputs_ctypes_ptr)
    # print("coder error: ", np.max(np.abs(out_coder - out_tflite_c)))
    time_end=time.time()
    print("coder error: ", np.max(np.abs(out_coder - out_tflite)))
    print("coder time: ", time_end-time_start)
    return np.max(np.abs(out_coder - out_tflite))

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name', type=str, default='lenet', help='name of the model')
#     parser.add_argument('--gpt2', type=bool, help='only for gpt2 model')
#     parser.add_argument('--latency', type=bool, help='test the latency (inference time)')
#     opt = parser.parse_args()

#     model_path = './tflite_model/' + opt.model_name + '.tflite'

#     interpreter = tf.lite.Interpreter(model_path)
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     # print(input_details)
#     input_tensor = input_details[0]["name"]
#     input_shape = input_details[0]["shape"]
#     input_shape[0] = size_batch

#     num_input = 1
#     for i in range(len(input_shape)):
#         num_input *= input_shape[i]

#     if not opt.gpt2:
#         inputs = np.random.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
#     else:
#         inputs = np.random.randint(0, 255, size=tuple(input_shape)).astype(np.int32)
#     input_shape = tuple(input_shape)

#     output_details = interpreter.get_output_details()
#     # print(output_details)
#     output_tensor = output_details[0]["name"]
#     output_shape = output_details[0]["shape"]
#     output_shape[0] = size_batch

#     num_output = 1
#     for i in range(len(output_shape)):
#         num_output *= output_shape[i]

#     output_shape = tuple(output_shape)
#     # output_shape = (1, 112, 112, 16)
#     # for t in interpreter.get_tensor_details():
#     #     # print(t)
#     #     if t['name'] == 'MobilenetV1/MobilenetV1/Conv2d_0/Relu6':
#     #         out_index = t['index']
#     # del interpreter
#     # gc.collect()
#     output_ori = model_inference(model_path, inputs)

#     C_inputs = inputs
#     if not C_inputs.flags['C_CONTIGUOUS']:
#         C_inputs = np.ascontiguousarray(C_inputs, dtype=C_inputs.dtype)
#     inputs_ctypes_ptr = cast(C_inputs.ctypes.data, POINTER(c_float))
#     time_start=time.time()
#     if opt.latency:
#         for i in range(100):
#             out_tflite = tflite_inference(model_path, "minimal_x86_build/libminimal.so", output_shape, inputs_ctypes_ptr, num_input, num_output)
#     else:
#         out_tflite = tflite_inference(model_path, "minimal_x86_build/libminimal.so", output_shape, inputs_ctypes_ptr, num_input, num_output)
#     time_end=time.time()
#     print("cmake time: ", time_end-time_start)
#     # print("cmake error: ", np.max(np.abs(out_tflite - output_ori)))
#     time.sleep(3)
#     time_start=time.time()
#     if opt.latency:
#         for i in range(100):
#             out_coder = CustomDLCoder_inference("coder_x86_build/libcoder.so", output_shape, inputs_ctypes_ptr)
#     else:
#         out_coder = CustomDLCoder_inference("coder_x86_build/libcoder.so", output_shape, inputs_ctypes_ptr)
#     time_end=time.time()
#     print("coder error: ", np.max(np.abs(out_coder - out_tflite)))
#     print("coder time: ", time_end-time_start)