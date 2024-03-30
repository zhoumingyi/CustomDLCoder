# ctypes_test.py
import ctypes
from ctypes import *
# import onnx
# import tensorflow as tf
import numpy as np
from numpy.ctypeslib import ndpointer
import gc
# import time
# from memory_profiler import profile
# from memory_profiler import memory_usage
# from utils.utils import *
# size_batch = 1
# input_dtype = "float32"
# import psutil


# from tvm.contrib import graph_executor as runtime

# @profile
def tflite_inference_ram(path, output_shape, inputs_ctypes_ptr, num_input, num_output):
    c_lib = ctypes.cdll.LoadLibrary("minimal_x86_build/libminimal.so")
    # c_lib.forward.restype = ctypes.POINTER(ctypes.c_float)
    c_lib.tflite_minimal.restype = ndpointer(dtype=c_float, shape=output_shape)
    c_lib.tflite_minimal.argtypes = [c_char_p, POINTER(c_float), c_int, c_int]
    out_tflite_c = c_lib.tflite_minimal(path.encode(), inputs_ctypes_ptr, num_input, num_output)
    # time.sleep(1)
    return out_tflite_c

# @profile
def CustomDLCoder_inference_ram(output_shape, inputs_ctypes_ptr):
    coder_lib = ctypes.cdll.LoadLibrary("coder_x86_build/libcoder.so")
    coder_lib.coder.restype = ndpointer(dtype=c_float, shape=output_shape)
    out_tflite_c = coder_lib.coder(inputs_ctypes_ptr)
    return out_tflite_c


if __name__ == "__main__":
    model_path = "./tflite_model/ssd.tflite"

    # interpreter = tf.lite.Interpreter(model_path)
    # interpreter.allocate_tensors()

    # input_details = interpreter.get_input_details()
    # # print(input_details)
    # input_tensor = input_details[0]["name"]
    # input_shape = input_details[0]["shape"]
    # input_shape[0] = size_batch

    # num_input = 1
    # for i in range(len(input_shape)):
    #     num_input *= input_shape[i]
    input_shape = [1, 320, 320, 3]
    inputs = np.random.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
    output_shape = [1,2034,1,4]
    # inputs = np.random.randint(0, 255, size=tuple(input_shape)).astype(np.int32)
    # print(inputs)
    # input_shape = tuple(input_shape)

    # output_details = interpreter.get_output_details()
    # # print(output_details)
    # output_tensor = output_details[0]["name"]
    # output_shape = output_details[0]["shape"]
    # output_shape[0] = size_batch

    # num_output = 1
    # for i in range(len(output_shape)):
    #     num_output *= output_shape[i]

    # output_shape = tuple(output_shape)
    # for t in interpreter.get_tensor_details():
    #     # print(t)
    #     if t['name'] == 'Identity':
    #         out_index = t['index']
    #         output_shape = tuple(t['shape'])
    # print("output shape: ", output_shape)
    # del interpreter
    # gc.collect()
    # output_ori = model_inference(model_path, inputs)
    # print("tflite lib shape: ", output_ori.shape)

    C_inputs = inputs
    if not C_inputs.flags['C_CONTIGUOUS']:
        C_inputs = np.ascontiguousarray(C_inputs, dtype=C_inputs.dtype)
    inputs_ctypes_ptr = cast(C_inputs.ctypes.data, POINTER(c_float))

    num_input = 1
    for i in range(len(input_shape)):
        num_input *= input_shape[i]
    num_output = 1
    for i in range(len(output_shape)):
        num_output *= output_shape[i]
    out_tflite = tflite_inference_ram(model_path, output_shape, inputs_ctypes_ptr, num_input, num_output)

    # if np.max(np.abs(out_tflite - output_ori)) == 0:
    #     print("validation pass")
    # else:
    #     print("validation failed")
    # time_start=time.time()
    # for i in range(100):
    #     out_tflite = tflite_inference(model_path, "minimal_x86_build/libminimal.so", output_shape, inputs_ctypes_ptr, num_input, num_output)
    # time_end=time.time()
    # print("tflite time cost: ", (time_end-time_start)*10, "ms")

    # out_coder = CustomDLCoder_inference_ram(output_shape, inputs_ctypes_ptr)
    # print("coder error: ", np.max(np.abs(out_coder - out_tflite)))

    # time_start=time.time()
    # for i in range(100):
    #     out_coder = CustomDLCoder_inference("coder_x86_build/libcoder.so", output_shape, inputs_ctypes_ptr)
    # time_end=time.time()
    # print("coder time cost: ", (time_end-time_start)*10, "ms")
