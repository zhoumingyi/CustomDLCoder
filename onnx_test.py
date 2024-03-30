# ctypes_test.py
import ctypes
from ctypes import *
# import onnx
import tensorflow as tf
import numpy as np
from numpy.ctypeslib import ndpointer
import time
from PIL import Image
import gc
from utils.utils import model_inference, TfliteToOnnx
# import tvm.relay as relay

# input_tensor = "conv2d_1_input"
size_batch = 1
input_dtype = "int32"

import onnxruntime as ort


def TVM_inference(lib, input_tensor, inputs):
    module = runtime.GraphModule(lib["default"](tvm.cpu(0)))
    time_start=time.time()
    module.set_input(input_tensor, tvm.nd.array(inputs))

    # Run
    module.run()

    # Get output
    tvm_output = module.get_output(0).numpy()
    time_end=time.time()
    return tvm_output, time_end-time_start

def tflite_inference(path, output_shape, inputs_ctypes_ptr, num_input, num_output):
    time_start=time.time()
    c_lib = ctypes.cdll.LoadLibrary("minimal_x86_build/libminimal.so")
    # c_lib.forward.restype = ctypes.POINTER(ctypes.c_float)
    c_lib.tflite_minimal.restype = ndpointer(dtype=c_float, shape=output_shape)
    c_lib.tflite_minimal.argtypes = [c_char_p, POINTER(c_float), c_int, c_int]
    out_tflite_c = c_lib.tflite_minimal(path.encode(), inputs_ctypes_ptr, num_input, num_output)
    time_end=time.time()
    return out_tflite_c, time_end-time_start

def CustomDLCoder_inference(output_shape, inputs_ctypes_ptr):
    time.sleep(1)
    coder_lib = ctypes.cdll.LoadLibrary("coder_x86_build/libcoder.so")
    time_start=time.time()
    coder_lib.coder.restype = ndpointer(dtype=c_float, shape=output_shape)
    out_tflite_c = coder_lib.coder(inputs_ctypes_ptr)
    time_end=time.time()
    return out_tflite_c, time_end-time_start

if __name__ == "__main__":
    # TfliteToOnnx("tflite_model/", model_name=None)
    model_path = "./tflite_model/gpt2.tflite"

    interpreter = tf.lite.Interpreter(model_path)
    # for i in range(10):
    interpreter.allocate_tensors()
    for t in interpreter.get_tensor_details():
        # print(t)
        if t['name'] == ' input_1':
            out_index = t['index']

    input_details = interpreter.get_input_details()
    # print(input_details)
    input_tensor = input_details[0]["name"]
    input_shape = input_details[0]["shape"]
    input_shape[0] = size_batch
    num_input = 1
    for i in range(len(input_shape)):
        num_input *= input_shape[i]

    # inputs = np.random.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
    inputs = np.random.randint(0, 255, size=tuple(input_shape)).astype(np.int32)
    input_shape = tuple(input_shape)

    output_details = interpreter.get_output_details()
    # print(output_details)
    output_tensor = output_details[0]["name"]
    print(output_details[0]["name"])
    output_shape = output_details[0]["shape"]
    # output_shape = [1,64,768]
    output_shape[0] = size_batch

    num_output = 1
    for i in range(len(output_shape)):
        num_output *= output_shape[i]

    output_shape = tuple(output_shape)
    del interpreter
    gc.collect()

    output_ori = model_inference(model_path, inputs)


    time_start=time.time()
    for i in range(10):
        options = ort.SessionOptions()
        providers = ['CPUExecutionProvider']
        options.enable_profiling=True
        ort_sess = ort.InferenceSession('out_model/gpt2.onnx', sess_options=options, providers=providers)
        input_names = [input.name for input in ort_sess.get_inputs()]
        input_shapes = {}
        for input in ort_sess.get_inputs():
            input_name = input.name
            input_shape = input.shape
            input_shapes[input_name] = input_shape
        print(tuple(input_shapes[input_name]))
        output_names = [input.name for input in ort_sess.get_outputs()]
        # print(ort_sess.get_outputs())
        outputs = np.array(ort_sess.run([output_names[0]], {input_names[0]: inputs}))
    time_end=time.time()
    print("onnx time cost: ", (time_end - time_start), "s")
    print("onnx loss: ", np.max(np.abs(outputs - output_ori)))



