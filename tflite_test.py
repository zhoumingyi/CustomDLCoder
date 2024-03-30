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
from utils.utils import model_inference
# import tvm.relay as relay

# input_tensor = "conv2d_1_input"
size_batch = 1
input_dtype = "int32"

# Parse TFLite model and convert it to a Relay module
from tvm import relay, transform
import tvm
from tvm import te
from tvm.contrib import graph_executor as runtime


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
    model_path = "./tflite_model/skin.tflite"

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

    inputs = np.random.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
    # inputs = np.random.randint(0, 255, size=tuple(input_shape)).astype(np.int32)
    input_shape = tuple(input_shape)

    output_details = interpreter.get_output_details()
    # print(output_details)
    output_tensor = output_details[0]["name"]
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

    tflite_model_buf = open(model_path, "rb").read()
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
    )

    # Build the module against to x86 CPU
    target = "llvm"
    with transform.PassContext():
        lib = relay.build(mod, target, params=params)
    # ----------------------------------------------------
    # this is the result of TVM. We use it as the baseline to compare with our methods.
    # ----------------------------------------------------
    tvm_output, tvm_time = TVM_inference(lib, input_tensor, inputs)

    print("tvm time cost: ", (tvm_time), "s")
    # print("tvm loss: ", np.linalg.norm(tvm_output - output_ori)/np.mean(np.abs(output_ori)))
    print("tvm loss: ", np.max(np.abs(tvm_output - output_ori)))
    del lib, tflite_model_buf, tflite_model, mod, params, target


