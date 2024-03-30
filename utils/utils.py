import os
import random
import numpy
import flatbuffers
import torch
import ctypes
from ctypes import *
from onnx2pytorch.convert.layer import *
from PIL import Image
from numpy import asarray
from numpy.ctypeslib import ndpointer
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb
from tvm.contrib import graph_executor as runtime
import tvm

def OnnxWeights2Torch(params):
    return torch.from_numpy(numpy_helper.to_array(params))

def OnnxWeights2Numpy(params):
    return numpy_helper.to_array(params)

def Torch2OnnxWeights(params):
    return numpy_helper.from_array(params.numpy())

def OutputsOffset(subgraph, j):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
    if o != 0:
        a = subgraph._tab.Vector(o)
        return a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4)
    return 0

def buffer_change_output_tensor_to(model_buffer, new_tensor_i):

    root = schema_fb.Model.GetRootAsModel(model_buffer, 0)
    output_tensor_index_offset = OutputsOffset(root.Subgraphs(0), 0)

    # Flatbuffer scalars are stored in little-endian.
    new_tensor_i_bytes = bytes([
    new_tensor_i & 0x000000FF, \
    (new_tensor_i & 0x0000FF00) >> 8, \
    (new_tensor_i & 0x00FF0000) >> 16, \
    (new_tensor_i & 0xFF000000) >> 24 \
    ])
    # Replace the 4 bytes corresponding to the first output tensor index
    return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[output_tensor_index_offset + 4:]

def getJpg(filename: str):
	return filename.endswith("jpg")

def generate_random_data(model_path):
    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    model = tf.lite.Interpreter(model_content=model_buffer)
    input_details = model.get_input_details()
    input_tensors = []
    # print(tuple(input_details[0]['shape'].astype(np.int32).tolist()))
    # if len(input_details) == 1:
    #     if input_details[0]['shape'].astype(np.int32).tolist()[3] == 3:
    #         image_path = '/fs03/rm46/dataset/plant_disease/Background_without_leaves/'
    #         filelist = os.listdir(image_path)
    #         jpgList = [i for i in filter(getJpg, filelist)]
    #         sample = random.sample(jpgList, 1)
    #         image = Image.open(image_path + sample[0])
    #         # torch.permute(inputs, (0,2,3,1))
    #         input = torch.permute(torch.from_numpy(np.expand_dims(asarray(image), axis=0)), (0,3,1,2))
    #         input = torchvision.transforms.Resize(tuple(input_details[0]['shape'].astype(np.int32).tolist()[1:3]))(input)
    #         input = torch.permute(input, (0,2,3,1))
    #         input_tensors.append(input.numpy())
    #         # print(input.size())
    # else:
    for i in range(len(input_details)):
        # print(input_details[i]['dtype'])
        shape_input = tuple(input_details[i]['shape'].astype(np.int32).tolist())
        # print(shape_input)
        if input_details[i]['dtype'] == numpy.uint8:
            inputs = torch.randint(low=0, high=255, size=shape_input).to(torch.uint8).numpy()
        elif input_details[i]['dtype'] == numpy.float32:
            inputs = torch.randn(shape_input).numpy()
        input_tensors.append(inputs)
    return input_tensors

def model_inference(model_path, inputs, out_index=None):
    interpreter = tf.lite.Interpreter(
        model_path, experimental_preserve_all_tensors=True
        )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    for i in range(len(inputs)):
        interpreter.set_tensor(input_details[0]["index"], np.expand_dims(inputs[i], 0))
        interpreter.invoke()
        if i == 0:
            if out_index is None:
                output = interpreter.get_tensor(output_details[0]['index'])
            else:
                output = interpreter.get_tensor(out_index)
        else:
            output = np.concatenate((output, interpreter.get_tensor(output_details[0]['index'])), axis=0)
    return output

def tflite_inference(model_path, lib_path, output_shape, inputs_ctypes_ptr, num_input, num_output):
    # c_lib = ctypes.cdll.LoadLibrary("minimal_build/libminimal.so")
    c_lib = ctypes.cdll.LoadLibrary(lib_path)
    # c_lib.tflite_minimal.restype = ctypes.POINTER(ctypes.c_float)
    c_lib.tflite_minimal.restype = ndpointer(dtype=c_float, shape=output_shape)
    # c_lib.tflite_minimal.restype = POINTER(c_float)
    c_lib.tflite_minimal.argtypes = [c_char_p, POINTER(c_float), c_int, c_int]
    out_tflite_c = c_lib.tflite_minimal(model_path.encode(), inputs_ctypes_ptr, num_input, num_output)
    return out_tflite_c

def CustomDLCoder_inference(lib_path, output_shape, inputs_ctypes_ptr):
    # coder_lib = ctypes.cdll.LoadLibrary("coder_build/libcoder.so")
    coder_lib = ctypes.cdll.LoadLibrary(lib_path)
    coder_lib.coder.restype = ndpointer(dtype=c_float, shape=output_shape)
    out_tflite_c = coder_lib.coder(inputs_ctypes_ptr)
    return out_tflite_c

def TVM_inference(lib, input_tensor, inputs):
    module = runtime.GraphModule(lib["default"](tvm.cpu(0)))
    module.set_input(input_tensor, tvm.nd.array(inputs))

    # Run
    module.run()

    # Get output
    tvm_output = module.get_output(0).numpy()
    return tvm_output

def TfliteToOnnx(path, model_name):
    if model_name == None:
        filelist = os.listdir(path)
        for i in range(len(filelist)):
            if os.path.splitext(filelist[i])[1] != ('.tflite') and os.path.splitext(filelist[i])[1] != ('.lite'):
                os.system("rm " + path + filelist[i])
            else:
                os.system("python -m tf2onnx.convert --opset 13 --tflite " +  path + filelist[i] +
                " --output " + "out_model/" + os.path.splitext(filelist[i])[0] + ".onnx")
    else:
        os.system("python -m tf2onnx.convert --opset 13 --tflite " +  path + model_name +
        " --output " + "out_model/" + os.path.splitext(model_name)[0] + ".onnx")


def conv_activation_parser(activation_name):
    if activation_name == 'RELU':
        return 'kTfLiteActRelu'
    elif activation_name == 'RELU6':
        return 'kTfLiteActRelu6'
    elif activation_name == 'TANH':
        return 'kTfLiteActTanh'
    elif activation_name == 'RELU_N1_TO_1':
        return 'kTfLiteActReluN1To1'
    elif activation_name == 'NONE':
        return 'kTfLiteActNone'
    else:
        raise TypeError('Activation type ' + activation_name + ' not supported by conv layers')

def conv_padding_parser(padding_name):
    if padding_name == 'SAME':
        return 'kTfLitePaddingSame'
    elif padding_name == 'VALID':
        return 'kTfLitePaddingValid'
    else:
        raise TypeError('padding type' + padding_name + ' not supported by conv layers')

def conv_data_type_parser(data_type):
    if data_type == 'float32' or data_type == np.float32:
        return 'kTfLiteFloat32', 'float'
    elif data_type == 'int8' or data_type == np.int8:
        return 'kTfLiteInt8', 'int8_t'
    elif data_type == 'uint8' or data_type == np.uint8:
        return 'kTfLiteUInt8', 'uint8_t'
    elif data_type == 'int16' or data_type == np.int16:
        return 'kTfLiteInt16', 'int16_t'
    elif data_type == 'int32' or data_type == np.int32:
        return 'kTfLiteInt32', 'int32_t'
    else:
        raise TypeError('filter type ' + data_type + ' not supported by conv layers')

def weights_format_parser(weights_format):
    if weights_format == 'DEFAULT':
        return 'kTfLiteFullyConnectedWeightsFormatDefault'
    else:
        return 'kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8'
