import numpy as np
from utils.utils import *

def extract_gather(op, kwargs, interpreter, unknown_config):
    # print(op['inputs'][2])
    # load the stride
    try:
        axis = op['builtin_options']['axis']
    except:
        kwargs['axis='] = 'axis=0'
        axis = 0
    else:
        kwargs['axis='] = 'axis=' + str(axis)
    try:
        batch_dims = op['builtin_options']['batch_dims']
    except:
        kwargs['batch_dims='] = 'batch_dims=0'
        batch_dims = 0
    else:
        kwargs['batch_dims='] = 'batch_dims=' + str(batch_dims)

    for tensor_details in interpreter.get_tensor_details():
        if tensor_details['index'] == op['inputs'][0]:
            input_tensor = interpreter.get_tensor(tensor_details["index"])
            input_item_num = input_tensor.size
            input_input_channel = input_tensor.shape[1]
            input_output_channel = input_tensor.shape[0]
            input_dims_raw = '{' + str(input_output_channel) + ',' + str(input_input_channel) +'}'
            input_dims_size = len(input_tensor.shape)
            tflite_type, type_str = conv_data_type_parser(input_tensor.dtype)
            kwargs['input_dims_size='] = 'input_dims_size=' + str(input_dims_size)
            kwargs['input_dims_raw='] = 'input_dims_raw[' + str(input_dims_size) + ']=' + input_dims_raw
            kwargs['input_raw='] = type_str + ' input_raw[' + str(input_item_num) + ']=' + '{' + str(input_tensor.flatten('C').tolist()).strip('[').strip(']') + '}'
            kwargs['input_tensor_data=input_raw'] = type_str + '* input_tensor_data=input_raw'

        elif tensor_details['index'] == op['outputs'][0]:
            # output_tensor = interpreter.get_tensor(tensor_details["index"])
            output_channel = tensor_details['shape'][2]
            output_f = tensor_details['shape'][1]
            # output_width = tensor_details['shape'][2]
            output_num = output_channel*output_f
            output_dims_raw = '{' + '1,' + str(output_f) + ',' + str(output_channel) + '}'
            output_dims_size = len(tensor_details['shape'])
            tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
            quantization_output = tensor_details['quantization']
            kwargs['output_dims_size='] = 'output_dims_size=' + str(output_dims_size)
            kwargs['output_dims_raw='] = 'output_dims_raw[' + str(output_dims_size) + ']=' + output_dims_raw
            kwargs['output_num='] = 'output_num=' + str(output_num)
            kwargs['output_type='] = 'output_type=' + tflite_type
            kwargs['scale_output='] = 'scale_output=' + str(quantization_output[0])
            kwargs['zero_point_output='] = 'zero_point_output=' + str(quantization_output[1])
        elif tensor_details['index'] == op['inputs'][1]:
            # input_tensor = interpreter.get_tensor(tensor_details["index"])
            positions_channel = tensor_details['shape'][1]
            # input_height = tensor_details['shape'][1]
            # input_width = tensor_details['shape'][2]
            positions_dims_raw = '{' + '1,' + str(positions_channel) + '}'
            positions_dims_size = len(tensor_details['shape'])
            tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
            # quantization_input = tensor_details['quantization']
            kwargs['positions_dims_size='] = 'positions_dims_size=' + str(positions_dims_size)
            kwargs['positions_dims_raw='] = 'positions_dims_raw[' + str(positions_dims_size) + ']=' + positions_dims_raw
            # kwargs['input_type='] = 'input_type=' + tflite_type
            # kwargs['scale_input='] = 'scale_input=' + str(quantization_input[0])
            # kwargs['zero_point_input='] = 'zero_point_input=' + str(quantization_input[1])

        # print(kwargs)
    return kwargs, unknown_config