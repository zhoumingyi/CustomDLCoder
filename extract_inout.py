import numpy as np
from utils.utils import *

def extract_inout(op, kwargs, interpreter, unknown_config):
    for tensor_details in interpreter.get_tensor_details():
        if tensor_details['index'] == op['outputs'][0]:
            # output_tensor = interpreter.get_tensor(tensor_details["index"])
            output_dims_size = len(tensor_details['shape'])
            output_num = 1
            for i in range(output_dims_size):
                output_num = output_num * tensor_details['shape'][i]
            tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
            quantization_output = tensor_details['quantization']
            kwargs['output_dims_size='] = 'output_dims_size=' + str(output_dims_size)
            kwargs['output_dims_raw='] = 'output_dims_raw[' + str(output_dims_size) + ']=' + str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}')
            kwargs['output_num='] = 'output_num=' + str(output_num)
            kwargs['output_type='] = 'output_type=' + tflite_type
            kwargs['scale_output='] = 'scale_output=' + str(quantization_output[0])
            kwargs['zero_point_output='] = 'zero_point_output=' + str(quantization_output[1])
        elif tensor_details['index'] == op['inputs'][0]:
            # input_tensor = interpreter.get_tensor(tensor_details["index"])
            input_channel = tensor_details['shape'][1]
            input_dims_size = len(tensor_details['shape'])
            tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
            quantization_input = tensor_details['quantization']
            kwargs['input_dims_size='] = 'input_dims_size=' + str(input_dims_size)
            kwargs['input_dims_raw='] = 'input_dims_raw[' + str(input_dims_size) + ']=' + str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}')
            kwargs['input_type='] = 'input_type=' + tflite_type
            kwargs['scale_input='] = 'scale_input=' + str(quantization_input[0])
            kwargs['zero_point_input='] = 'zero_point_input=' + str(quantization_input[1])
    return kwargs, unknown_config


def extract_out(op, kwargs, interpreter, unknown_config):
    for tensor_details in interpreter.get_tensor_details():
        if tensor_details['index'] == op['outputs'][0]:
            # output_tensor = interpreter.get_tensor(tensor_details["index"])
            output_dims_size = len(tensor_details['shape'])
            output_num = 1
            for i in range(output_dims_size):
                output_num = output_num * tensor_details['shape'][i]
            tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
            quantization_output = tensor_details['quantization']
            kwargs['output_dims_size='] = 'output_dims_size=' + str(output_dims_size)
            kwargs['output_dims_raw='] = 'output_dims_raw[' + str(output_dims_size) + ']=' + str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}')
            kwargs['output_num='] = 'output_num=' + str(output_num)
            kwargs['output_type='] = 'output_type=' + tflite_type
            kwargs['scale_output='] = 'scale_output=' + str(quantization_output[0])
            kwargs['zero_point_output='] = 'zero_point_output=' + str(quantization_output[1])
    return kwargs, unknown_config