import numpy as np
from utils.utils import *

def extract_pool(op, kwargs, interpreter, unknown_config):
    try:
        stride_w = op['builtin_options']['stride_w']
    except:
        kwargs['stride_width='] = 'stride_width=1'
    else:
        kwargs['stride_width='] = 'stride_width=' + str(stride_w)
    try:
        stride_h = op['builtin_options']['stride_h']
    except:
        kwargs['stride_height='] = 'stride_height=1'
    else:
        kwargs['stride_height='] = 'stride_height=' + str(stride_h)
    # load the dilation_
    try:
        filter_height = op['builtin_options']['filter_height']
    except:
        # kwargs['filter_height='] = 'filter_height=2'
        raise ValueError("no filter_height found")
    else:
        kwargs['filter_height='] = 'filter_height=' + str(filter_height)
    try:
        filter_width = op['builtin_options']['filter_width']
    except:
        # kwargs['filter_width='] = 'filter_width=1'
        raise ValueError("no filter_width found")
    else:
        kwargs['filter_width='] = 'filter_width=' + str(filter_width)
    # load the activation
    try:
        fused_activation_function = op['builtin_options']['fused_activation_function']
    except:
        kwargs['activation='] = 'activation=kTfLiteActNone'
        print("Warning: no activation function found")
    else:
        kwargs['activation='] = 'activation=' + conv_activation_parser(fused_activation_function)
    # load the padding
    try:
        padding = op['builtin_options']['padding']
    except:
        kwargs['paddings='] = 'paddings=kTfLitePaddingSame'
        print("Warning: no paddings found and using default padding SAME")
    else:
        kwargs['paddings='] = 'paddings=' + conv_padding_parser(padding)
    for tensor_details in interpreter.get_tensor_details():
        if tensor_details['index'] == op['outputs'][0]:
            # output_tensor = interpreter.get_tensor(tensor_details["index"])
            output_channel = tensor_details['shape'][3]
            output_height = tensor_details['shape'][1]
            output_width = tensor_details['shape'][2]
            output_num = output_channel*output_height*output_width
            output_dims_raw = '{' + '1,' + str(output_height) + ',' + str(output_width)  + ',' + str(output_channel) + '}'
            output_dims_size = len(tensor_details['shape'])
            tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
            quantization_output = tensor_details['quantization']
            kwargs['output_dims_size='] = 'output_dims_size=' + str(output_dims_size)
            kwargs['output_dims_raw='] = 'output_dims_raw[' + str(output_dims_size) + ']=' + output_dims_raw
            kwargs['output_num='] = 'output_num=' + str(output_num)
            kwargs['output_type='] = 'output_type=' + tflite_type
            kwargs['scale_output='] = 'scale_output=' + str(quantization_output[0])
            kwargs['zero_point_output='] = 'zero_point_output=' + str(quantization_output[1])
        elif tensor_details['index'] == op['inputs'][0]:
            # input_tensor = interpreter.get_tensor(tensor_details["index"])
            input_channel = tensor_details['shape'][3]
            input_height = tensor_details['shape'][1]
            input_width = tensor_details['shape'][2]
            input_dims_raw = '{' + '1,' + str(input_height) + ',' + str(input_width)  + ',' + str(input_channel) + '}'
            input_dims_size = len(tensor_details['shape'])
            tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
            quantization_input = tensor_details['quantization']
            kwargs['input_dims_size='] = 'input_dims_size=' + str(input_dims_size)
            kwargs['input_dims_raw='] = 'input_dims_raw[' + str(input_dims_size) + ']=' + input_dims_raw
            kwargs['input_type='] = 'input_type=' + tflite_type
            kwargs['scale_input='] = 'scale_input=' + str(quantization_input[0])
            kwargs['zero_point_input='] = 'zero_point_input=' + str(quantization_input[1])
            # kwargs['input_tensor_data=input_raw'] = type_str + '* input_tensor_data=input_raw'
    if kwargs['paddings='] == 'paddings=kTfLitePaddingValid':
        kwargs['padding_values_width='] = 'padding_values_width=0'
        kwargs['padding_values_height='] = 'padding_values_height=0'
    elif kwargs['paddings='] == 'paddings=kTfLitePaddingSame':
        kwargs['padding_values_width='] = 'padding_values_width=' + str(int((filter_width - 1) / 2))
        kwargs['padding_values_height='] = 'padding_values_height=' + str(int((filter_height - 1) / 2))
    return kwargs, unknown_config