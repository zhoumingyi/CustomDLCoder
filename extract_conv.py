import numpy as np
from utils.utils import *

def extract_conv(op, kwargs, interpreter, unknown_config):
    # print(op['inputs'][2])
    # load the stride
    try:
        stride_w = op['builtin_options']['stride_w']
    except:
        kwargs['stride_width='] = 'stride_width=1'
        stride_w = 1
    else:
        kwargs['stride_width='] = 'stride_width=' + str(stride_w)
    try:
        stride_h = op['builtin_options']['stride_h']
    except:
        kwargs['stride_height='] = 'stride_height=1'
        stride_h = 1
    else:
        kwargs['stride_height='] = 'stride_height=' + str(stride_h)
    # load the dilation_
    try:
        dilation_w_factor = op['builtin_options']['dilation_w_factor']
    except:
        kwargs['dilation_width_factor='] = 'dilation_width_factor=1'
    else:
        kwargs['dilation_width_factor='] = 'dilation_width_factor=' + str(dilation_w_factor)
    try:
        dilation_h_factor = op['builtin_options']['dilation_h_factor']
    except:
        kwargs['dilation_height_factor='] = 'dilation_height_factor=1'
    else:
        kwargs['dilation_height_factor='] = 'dilation_height_factor=' + str(dilation_h_factor)
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
    if len(op['inputs']) > 2:
        kwargs['has_conv_bias='] = 'has_conv_bias=true'
    else:
        kwargs['has_conv_bias='] = 'has_conv_bias=false'
        kwargs['const TfLiteType bias_type=;'] = ' '
        kwargs['const int bias_dims_size=;'] = ' '
        kwargs['const int32_t bias_dims_raw=;'] = ' '
        kwargs['float bias_raw=;'] = ' '

    for tensor_details in interpreter.get_tensor_details():
        if tensor_details['index'] == op['inputs'][1]:
            for tensor_details_lower in interpreter.get_tensor_details():
                if tensor_details_lower['index'] == op['inputs'][0]:
                    tflite_type_in, _ = conv_data_type_parser(tensor_details_lower['dtype'])
                    if tflite_type_in == 'kTfLiteFloat32':
                        need_hwcn = True
            filter_tensor = interpreter.get_tensor(tensor_details["index"])
            filter_item_num = filter_tensor.size
            filter_input_channel = filter_tensor.shape[3]
            filter_output_channel = filter_tensor.shape[0]
            filter_height = filter_tensor.shape[1]
            filter_width = filter_tensor.shape[2]
            filter_dims_raw = '{' + str(filter_output_channel) + ',' + str(filter_height)  + ',' + str(filter_width)  + ',' + str(filter_input_channel) +'}'
            filter_dims_size = len(filter_tensor.shape)
            tflite_type, type_str = conv_data_type_parser(filter_tensor.dtype)
            quantization_filter = tensor_details['quantization']
            kwargs['filter_dims_size='] = 'filter_dims_size=' + str(filter_dims_size)
            kwargs['filter_dims_raw='] = 'filter_dims_raw[' + str(filter_dims_size) + ']=' + filter_dims_raw
            kwargs['filter_type='] = 'filter_type=' + tflite_type
            if need_hwcn:
                kwargs['filter_raw='] = type_str + ' filter_raw[' + str(filter_item_num) + ']=' + '{' + str(np.transpose(filter_tensor, (1,2,3,0)).flatten('C').tolist()).strip('[').strip(']') + '}'
            else:
                kwargs['filter_raw='] = type_str + ' filter_raw[' + str(filter_item_num) + ']=' + '{' + str(filter_tensor.flatten('C').tolist()).strip('[').strip(']') + '}'
            kwargs['scale_filter='] = 'scale_filter=' + str(quantization_filter[0])
            kwargs['zero_point_filter='] = 'zero_point_filter=' + str(quantization_filter[1])
            kwargs['filter_tensor_data=filter_raw'] = type_str + '* filter_tensor_data=filter_raw'

        elif tensor_details['index'] == op['inputs'][2]:
            bias_tensor = interpreter.get_tensor(tensor_details["index"])
            bias_item_num = bias_tensor.size
            bias_channel = bias_tensor.shape[0]
            bias_dims_raw = '{' + str(bias_channel) + '}'
            bias_dims_size = len(bias_tensor.shape)
            tflite_type, type_str = conv_data_type_parser(bias_tensor.dtype)
            quantization_bias = tensor_details['quantization']
            kwargs['bias_type='] = 'bias_type=' + tflite_type
            kwargs['bias_dims_size='] = 'bias_dims_size=' + str(bias_dims_size)
            kwargs['bias_dims_raw='] = 'bias_dims_raw[' + str(bias_dims_size) + ']=' + bias_dims_raw
            kwargs['bias_raw='] = type_str + ' bias_raw[' + str(bias_item_num) + ']=' + '{' + str(bias_tensor.tolist()).strip('[').strip(']') + '}'
            kwargs['scale_bias='] = 'scale_bias=' + str(quantization_bias[0])
            kwargs['zero_point_bias='] = 'zero_point_bias=' + str(quantization_bias[1])
            kwargs['bias_tensor_data=bias_raw'] = type_str + '* bias_tensor_data=bias_raw'
        elif tensor_details['index'] == op['outputs'][0]:
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
        kwargs['padding_values_width='] = 'padding_values_width=' + str(np.floor((filter_width - stride_w) / 2.0))
        kwargs['padding_values_height='] = 'padding_values_height=' + str(np.floor((filter_height - stride_h) / 2.0))
    return kwargs, unknown_config