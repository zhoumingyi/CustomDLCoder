import numpy as np
from utils.utils import *

def extract_concat(op, kwargs, interpreter, unknown_config):
    lenth_in = len(op['inputs'])
    print('the lenth of input is: ', lenth_in)
    # kwargs['float* input_v'] = ''
    input_string = ''
    for i in range(lenth_in):
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op['inputs'][i]:
                # print(str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}'))
                kwargs['input_' + str(i) + '_dims_raw='] = 'input_' + str(i) + '_dims_raw[' + str(len(tensor_details['shape'])) + ']=' + str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}')
                kwargs['input_' + str(i) + '_dims_size='] = 'input_' + str(i) + '_dims_size=' + str(len(tensor_details['shape']))
                kwargs['all_data_input_v_' + str(i) + ';'] = 'all_data_.push_back(input_v_' + str(i) + ');'
                kwargs['RuntimeShape_input_' + str(i) + ';'] = 'all_shape_.push_back(RuntimeShape(input_' + str(i) + '_dims_size, input_' + str(i) + '_dims_raw));'
        input_string += 'float* input_v_' + str(i) + ','
    kwargs['list_input_v'] = input_string[:-1]
    kwargs['tensor_num='] = 'tensor_num='+str(lenth_in)

    for i in range(10 - lenth_in):
        kwargs['const int32_t input_' + str(9-i) + '_dims_raw=;'] = ''
        kwargs['const int input_' + str(9-i) + '_dims_size=;'] = ''
        kwargs['all_data_input_v_' + str(9-i) + ';'] = ''
        kwargs['RuntimeShape_input_' + str(9-i) + ';'] = ''
        kwargs['RuntimeShape(input_' + str(9-i) + '_dims_size, input_' + str(9-i) + '_dims_raw)'] = ''


    for tensor_details in interpreter.get_tensor_details():
        if tensor_details['index'] == op['outputs'][0]:
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
        # elif tensor_details['index'] == op['inputs'][0]:
        #     # input_tensor = interpreter.get_tensor(tensor_details["index"])
        #     input_channel = tensor_details['shape'][3]
        #     input_height = tensor_details['shape'][1]
        #     input_width = tensor_details['shape'][2]
        #     input_dims_raw = '{' + '1,' + str(input_height) + ',' + str(input_width)  + ',' + str(input_channel) + '}'
        #     input_dims_size = len(tensor_details['shape'])
        #     tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
        #     quantization_input = tensor_details['quantization']
        #     kwargs['input_dims_size='] = 'input_dims_size=' + str(input_dims_size)
        #     kwargs['input_dims_raw='] = 'input_dims_raw[' + str(input_dims_size) + ']=' + input_dims_raw
        #     kwargs['input_type='] = 'input_type=' + tflite_type
        #     kwargs['scale_input='] = 'scale_input=' + str(quantization_input[0])
        #     kwargs['zero_point_input='] = 'zero_point_input=' + str(quantization_input[1])
        #     kwargs['input_tensor_data=input_raw'] = type_str + '* input_tensor_data=input_raw'
    return kwargs, unknown_config

def extract_pack(op, kwargs, interpreter, unknown_config):
    lenth_in = len(op['inputs'])
    print('the lenth of input is: ', lenth_in)
    # kwargs['float* input_v'] = ''
    input_string = ''
    for i in range(lenth_in):
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op['inputs'][i]:
                # print(str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}'))
                kwargs['input_' + str(i) + '_dims_raw='] = 'input_' + str(i) + '_dims_raw[' + str(len(tensor_details['shape'])) + ']=' + str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}')
                kwargs['input_' + str(i) + '_dims_size='] = 'input_' + str(i) + '_dims_size=' + str(len(tensor_details['shape']))
                kwargs['all_data_input_v_' + str(i) + ';'] = 'all_data_.push_back(input_v_' + str(i) + ');'
                kwargs['RuntimeShape_input_' + str(i) + ';'] = 'all_shape_.push_back(RuntimeShape(input_' + str(i) + '_dims_size, input_' + str(i) + '_dims_raw));'
        input_string += 'float* input_v_' + str(i) + ','
    kwargs['list_input_v'] = input_string[:-1]
    kwargs['tensor_num='] = 'tensor_num='+str(lenth_in)

    for i in range(12 - lenth_in):
        kwargs['const int32_t input_' + str(11-i) + '_dims_raw=;'] = ''
        kwargs['const int input_' + str(11-i) + '_dims_size=;'] = ''
        kwargs['  all_data_input_v_' + str(11-i) + ';'] = ''
        kwargs['  RuntimeShape_input_' + str(11-i) + ';'] = ''
        # kwargs['RuntimeShape(input_' + str(9-i) + '_dims_size, input_' + str(9-i) + '_dims_raw)'] = ''


    for tensor_details in interpreter.get_tensor_details():
        if tensor_details['index'] == op['outputs'][0]:
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
        # elif tensor_details['index'] == op['inputs'][0]:
        #     # input_tensor = interpreter.get_tensor(tensor_details["index"])
        #     input_channel = tensor_details['shape'][3]
        #     input_height = tensor_details['shape'][1]
        #     input_width = tensor_details['shape'][2]
        #     input_dims_raw = '{' + '1,' + str(input_height) + ',' + str(input_width)  + ',' + str(input_channel) + '}'
        #     input_dims_size = len(tensor_details['shape'])
        #     tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
        #     quantization_input = tensor_details['quantization']
        #     kwargs['input_dims_size='] = 'input_dims_size=' + str(input_dims_size)
        #     kwargs['input_dims_raw='] = 'input_dims_raw[' + str(input_dims_size) + ']=' + input_dims_raw
        #     kwargs['input_type='] = 'input_type=' + tflite_type
        #     kwargs['scale_input='] = 'scale_input=' + str(quantization_input[0])
        #     kwargs['zero_point_input='] = 'zero_point_input=' + str(quantization_input[1])
        #     kwargs['input_tensor_data=input_raw'] = type_str + '* input_tensor_data=input_raw'
    return kwargs, unknown_config