import numpy as np
from utils.utils import *

def extract_split(op, kwargs, interpreter, unknown_config):
    lenth_out = len(op['outputs'])
    print('the lenth of outputs is: ', lenth_out)
    # kwargs['float* input_v'] = ''
    output_string = ''
    type_string = ''
    for i in range(lenth_out):
        for tensor_details in interpreter.get_tensor_details():
            if tensor_details['index'] == op['outputs'][i]:
                # print(str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}'))
                kwargs['output_' + str(i) + '_dims_raw='] = 'output_' + str(i) + '_dims_raw[' + str(len(tensor_details['shape'])) + ']=' + str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}')
                kwargs['output_' + str(i) + '_dims_size='] = 'output_' + str(i) + '_dims_size=' + str(len(tensor_details['shape']))
                kwargs['all_data_output_v_' + str(i) + ';'] = 'all_data_.push_back(output_' + str(i) + '_data);'
                kwargs['RuntimeShape_output_' + str(i) + ';'] = 'all_shape_.push_back(RuntimeShape(output_' + str(i) + '_dims_size, output_' + str(i) + '_dims_raw));'

                output_num = 1
                for j in range(len(tensor_details['shape'])):
                    output_num = output_num * tensor_details['shape'][j]
                kwargs['output_' + str(i) + '_num='] = 'output_' + str(i) + '_num=' + str(output_num)
        output_string += 'output_' + str(i) + '_data,'
        type_string += 'float*,'
    kwargs['outputs_placeholder'] = output_string[:-1]
    kwargs['type_placeholder'] = type_string[:-1]
    kwargs['num_splits='] = 'num_splits='+str(lenth_out)

    for i in range(3 - lenth_out):
        kwargs['const int32_t output_' + str(9-i) + '_dims_raw=;'] = ''
        kwargs['const int output_' + str(9-i) + '_dims_size=;'] = ''
        kwargs['all_data_output_v_' + str(9-i) + ';'] = ''
        kwargs['RuntimeShape_output_' + str(9-i) + ';'] = ''
        # kwargs['RuntimeShape(output_' + str(9-i) + '_dims_size, output_' + str(9-i) + '_dims_raw)'] = ''


    for tensor_details in interpreter.get_tensor_details():
        if tensor_details['index'] == op['inputs'][1]:
            input_dims_size = len(tensor_details['shape'])
            tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
            quantization_input = tensor_details['quantization']
            kwargs['input_dims_size='] = 'input_dims_size=' + str(input_dims_size)
            kwargs['input_dims_raw='] = 'input_dims_raw[' + str(input_dims_size) + ']=' + str(tuple(tensor_details['shape'])).replace('(', '{').replace(')', '}')
            kwargs['input_type='] = 'input_type=' + tflite_type
            kwargs['scale_input='] = 'scale_input=' + str(quantization_input[0])
            kwargs['zero_point_input='] = 'zero_point_input=' + str(quantization_input[1])
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