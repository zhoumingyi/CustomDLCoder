import numpy as np
from utils.utils import *

def extract_fc(op, kwargs, interpreter, unknown_config):
    try:
        activation = op['builtin_options']['fused_activation_function']
    except:
        kwargs['activation='] = 'activation=kTfLiteActNone'
        print("Warning: no activation function found")
    else:
        kwargs['activation='] = 'activation=' + conv_activation_parser(activation)
    try:
        weights_format = op['builtin_options']['weights_format']
    except:
        kwargs['weights_format='] = 'weights_format=kTfLiteFullyConnectedWeightsFormatDefault'
        print("Warning: no weights_format function found")
    else:
        kwargs['weights_format='] = 'weights_format=' + weights_format_parser(weights_format)
    try:
        asymmetric_quantize_inputs = op['builtin_options']['asymmetric_quantize_inputs']
    except:
        kwargs['asymmetric_quantize_inputs='] = 'asymmetric_quantize_inputs=false'
        print("Warning: no asymmetric_quantize_inputs function found")
    else:
        kwargs['asymmetric_quantize_inputs='] = 'asymmetric_quantize_inputs=' + str.lower(str(asymmetric_quantize_inputs))
    try:
        keep_num_dims = op['builtin_options']['keep_num_dims']
    except:
        kwargs['keep_num_dims='] = 'keep_num_dims=false'
        print("Warning: no keep_num_dims function found")
    else:
        kwargs['keep_num_dims='] = 'keep_num_dims=' + str.lower(str(keep_num_dims))
    if len(op['inputs']) > 2:
        kwargs['has_conv_bias='] = 'has_conv_bias=true'
    else:
        kwargs['has_conv_bias='] = 'has_conv_bias=false'
        kwargs['const TfLiteType bias_type=;'] = ''
        kwargs['const int bias_dims_size=;'] = ''
        kwargs['const int32_t bias_dims_raw=;'] = ''
        kwargs['bias_raw=;'] = ''
        kwargs['const float scale_bias=;'] = ''
        kwargs['const int32_t zero_point_bias=;'] = ''
        kwargs['bias_tensor_data=bias_raw;'] = ''
        kwargs['RuntimeShape(bias_dims_size, bias_dims_raw), bias_tensor_data,'] = 'GetTensorShape(bias),GetTensorData<float>(bias),'

    unknown_status = {}
    for tensor_details in interpreter.get_tensor_details():
        if tensor_details['index'] == op['inputs'][1]:

            add_aug_num = 0
            input_arg = []
            try:
                filter_tensor = interpreter.get_tensor(tensor_details["index"])
            except:
                kwargs['filter_raw=;'] = ''
                kwargs['filter_tensor_data=filter_raw;'] = ''
                add_aug_num += 1
                input_arg.append('float* input_v_' + str(i))
                kwargs['input_2_placeholder,'] = ''
            else:
                if np.all(filter_tensor == 0):
                    kwargs['filter_raw=;'] = ''
                    kwargs['filter_tensor_data=filter_raw;'] = ''
                    add_aug_num += 1
                    input_arg.append('float* input_v_' + str(i))
                    kwargs['input_2_placeholder'] = 'float* filter_tensor_data'
                else:
                    filter_tensor = interpreter.get_tensor(tensor_details["index"])
                    filter_item_num = 1
                    for i in range(0, len(tensor_details['shape'])):
                        filter_item_num = filter_item_num * tensor_details['shape'][i]
                    tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
                    kwargs['filter_raw='] = type_str + ' filter_raw[' + str(filter_item_num) + ']=' + '{' + str(filter_tensor.flatten('C').tolist()).strip('[').strip(']') + '}'
                    kwargs['filter_tensor_data=filter_raw'] = type_str + '* filter_tensor_data=filter_raw'
                    kwargs['input_2_placeholder,'] = ''

            # if np.size(filter_tensor) > 4000000:
            #     rows_per_split = np.size(filter_tensor) // 100
            #     print("rows_per_split: ", rows_per_split)
            #     # Create and initialize the split arrays
            #     split_arrays = []
            #     for i in range(100):
            #         start_row = i * rows_per_split
            #         end_row = (i + 1) * rows_per_split if i < 99 else None
            #         split_arrays.append(filter_tensor[start_row:end_row, :])
            #     # split_arrays = np.array_split(filter_tensor, 10)
            #     kwargs['filter_raw='] = ''

            #     for i, split_array in enumerate(split_arrays):
            #         header_file_name = f"array_part_{i + 1}.h"

            #         with open(header_file_name, 'w') as header_file:
            #             header_file.write(f'# ifndef ARRAY_PART_{i + 1}_H\n')
            #             header_file.write(f'# define ARRAY_PART_{i + 1}_H\n\n')
            #             header_file.write('extern float myLargeArrayPart{}[{}];\n\n'.format(i + 1, split_array.shape[0] * split_array.shape[1]))
            #             header_file.write('float myLargeArrayPart' + str(i + 1) + '[' + str(split_array.shape[0] * split_array.shape[1]) + ']=' + '{' + str(split_array.flatten('C').tolist()).strip('[').strip(']'))

            #             # # Write array initialization values
            #             # for row in split_array:
            #             #     header_file.write('    {')
            #             #     header_file.write(', '.join(map(str, row)))
            #             #     header_file.write('},\n')

            #             header_file.write('};\n\n')
            #             header_file.write('# endif')
            # filter_dims_raw = '{' + ','.join(str(e) for e in list(filter_tensor.shape)) +'}'
            filter_dims_raw = '{' + ','.join(str(e) for e in list(tuple(tensor_details['shape']))) +'}'
            filter_dims_size = len(tensor_details['shape'])
            quantization_filter = tensor_details['quantization']
            kwargs['filter_dims_size='] = 'filter_dims_size=' + str(filter_dims_size)
            kwargs['filter_dims_raw='] = 'filter_dims_raw[' + str(filter_dims_size) + ']=' + filter_dims_raw
            kwargs['filter_type='] = 'filter_type=' + tflite_type
            kwargs['scale_filter='] = 'scale_filter=' + str(quantization_filter[0])
            kwargs['zero_point_filter='] = 'zero_point_filter=' + str(quantization_filter[1])

        elif tensor_details['index'] == op['inputs'][2]:
            bias_tensor = interpreter.get_tensor(tensor_details["index"])
            # if np.all(bias_tensor) == 0:
            #     kwargs['has_conv_bias='] = 'has_conv_bias=false'
            #     kwargs['const TfLiteType bias_type=;'] = ''
            #     kwargs['const int bias_dims_size=;'] = ''
            #     kwargs['const int32_t bias_dims_raw=;'] = ''
            #     kwargs['bias_raw=;'] = ''
            #     kwargs['const float scale_bias=;'] = ''
            #     kwargs['const int32_t zero_point_bias=;'] = ''
            #     kwargs['bias_tensor_data=bias_raw;'] = ''
            #     kwargs['RuntimeShape(bias_dims_size, bias_dims_raw), bias_tensor_data,'] = 'GetTensorShape(bias), GetTensorData<float>(bias),'
            # elif kwargs['has_conv_bias='] == 'has_conv_bias=true':
            bias_item_num = bias_tensor.size
            # bias_channel = bias_tensor.shape[0]
            bias_dims_raw = '{' + ','.join(str(e) for e in list(bias_tensor.shape)) + '}'
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
            kwargs['  const TfLiteTensor* bias = nullptr;'] = ''
        elif tensor_details['index'] == op['outputs'][0]:
            # output_tensor = interpreter.get_tensor(tensor_details["index"])
            # output_channel = tensor_details['shape'][1]
            output_num = 1
            for i in range(0, len(tensor_details['shape'])):
                output_num = output_num * tensor_details['shape'][i]
            output_dims_raw = '{' + ','.join(str(e) for e in list(tensor_details['shape'])) + '}'
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
            # if len(tensor_details['shape']) == 4:
            #     input_channel = tensor_details['shape'][3]
            #     input_height = tensor_details['shape'][1]
            #     input_width = tensor_details['shape'][2]
            #     input_dims_raw = '{' + '1,' + str(input_height) + ',' + str(input_width)  + ',' + str(input_channel) + '}'
            # else:
            #     input_channel = tensor_details['shape'][1]
            #     input_dims_raw = '{' + '1,' + str(input_channel) + '}'
            input_dims_raw = '{' + ','.join(str(e) for e in list(tensor_details['shape'])) + '}'
            input_dims_size = len(tensor_details['shape'])
            tflite_type, type_str = conv_data_type_parser(tensor_details['dtype'])
            quantization_input = tensor_details['quantization']
            kwargs['input_dims_size='] = 'input_dims_size=' + str(input_dims_size)
            kwargs['input_dims_raw='] = 'input_dims_raw[' + str(input_dims_size) + ']=' + input_dims_raw
            kwargs['input_type='] = 'input_type=' + tflite_type
            kwargs['scale_input='] = 'scale_input=' + str(quantization_input[0])
            kwargs['zero_point_input='] = 'zero_point_input=' + str(quantization_input[1])
            # kwargs['input_tensor_data=input_raw'] = type_str + '* input_tensor_data=input_raw'
    unknown_status['lhs_cacheable=true'] = ["lhs_cacheable=true", "lhs_cacheable=false"]
    unknown_status['rhs_cacheable=true'] = ["rhs_cacheable=false", "rhs_cacheable=true"]
    unknown_status["opname"] = "FullyConnectedOptions"
    unknown_config.append(unknown_status)

    return kwargs, unknown_config