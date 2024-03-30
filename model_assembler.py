import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import json
import numpy as np
import tensorflow as tf
import fileinput
from utils.utils import *
# from tf_model import *
# model_path = '/data/mingyi/code/obf_tf/tflite_model/mobilenet_v1_0.75_224_1_default_1.tflite'
# def model_assembler(input, model_json, interpreter):


def model_assembler(interpreter, json_path='./ObfusedModel.json', free_unused_data=True, enable_sig=True):
    with open(json_path,'r') as f:
        model_json_f = f.read()
    model_json = json.loads(model_json_f)

    input_id = interpreter.get_input_details()[0]['index']
    output_id = [idx['index'] for idx in interpreter.get_output_details()]

    OpList = model_json['oplist']
    OpIDList =[]
    for op in model_json['oplist']:
        OpIDList.append(op['LayerID'])

    model_file = './tensorflow-2.9.1/tensorflow/lite/examples/coder/coder.cc'
    # with fileinput.input(files=model_file, inplace=True) as f:
    #     del_sign = False
    #     for line in f:
    #         if 'end file' in line or 'end function' in line:
    #             del_sign = False
    #         if 'add file' in line or 'add function' in line:
    #             print(line, end="")
    #             del_sign = True
    #         if not del_sign:
    #             print(line, end="")


    def get_inout_string(inout_list):
        input_string = []
        for i in range(len(inout_list)):
            input_string.append('out_' + str(inout_list[i]))
        return str(input_string).strip('[').strip(']').replace('\'', '')


    get_tensor= []
    get_tensor.append(input_id)

    out_id_list = []
    one_time_list = ['cpu_backend']

    with fileinput.input(files=model_file, inplace=True) as f:
        for line in f:
            if 'end file' in line:
                for i in range(len(OpIDList)):
                    print('#include \"%s.cc\"' % (OpIDList[i]))
            elif 'end function' in line:
                print(' auto* out_%s = input_v;' % (input_id))
                while(len(OpIDList)):
                    for i in range(len(OpIDList)):
                        if set(OpList[i]['input']) <= set(get_tensor):
                            if OpList[i]['OpName'] == 'Conv2DOptions':
                                print((' auto* %s = %s::%s(%s, device);' % (get_inout_string(OpList[i]['output']), OpList[i]['LayerID'], OpList[i]['LayerID'], get_inout_string(OpList[i]['input']))))
                            elif OpList[i]['OpName'] == 'DepthwiseConv2DOptions' or OpList[i]['OpName'] == 'FullyConnectedOptions' or OpList[i]['OpName'] == 'SoftmaxOptions':
                                print((' auto* %s = %s::%s(%s, cpu_backend);' % (get_inout_string(OpList[i]['output']), OpList[i]['LayerID'], OpList[i]['LayerID'], get_inout_string(OpList[i]['input']))))
                            elif OpList[i]['OpName'] == 'SplitOptions':
                                print((' auto [%s] = %s::%s(%s);' % (get_inout_string(OpList[i]['output']), OpList[i]['LayerID'], OpList[i]['LayerID'], get_inout_string(OpList[i]['input']))))
                            else:
                                print((' auto* %s = %s::%s(%s);' % (get_inout_string(OpList[i]['output']), OpList[i]['LayerID'], OpList[i]['LayerID'], get_inout_string(OpList[i]['input']))))
                            for j in range(len(OpList[i]['output'])):
                                get_tensor.append(OpList[i]['output'][j])

                            for m in range(len(OpList[i]['output'])):
                                if OpList[i]['output'][m] not in output_id:
                                    out_id_list.append(OpList[i]['output'][m])

                            del OpIDList[i]
                            del OpList[i]

                            free_sign = True
                            cpu_backend_sign = True
                            removal_list = []
                            for m in range(len(out_id_list)):
                                for n in range(len(OpList)):
                                    if out_id_list[m] in OpList[n]['input']:
                                        free_sign = False
                                    if (OpList[n]['OpName'] in ['DepthwiseConv2DOptions', 'FullyConnectedOptions', 'SoftmaxOptions']) and cpu_backend_sign:
                                        cpu_backend_sign = False

                                if free_sign and free_unused_data:
                                    print(' free(%s);' % (get_inout_string(out_id_list[m:(m+1)])))
                                    removal_list.append(out_id_list[m])
                            if cpu_backend_sign and 'cpu_backend' in one_time_list:
                                print(' free(cpu_backend);')
                                one_time_list.remove('cpu_backend')

                            for m in range(len(removal_list)):
                                out_id_list.remove(removal_list[m])
                            break
                output_id = output_id[:1]
                print(' return %s;' % (get_inout_string(output_id)))
            if 'const Eigen::ThreadPoolDevice' in line:
                if enable_sig:
                    print(' const Eigen::ThreadPoolDevice* device = tflite::eigen_support::CreateThreadPoolDevice(-1);')
                else:
                    print('// const Eigen::ThreadPoolDevice* device = tflite::eigen_support::CreateThreadPoolDevice(1);')
            elif 'const float* coder' in line: 
                if enable_sig:
                    print('extern \"C\" const float* coder(float *input_v) {')
                else:
                    print('extern \"C\" const float* coder(int *input_v) {')
            else:
                print(line, end="")

