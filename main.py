import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import json
import random
import argparse
# import orjson
# from tensorflow import keras
from model_assembler import model_assembler
from model_parser import *
from utils.utils import *
from dinamic_config import dinamic_config

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='lenet', help='name of the model')
parser.add_argument('--free_unused_data', action='store_true', help='free unused intermediate data')
parser.add_argument('--executable', action='store_true', help='generate executable file')
opt = parser.parse_args()

def reduce_size_json(json_file):
    with fileinput.input(files=json_file, inplace=True) as f:
        keep_sign = True
        for line in f:
            if 'buffers:' in line:
                keep_sign = False
                print('}', end="")
            if keep_sign:
                print(line, end="")

model_path = './tflite_model/'
model_name = opt.model_name + '.tflite'
if opt.model_name == 'gpt2':
    enable_sig = False
else:
    enable_sig = True
interpreter = tf.lite.Interpreter(
 os.path.join(model_path, model_name)
)
interpreter.allocate_tensors()
# --------------------------------------------------
# parse the TFLite model and generate code
# --------------------------------------------------
os.system('flatc -t schema.fbs -- %s' % os.path.join(model_path, model_name))
reduce_size_json(os.path.splitext(model_name)[0] + '.json')
os.system('jsonrepair %s.json --overwrite' % os.path.splitext(model_name)[0])
# for op in interpreter._get_ops_details():
#     print(op)

with open('%s.json' % os.path.splitext(model_name)[0],'r') as f:
    model_json_f = f.read()
model_json = json.loads(model_json_f)

# op_details = interpreter._get_ops_details()
# print(op_details)

# for tensor_details in interpreter.get_tensor_details():
#     print(tensor_details)

tensor_list = []
for input in interpreter.get_input_details():
    tensor_list.append(input['index'])
for tensor_details in interpreter.get_tensor_details():
    tensor_list.append(tensor_details["index"])
tensor_list.sort()

inout_list = []
for i in range(len(model_json['subgraphs'][0]["operators"])):
    # print(model_json['subgraphs'][0]["operators"][i])
    for j in range(len(model_json['subgraphs'][0]["operators"][i]['outputs'])):
        inout_list.append(model_json['subgraphs'][0]["operators"][i]['outputs'][j])

for input in interpreter.get_input_details():
    inout_list.append(input['index'])

# for output in interpreter.get_output_details():
#     inout_list.append(output['index'])

jsontext, unknown_config = lib_generator(model_json, interpreter, inout_list)

# --------------------------------------------------
# dinamic config & build
# --------------------------------------------------
currentPath = os.getcwd().replace('\\','/')
# os.chdir('./tensorflow-2.9.1/')
# os.system("bash build.sh")
os.chdir(currentPath)
# print(inout_list)
for op in jsontext['oplist']:
    del_list = []
    # print("input:", op['input'])
    for i in range(len(op['input'])):
        if not (op['input'][i] in inout_list):
            # print("not in inout_list:", op['input'][i])
            del_list.append(op['input'][i])
    # print("del:", del_list)
    for j in range(len(del_list)):
        op['input'].remove(del_list[j])

    out_node = op['output'][0]
    try:
        model_json['subgraphs'][0]["tensors"][out_node]["type"]
    except:
        op['type'] = "FLOAT32"
    else:
        op['type'] = model_json['subgraphs'][0]["tensors"][out_node]["type"]
    try:
        model_json['subgraphs'][0]["tensors"][out_node]["quantization"]
    except:
        op["quantization"] = {}
    else:
        op["quantization"] = model_json['subgraphs'][0]["tensors"][out_node]["quantization"]

input_list = model_json['subgraphs'][0]['inputs']
jsontext['inputs'] = []
for i in range(len(input_list)):
    try:
        tensor_type = model_json['subgraphs'][0]["tensors"][input_list[i]]["type"]
    except:
        tensor_type = "FLOAT32"
    else:
        tensor_type = model_json['subgraphs'][0]["tensors"][input_list[i]]["type"]
    jsontext['inputs'].append({'name': 'serving_default_x:'+str(i), 'type': tensor_type, 'quantization': model_json['subgraphs'][0]["tensors"][input_list[i]]["quantization"]})

output_list = model_json['subgraphs'][0]['outputs']
jsontext['outputs'] = []
for i in range(len(output_list)):
    try:
        tensor_type = model_json['subgraphs'][0]["tensors"][input_list[i]]["type"]
    except:
        tensor_type = "FLOAT32"
    else:
        tensor_type = model_json['subgraphs'][0]["tensors"][input_list[i]]["type"]
    jsontext['outputs'].append({'name': 'PartitionedCall:'+str(i), 'type': tensor_type, 'quantization': model_json['subgraphs'][0]["tensors"][output_list[i]]["quantization"]})

jsondata = json.dumps(jsontext,indent=4,separators=(',', ': '))

file = open('./obfjson/model' + '_' + opt.model_name + '.json', 'w')
file.write(jsondata)
file.close()

model_assembler(interpreter, './obfjson/model' + '_' + opt.model_name + '.json', opt.free_unused_data, enable_sig=enable_sig, executable=opt.executable)
dinamic_config(unknown_config, './obfjson/model' + '_' + opt.model_name + '.json', os.path.join(model_path, model_name), enable_sig=enable_sig, executable=opt.executable)
os.system('python eval.py --model_name=' + opt.model_name + ' --latency=True')
