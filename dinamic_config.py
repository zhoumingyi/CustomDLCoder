import json
import itertools
import re,os
import fileinput
from eval import eval
source_path = './tensorflow-2.9.1/tensorflow/lite/examples/coder'


def data_checking(choice_list):
    t = choice_list[-1]
    for i in range(len(choice_list)-2,-1,-1):

        if t == choice_list[i]:
            choice_list.remove(choice_list[i])
        else:
            t = choice_list[i]
    return choice_list


def dinamic_config(unknown_config, json_path, model_path, enable_sig):
    if len(unknown_config) == 0 or enable_sig == False:
        os.system("cd coder_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/coder -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF && cd ..")
        os.system("cd coder_x86_build && cmake --build . -j && cd ..")
        diff = eval(model_path, enable_sig)
        if diff < 1e-5:
            print("find the config")
            print("Output difference: ", diff)
    else:
        tfl_filelist = os.listdir(source_path)

        with open(json_path,'r') as f:
            model_json_f = f.read()
        model_json = json.loads(model_json_f)

        # print(unknown_config)
        choice_list = []
        for unknown_data in unknown_config:

            for values in unknown_data.values():
                choice_list.append(values)
            choice_list.pop()                 # remove opname
            print(choice_list)
        choice_list = data_checking(choice_list)  # reduce searching space
        iter_prod = []
        for i in range(len(choice_list)):
            if i == 0:
                iter_prod = choice_list[i]
            else:
                iter_prod = itertools.product(iter_prod, choice_list[i])
        # os.system("cd minimal_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/minimal -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=ON -DTFLITE_ENABLE_GPU=OFF && cd ..")
        # os.system("cd minimal_x86_build && cmake --build . -j && cd ..")
        for choice_comb in iter_prod:
            print(choice_comb)
            for unknown_data in unknown_config:
                kwargs = {}
                for op in model_json['oplist']:
                    if op["OpName"] == unknown_data["opname"]:
                        for unknown_key, unknown_value in unknown_data.items():
                            if unknown_key == "opname":
                                continue
                            for choice in choice_comb:
                                if choice in unknown_value:
                                    kwargs[unknown_key] = choice
                        # print(kwargs)
                        for i in range(len(tfl_filelist)):
                            if op["LayerID"] == os.path.splitext(tfl_filelist[i])[0]:
                                with fileinput.input(files=os.path.join(source_path,("%s.cc" % op["LayerID"])), inplace=True) as f:

                                    for line in f:
                                        find_key = False
                                        for key in kwargs:
                                            if key in line:
                                                print(re.sub(key,kwargs[key],line), end="")
                                                find_key = True
                                        if not find_key:
                                            print(line, end="")
                        break
            os.system("cd coder_x86_build && cmake ../tensorflow-2.9.1/tensorflow/lite/examples/coder -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_MMAP=OFF -DTFLITE_ENABLE_RUY=OFF -DTFLITE_ENABLE_NNAPI=OFF -DTFLITE_ENABLE_GPU=OFF && cd ..")
            os.system("cd coder_x86_build && cmake --build . -j && cd ..")
            diff = eval(model_path, enable_sig)
            if diff < 1e-5:
                print("find the config")
                print("Output difference: ", diff)
                break


