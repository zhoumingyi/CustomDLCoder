# import onnx
import numpy as np
from numpy.ctypeslib import ndpointer
import argparse
import onnxruntime as ort

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='lenet', help='name of the model')
    parser.add_argument('--int', type=bool, help='int input')
    opt = parser.parse_args()

    options = ort.SessionOptions()
    providers = ['CPUExecutionProvider']
    ort_sess = ort.InferenceSession('out_model/' + opt.model_name + '.onnx', sess_options=options)
    input_names = [input.name for input in ort_sess.get_inputs()]
    input_shapes = {}
    for input in ort_sess.get_inputs():
        input_name = input.name
        input_shape = input.shape
        input_shapes[input_name] = input_shape

    if opt.int:
        inputs = np.random.randint(0, 255, size=tuple(input_shapes[input_name])).astype(np.int32)
    else:
        inputs = np.random.randn(input_shapes[input_name][0], input_shapes[input_name][1], input_shapes[input_name][2], input_shapes[input_name][3]).astype(np.float32)

    output_names = [input.name for input in ort_sess.get_outputs()]
    outputs = np.array(ort_sess.run([output_names[0]], {input_names[0]: inputs}))




