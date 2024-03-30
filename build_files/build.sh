#!/usr/bin/env bash 

bazel  build //tensorflow/tools/pip_package:build_pip_package
# sleep 5
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
python -m pip uninstall -y tensorflow
python -m pip install /tmp/tensorflow_pkg/tensorflow-2.9.1-cp39-cp39-linux_x86_64.whl