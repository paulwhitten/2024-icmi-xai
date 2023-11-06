#!/bin/bash

mkdir -p /archive/xai/emnist/transforms_emnist_train

python transform_parallel.py -i /archive/xai/emnist/downloads/emnist-balanced-train-images-idx3-ubyte -l /archive/xai/emnist/downloads/emnist-balanced-train-labels-idx1-ubyte -o /archive/xai/emnist/transforms_emnist_train

mkdir -p /archive/xai/emnist/transforms_emnist_test

python transform_parallel.py -i /archive/xai/emnist/downloads/emnist-balanced-test-images-idx3-ubyte -l /archive/xai/emnist/downloads/emnist-balanced-test-labels-idx1-ubyte -o /archive/xai/emnist/transforms_emnist_test

mkdir -p /archive/xai/models/emnist_svm

python train_svm_transforms.py -r /archive/xai/emnist/transforms_emnist_train -e /archive/xai/emnist/transforms_emnist_test -o /archive/xai/models/emnist_svm > out-svm-emnist-training.txt

mkdir -p kb_svm_emnist

python build_kb.py -r /archive/xai/emnist/transforms_emnist_train -e /archive/xai/emnist/transforms_emnist_test -m /archive/xai/models/emnist_svm -o kb_svm_emnist

