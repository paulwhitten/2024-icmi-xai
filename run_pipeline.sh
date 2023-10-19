#!/bin/bash

#python transform_parallel.py -i /home/pcw/mnist/download/train-images-idx3-ubyte -l /home/pcw/mnist/download/train-labels-idx1-ubyte -o ./transform_train -d

python transform_parallel.py -i ~/emnist/data/emnist-balanced-test-images-idx3-ubyte -l ~/emnist/data/emnist-balanced-test-labels-idx1-ubyte -o transform_emnist_test

python train_svm_transforms.py -r ./transform_emnist_train -e ./transform_emnist_test -o ./models_svm_emnist > out-svm-emnist-training.txt

python build_kb.py -r ./transform_emnist_train -e ./transform_emnist_test -m models_svm_emnist -o ./kb_svm_emnist
