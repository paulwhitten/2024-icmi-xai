#!/bin/bash

#python transform_parallel.py -i /home/pcw/mnist/download/train-images-idx3-ubyte -l /home/pcw/mnist/download/train-labels-idx1-ubyte -o ./transform_train -d

#python transform_parallel.py -i /home/pcw/mnist/download/t10k-images-idx3-ubyte -l /home/pcw/mnist/download/t10k-labels-idx1-ubyte -o ./transform_test -d

python train_svm_transforms.py -r ./transform_train -e ./transform_test -o ./svm_models > out-svm-training.txt

python train_nn_transforms.py -r ./transform_train -e ./transform_test -o ./mlp_relu_models > out-mlp_relu-training.txt

