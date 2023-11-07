#!/bin/bash

#mkdir -p ./transforms/emnist/train

#python transform_parallel.py -i /archive/xai/emnist/downloads/emnist-balanced-train-images-idx3-ubyte -l /archive/xai/emnist/downloads/emnist-balanced-train-labels-idx1-ubyte -o ./transforms/emnist/train

#mkdir -p ./transforms/emnist/test

#python transform_parallel.py -i /archive/xai/emnist/downloads/emnist-balanced-test-images-idx3-ubyte -l /archive/xai/emnist/downloads/emnist-balanced-test-labels-idx1-ubyte -o ./transforms/emnist/test

mkdir -p ./models/emnist/svm

python train_svm_transforms.py -r ./transforms/emnist/train -e ./transforms/emnist/test -o ./models/emnist/svm > out-svm-emnist-training.txt

mkdir -p ./kb/emnist/svm

cp ./models/emnist/svm/*.json kb/emnist/svm

python build_kb.py -r ./transforms/emnist/train -e ./transforms/emnist/test -m ./models/emnist/svm -o kb/emnist/svm

node process_kb.js -k kb/emnist/svm > output-emnist-svm-result.txt
