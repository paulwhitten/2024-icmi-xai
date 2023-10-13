import sys
import pickle
import argparse
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from load_mnist_data import load_mnist_float, get_num_classes
#from joblib import dump, load # more efficient serialization

def analyze_results(batch, labels, predictions):
    print("\n################################")
    print(batch, "accuracy analysis ")
    num_classes = get_num_classes(labels)
    class_counts = [0.0] * num_classes
    class_correct = [0.0] * num_classes
    conf_matrix = []
    class_header = []
    for i in range(num_classes):
        class_header.append(i)
        conf_matrix.append([0] * num_classes)
    for pred_ix, p in enumerate(predictions):
        if labels[pred_ix] == p:
            class_correct[labels[pred_ix]] += 1.0
        class_counts[labels[pred_ix]] += 1.0
        conf_matrix[labels[pred_ix]][p] += 1
    # output results
    print("\n\nAccuracy per class:\n| class | accuracy |\n| :---: | :---: |")
    for i in range(num_classes):
        print("|", i, " | ", class_correct[i] / class_counts[i], " |")
    print("\n")
    print("confusion matrix:")
    conf_matrix_string = str(class_header) + "\n"
    for row_ix, row in enumerate(conf_matrix):
        conf_matrix_string += str(row_ix)
        for n in row:
            conf_matrix_string += ", " + str(n)
        conf_matrix_string += "\n"
    print(conf_matrix_string)
    print("\n################################")

"""
This program trains all of the transform models
"""

def train_batch(batch):
    print("====", batch[0], "====")
    batch_start_time = datetime.now()

    N, train_rows, train_columns, train_images, train_labels = load_mnist_float(batch[1], batch[2])
    n, test_rows, test_columns, test_images, test_labels = load_mnist_float(batch[3], batch[4])

    # create the mlp model
    hidden_layers = (128, 128)
    act_func = "relu"
    start_time = datetime.now()
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=act_func, max_iter=500, alpha=1e-4,
                        solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=.1)

    print("Fitting MLP")
    mlp.fit(train_images, train_labels)
    end_time = datetime.now()
    print("Fit MLP in:", end_time - start_time)

    #test MLP
    start_time = datetime.now()
    mlp_predict = mlp.predict(test_images)
    end_time = datetime.now()
    print("Test MLP in:", end_time - start_time)
    mlp_p = np.argmax(mlp_predict, axis=-1)
    t_labels = np.argmax(test_labels, axis=-1)
    print("MLP Accuracy:", accuracy_score(t_labels, mlp_p))

    analyze_results(batch[0], t_labels, mlp_p)

    # save https://scikit-learn.org/stable/model_persistence.html
    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
    pickle.dump(mlp, open(batch[5] + "/" + batch[0] + ".model", 'wb'))
    # open by running loaded_model = pickle.load(open(filename, 'rb'))

    batch_end_time = datetime.now()
    print(batch[0], "done in:", batch_end_time - batch_start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trains transforms for a set')
    parser.add_argument('-r', '--train_folder', 
                        help='The training input folder')
    parser.add_argument('-e', '--test_folder', 
                        help='The test input folder')
    parser.add_argument('-o', '--output_folder', 
                        help='The folder to output')
    args = parser.parse_args()

    batches = [
        ["raw", f'{args.train_folder}/raw-image', f'{args.train_folder}/raw-labels',
        f'{args.test_folder}/raw-image', f'{args.test_folder}/raw-labels', args.output_folder],

        ["crossing", f'{args.train_folder}/crossing-image', f'{args.train_folder}/crossing-labels',
        f'{args.test_folder}/crossing-image', f'{args.test_folder}/crossing-labels', args.output_folder],

        ["endpoint", f'{args.train_folder}/endpoint-image', f'{args.train_folder}/endpoint-labels',
        f'{args.test_folder}/endpoint-image', f'{args.test_folder}/endpoint-labels', args.output_folder],

        ["fill", f'{args.train_folder}/fill-image', f'{args.train_folder}/fill-labels',
        f'{args.test_folder}/fill-image', f'{args.test_folder}/fill-labels', args.output_folder],

        ["skel-fill", f'{args.train_folder}/skel-fill-image', f'{args.train_folder}/skel-fill-labels',
        f'{args.test_folder}/skel-fill-image', f'{args.test_folder}/skel-fill-labels', args.output_folder],

        ["skel", f'{args.train_folder}/skel-image', f'{args.train_folder}/skel-labels',
        f'{args.test_folder}/skel-image', f'{args.test_folder}/skel-labels', args.output_folder],

        ["thresh", f'{args.train_folder}/thresh-image', f'{args.train_folder}/thresh-labels',
        f'{args.test_folder}/thresh-image', f'{args.test_folder}/thresh-labels', args.output_folder],

        ["line", f'{args.train_folder}/line-image', f'{args.train_folder}/line-labels',
        f'{args.test_folder}/line-image', f'{args.test_folder}/line-labels', args.output_folder],

        ["ellipse", f'{args.train_folder}/ellipse-image', f'{args.train_folder}/ellipse-labels',
        f'{args.test_folder}/ellipse-image', f'{args.test_folder}/ellipse-labels', args.output_folder],

        ["circle", f'{args.train_folder}/circle-image', f'{args.train_folder}/circle-labels',
        f'{args.test_folder}/circle-image', f'{args.test_folder}/circle-labels', args.output_folder],

        ["ellipse-circle", f'{args.train_folder}/ellipse_circle-image', f'{args.train_folder}/ellipse_circle-labels',
        f'{args.test_folder}/ellipse_circle-image', f'{args.test_folder}/ellipse_circle-labels', args.output_folder],

        ["chull", f'{args.train_folder}/chull-image', f'{args.train_folder}/chull-labels',
        f'{args.test_folder}/chull-image', f'{args.test_folder}/chull-labels', args.output_folder],

        ["corner", f'{args.train_folder}/corner-image', f'{args.train_folder}/corner-labels',
        f'{args.test_folder}/corner-image', f'{args.test_folder}/corner-labels', args.output_folder]
    ]

    start_time = datetime.now()
    for batch in batches:
        train_batch((batch))
    end_time = datetime.now()
    print("Completed training", len(batches), "transforms in:", end_time - start_time)

    # python train_transforms.py -r ./transforms/mnist -e ./transforms/mnist-test -o models/mnist-svc-rbf