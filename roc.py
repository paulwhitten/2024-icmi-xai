from sklearn.preprocessing import LabelBinarizer
from load_mnist_data import load_mnist_float
import json
import pickle

p = open("./kb_svm_mnist/corner.test.json")
pred = json.load(p)
p.close()
l = open("./kb_svm_mnist/corner.test-labels.json")
labels = json.load(l)
l.close()

print(pred)
#print(labels)

label_binarizer = LabelBinarizer().fit(labels)
y_onehot_test = label_binarizer.transform(pred)
print(y_onehot_test.shape)  # (n_samples, n_classes)

print(y_onehot_test)

import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay

class_id = 0
class_of_interest = "zero"

model = pickle.load(open("./models/mnist_svm/corner.model", 'rb'))
n, rows, columns, train_images, train_labels = load_mnist_float("/home/pcwhitte/mnist/transforms_test/corner-image", "/home/pcwhitte/mnist/transforms_test/corner-labels")
y_score = model.predict_proba(train_images)
print("Y_SCORE:", y_score)
print("Y_SCORE shape:", y_score.shape)

RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    y_score[:, class_id],
    name=f"{class_of_interest} vs the rest",
    color="darkorange"
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\n0 vs [1,9]")
plt.legend()
plt.show()