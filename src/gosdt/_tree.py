# Python standard library imports
from typing import List
import numpy as np

# External imports
from sklearn.base import check_array


class Leaf:

    def __init__(self, prediction: int, loss: float):
        self.prediction = prediction
        self.loss = loss

    def __str__(self) -> str:
        return "{ prediction: " + str(self.prediction) + ", loss: " + str(self.loss) + " }"


class Node:

    def __init__(self, feature: int, orig_feature: int, left_child, right_child):
        self.feature = feature
        self.orig_feature = orig_feature
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self) -> str:
        return "{ feature: " + str(self.feature) + ", orig feature: " +str(self.orig_feature) + ", [ left child: " + str(self.left_child) + ", right child: " + str(self.right_child) + "] }"

class Tree:
    # Left is TRUE, Right is FALSE

    def __init__(self, gosdt_result, features: List[str], n_classes: int, classes: np.ndarray):
        json_result = gosdt_result

        # Recursive tree creation.
        def create_tree(json_object):
            # Identify Leaf:
            if "prediction" in json_object:
                return Leaf(json_object["prediction"], json_object["loss"])

            # This is a node:
            left_child = create_tree(json_object["true"])
            right_child = create_tree(json_object["false"])
            feature = json_object["feature"]
            orig_feature = json_object["orig_feature"]
            return Node(feature, orig_feature, left_child, right_child)

        self.tree = create_tree(json_result)
        self.features = features
        self.n_classes = n_classes
        self.classes = classes

    def predict(self, X):
        # Validate X
        X = check_array(X, ensure_2d=True, dtype=bool)
        n, m = X.shape

        # Ensure that X has the correct number of features
        if m != len(self.features):
            raise ValueError(
                f"X must have the same number of features as the training data ({len(self.features)}), but has {m} features."
            )

        # Predict for a sample row in the dataset
        def predict_sample(x_i, node):
            if isinstance(node, Leaf):
                return self.classes[node.prediction]
            elif x_i[node.feature]:
                return predict_sample(x_i, node.left_child)
            else:
                return predict_sample(x_i, node.right_child)

        return np.array([predict_sample(X[i, :], self.tree) for i in range(n)])

    def predict_proba(self, X):
        # Input validation performed by predict function.
        y_1d = self.predict(X)

        # Create a probability matrix
        y_proba = np.zeros((len(y_1d), self.n_classes), dtype=float)
        for i in range(len(y_1d)):
            y_proba[i, y_1d[i]] = 1.0

        return y_proba

    def __str__(self) -> str:
        return str(self.__class__) + ": " + str(self.tree) + ", " + str(self.features) + ", " + str(self.n_classes)
