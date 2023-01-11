import pandas as pd
import numpy as np
import time
import pathlib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from model.threshold_guess import compute_thresholds, cut
from model.gosdt import GOSDT

# read the dataset
df = pd.read_csv("../experiments/datasets/compas.csv")
X, y = df.iloc[:,:-1], df.iloc[:,-1]
h = df.columns[:-1]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)
print("X train shape:{}, X test shape:{}".format(X_train.shape, X_test.shape))

# GBDT parameters for threshold and lower bound guesses
n_est = 40
max_depth = 1

# 1.guess thresholds
X_train = pd.DataFrame(X_train, columns=h)
X_test = pd.DataFrame(X_test, columns=h)
X_train_guessed, thresholds, header, threshold_guess_time = compute_thresholds(X_train.copy(), y_train, n_est, max_depth)
X_test_guessed = cut(X_test.copy(), thresholds)
X_test_guessed = X_test_guessed[header]
print("After guessing, X train shape:{}, X test shape:{}".format(X_train_guessed.shape, X_test_guessed.shape))
print("train set column names == test set column names: {}".format(list(X_train_guessed.columns)==list(X_test_guessed.columns)))



# 2.guess lower bound
import time
start_time = time.perf_counter()
clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
clf.fit(X_train_guessed, y_train.values.flatten())
warm_labels = clf.predict(X_train_guessed)

elapsed_time = time.perf_counter() - start_time

lb_time = elapsed_time

# save the labels as a tmp file and return the path to it.
labelsdir = pathlib.Path('/tmp/warm_lb_labels')
labelsdir.mkdir(exist_ok=True, parents=True)

labelpath = labelsdir / 'warm_label.tmp'
labelpath = str(labelpath)
pd.DataFrame(warm_labels).to_csv(labelpath, header="class_labels",index=None)



# 3.train GOSDT model
config = {
            "regularization": 0.001,
            "depth_budget": 6,
            "time_limit": 60,
            "warm_LB": True,
            "path_to_labels": labelpath,
            "similar_support": False,
        }

model = GOSDT(config)

model.fit(X_train_guessed, y_train)

print("evaluate the model, extracting tree and scores", flush=True)

# 4.get the results
train_acc = model.score(X_train_guessed, y_train)
test_acc = model.score(X_test_guessed, y_test)
n_leaves = model.leaves()
n_nodes = model.nodes()
time = model.utime

print("Model training time: {}".format(time))
print("Training accuracy: {}".format(train_acc))
print("Test accuracy: {}".format(test_acc))
print("# of leaves: {}".format(n_leaves))
print(model.tree)



