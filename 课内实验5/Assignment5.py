import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

X_train = pd.read_csv("breast_cancer_Xtrain.csv").to_numpy()
X_test = pd.read_csv("breast_cancer_Xtest.csv").to_numpy()
Y_train = pd.read_csv("breast_cancer_Ytrain.csv").to_numpy()
Y_test = pd.read_csv("breast_cancer_Ytest.csv").to_numpy()

model1 = svm.LinearSVC(C=1)
model2 = svm.LinearSVC(C=1000, dual=False)
model3 = svm.SVC(C=1, kernel='poly', degree=2)
model4 = svm.SVC(C=1000, kernel='poly', degree=2)

def train_and_pred(model):
    model.fit(X_train, Y_train.ravel())
    pred = model.predict(X_test)
    acc = accuracy_score(Y_test, pred)
    return acc

acc_1 = train_and_pred(model1)
acc_2 = train_and_pred(model2)
acc_3 = train_and_pred(model3)
acc_4 = train_and_pred(model4)

print("", acc_1)
print(acc_2)
print(acc_3)
print(acc_4)