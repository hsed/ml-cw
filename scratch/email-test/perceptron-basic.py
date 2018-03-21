import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import dataImporter as dI
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

data = dI.dataImporter(shuffle=True, stratify=True)
X_train, y_train = data.getTrainData()

clf = make_pipeline(Perceptron(max_iter=5))
scores = cross_val_score(clf, X_train[:,:-3], y_train,cv=10)

print("Mean Accuracy: %.2f%% (+/- %.2f%%)" % (scores.mean()*100, scores.std()*100))