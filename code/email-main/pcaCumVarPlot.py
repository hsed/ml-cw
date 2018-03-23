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

data = dI.dataImporter(shuffle=True, stratify=True)
X_train, y_train = data.getTrainData()
#X_train = X_train[:,:-3]
# define 10-fold cross validation test harness
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
#multiNB = make_pipeline(MultinomialNB())

#multinomial needs correct probability!!!

clf = make_pipeline(MultinomialNB())
clf2 = make_pipeline(preprocessing.StandardScaler(), BernoulliNB())
clf3 = make_pipeline(preprocessing.StandardScaler(), GaussianNB())
scores = cross_val_score(clf, X_train[:,:-3], y_train,cv=10)
scores2 = cross_val_score(clf2, X_train, y_train,cv=10)
scores3 = cross_val_score(clf2, X_train, y_train,cv=10)

print("Mean ROC_AUC Multi: %.2f%% (+/- %.2f%%)" % (scores.mean()*100, scores.std()*100))
print("Mean ROC_AUC Bernu: %.2f%% (+/- %.2f%%)" % (scores2.mean()*100, scores2.std()*100))
print("Mean ROC_AUC Gauss: %.2f%% (+/- %.2f%%)" % (scores3.mean()*100, scores3.std()*100))


# from sklearn.metrics import roc_curve
# X_test, y_test = data.getTestData()
# probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
# probas_2 = clf2.fit(X_train, y_train).predict_proba(X_test)
# fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
# fpr2, tpr2, thresholds = roc_curve(y_test, probas_2[:, 1])
#plt.plot(fpr, tpr)
#plt.plot(fpr2, tpr2)

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
pca_p = Pipeline([ ('pca', PCA())])
#show the importance of standardisation using pca
#without standardisation 3 feature vector accounts for 99% variance
#with std, all feature vectors are equivally imp due to the smooth curve
#hence generally dim reduction will not be attempted due to equal affect on the classification from each feature.
#pca_p = Pipeline([('pca', PCA())])

pca_p.fit(X_train)
plt.plot( pca_p.named_steps['pca'].explained_variance_ratio_.cumsum(), '--o')

plt.show()