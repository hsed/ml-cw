import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

data = pd.DataFrame(pd.read_csv('spambase.data', header=None, sep=','))
np_data = data.as_matrix()
X_all = np_data[:, :-1]
y_all = np_data[:, -1].astype(int)

print("Data Shape: " + str(np_data.shape))

X_train, X_test, y_train, y_test = \
                train_test_split(X_all, y_all, test_size=0.3, random_state=RandomState(42))

# Plot the PCA spectrum
pca.fit(X_train, y_train)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# Prediction
n_components = [20, 40, 57]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(X_train, y_train)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()