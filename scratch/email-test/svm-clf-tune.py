from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import dataImporter 
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pandas as pd
import numpy as np

# Get data using stratified split
data = dataImporter.dataImporter()
X_train, y_train = data.getTrainData()
ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=data.RANDOM_STATE)



# define the estimator base model, include standardisation when training
model_base = make_pipeline(preprocessing.StandardScaler(), svm.SVC()) #LogisticRegression()

# Set the parameters by cross-validation
tuning_parameters = [{'svc__kernel': ['rbf'], 'svc__gamma': [1e-3, 1e-4],
                    'svc__C': [1, 10, 100, 1000]},
                    {'svc__kernel': ['linear'], 'svc__C': [1, 10, 100]}]

#print(model_base.named_steps)
if __name__ == '__main__':
    # defined tuned model as an extension of base model and using grid search of tuning_parameters
    model_tuner = joblib.load('grid_search_svm_full.pkl') # GridSearchCV(model_base, tuning_parameters, cv=ten_fold_cv, n_jobs=-1, verbose=2)

    #model_tuner.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(model_tuner.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = model_tuner.cv_results_['mean_test_score']
    stds = model_tuner.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model_tuner.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()

    #joblib.dump(model_tuner, 'grid_search_svm_full.pkl', compress = 1)

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    X_test, y_test = data.getTestData()
    y_true, y_pred = y_test, model_tuner.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    results = model_tuner.cv_results_
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")
    plt.grid()
    ax = plt.axes()
    ax.set_ylim(0.73, 1)

    ax.set_ylim(0.73, 1)
    df = pd.DataFrame(results)
    print(df)
    #print(results.keys)
    pass
    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_svc__C'].data, dtype=float)[:-3]
    ax.set_xlim(np.min(X_axis), np.max(X_axis)+10**2)
    print(X_axis)
    # plt.plot(y_true) #label=name
    # plt.plot(y_pred) #label=name

    # plt.legend(loc="upper right")
    # plt.xlabel("Proportion train")
    # plt.ylabel("Test Error Rate")
    # plt.show()
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, "score")][:-3]
        sample_score_std = results['std_%s_%s' % (sample, "score")][:-3]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color='g')
        ax.semilogx(X_axis, sample_score_mean, style, color='g',
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % ("Accuracy", sample))

    best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
    best_score = results['mean_test_%s' % "score"][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
             linestyle='-.', color='g', marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                 (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.show()


# main()

# accuracy from 
