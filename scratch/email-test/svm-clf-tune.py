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


# Get data using stratified split
data = dataImporter.dataImporter()
X_train, y_train = data.getTrainData()
ten_fold_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=data.RANDOM_STATE)



# define the estimator base model, include standardisation when training
model_base = make_pipeline(preprocessing.StandardScaler(), svm.SVC()) #LogisticRegression()

# Set the parameters by cross-validation
tuning_parameters = [{'svc__kernel': ['rbf'], 'svc__gamma': [1e-3, 1e-4],
                    'svc__C': [1, 10, 100, 1000]},
                    {'svc__kernel': ['linear'], 'svc__C': [1, 10, 100, 1000]}]

#print(model_base.named_steps)
if __name__ == '__main__':
    # defined tuned model as an extension of base model and using grid search of tuning_parameters
    model_tuned =  GridSearchCV(model_base, tuning_parameters, cv=ten_fold_cv, n_jobs=-1, verbose=2)

    model_tuned.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(model_tuned.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = model_tuned.cv_results_['mean_test_score']
    stds = model_tuned.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model_tuned.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    X_test, y_test = data.getTestData()
    y_true, y_pred = y_test, model_tuned.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    # plt.plot(y_true) #label=name
    # plt.plot(y_pred) #label=name

    # plt.legend(loc="upper right")
    # plt.xlabel("Proportion train")
    # plt.ylabel("Test Error Rate")
    # plt.show()


# main()

# accuracy from 
