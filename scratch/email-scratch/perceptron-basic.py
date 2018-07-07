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
from sklearn.model_selection import GridSearchCV
from multiprocessing import Process
from perceptronOption import simple_perceptron, tuned_perceptron, fine_tuned_perceptron
import sys


## global
nextOption = None

def menu(defaultVal=None, runAll=False, loadWeight=False):
    is_valid_input = False
    if runAll and not loadWeight:
        print("Warning: loadWeight will be set to True because runAll was requested.")
        loadWeight = True
    option_dict = {
        1: simple_perceptron,
        2: tuned_perceptron,
        3: fine_tuned_perceptron
    }
    print("\nPlease enter an option number [1-3]:",
              "\n1. Basic perceptron evaluation",
              "\n2. Perceptron evaluation with [1, 10, 100, 1000, 10000] epochs test",
              "\n3. Perceptron narrower evaluation with epochs test",
              "\n4. Run all tests using saved weights",)
    while is_valid_input != True:
        try:
            option = int(input()) if defaultVal is None else defaultVal
            if option == len(option_dict)+1:
                option = 1 # if it is the last option
                runAll = True
            if option not in option_dict: raise ValueError
            else: 
                p = Process(target=option_dict[option](loadWeight))
                p.start()
                p.join()
                if runAll == False or option == len(option_dict):
                    return 0
                else:
                    return menu(defaultVal=option+1,runAll=True,loadWeight=loadWeight)
        except ValueError:
            print("Please enter a valid integer within the given range.")
        #if()


# clf = make_pipeline(preprocessing.StandardScaler(),Perceptron())
# tuning_parameters = [{'perceptron__max_iter': [1, 10, 100, 1000]}]

if __name__ == '__main__': 
    if len(sys.argv) > 1 and (sys.argv[1] == '--load-weight' or sys.argv[1] == '-w'):
        print("LOAD MODE: All functions with weight load/store capability will load weights.")
        menu(loadWeight=True)
    else: menu()

