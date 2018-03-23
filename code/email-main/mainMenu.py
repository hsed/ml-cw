# import numpy as np
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.naive_bayes import MultinomialNB
# import dataImporter as dI
# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import make_pipeline
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Perceptron
# from sklearn.model_selection import GridSearchCV
from multiprocessing import Process
from perceptronModels import simple_perceptron, tuned_perceptron, fine_tuned_perceptron
from logisticModels import simple_logisticreg, tuned_logisticreg_l1_l2, tuned_logisticreg_multi_param
from functools import partial
import sys


## global
nextOption = None


def menu(loadWeight=False, defaultVal=None, runAll=False):
    
    option_dict = {
        1: simple_perceptron,
        2: tuned_perceptron,
        3: fine_tuned_perceptron,
        4: simple_logisticreg,
        5: tuned_logisticreg_l1_l2,
        6: tuned_logisticreg_multi_param,
        7: partial(menu,defaultVal=1,runAll=True),
    }


    def dispatchProcess(option=1, loadW=False):
        p = Process(target=option_dict[option](loadW))
        p.start()
        p.join()


    is_valid_input = False
    if runAll and not loadWeight:
        print("Warning: loadWeight will be set to True because runAll was requested.")
        loadWeight = True
    print("\nPlease enter an option number:",
              "\n1. Basic perceptron evaluation",
              "\n2. Perceptron evaluation with [1, 10, 100, 1000, 10000] epochs test",
              "\n3. Perceptron narrower evaluation with epochs test",
              "\n4. Basic logistic regression evaluation",
              "\n5. Tuned logistic regression l1 and l2 evaluation",
              "\n6. Tuned logistic regression multiple hyperParam",
              "\n7. ",
              "\nN. Run all tests using saved weights")
    while is_valid_input != True:
        try:
            option = int(input()) if defaultVal is None else defaultVal #get from input or arg
            if option not in option_dict.keys(): raise ValueError
        except ValueError:
            print("Please enter a valid integer within the given range.") # except matches any error even if its not a value error?
            print("Available options are:", list(option_dict.keys()))
            continue
        print("Option selected:", option)
        
        #start the process
        dispatchProcess(option, loadWeight)
        
        #base case, curr option must not be second last option in dict because new option will then be reloop
        if runAll == True and option != list(option_dict.keys())[-2]:
            # recursive call
            return menu(loadWeight=loadWeight, defaultVal=option+1, runAll=True)
        else:
            #reached the end or this was the last option so exit or runAll was set to false
            return 0
        
        #if()


# clf = make_pipeline(preprocessing.StandardScaler(),Perceptron())
# tuning_parameters = [{'perceptron__max_iter': [1, 10, 100, 1000]}]

if __name__ == '__main__': 
    if len(sys.argv) > 1 and (sys.argv[1] == '--load-weight' or sys.argv[1] == '-w'):
        print("LOAD MODE: All functions with weight load/store capability will load weights.")
        menu(loadWeight=True)
    else: menu()

