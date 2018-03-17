# module data-importer
import numpy as np
import pandas as pd
from numpy.random import RandomState

class dataImporter(object):
    def __init__(self, filePath='spambase.data', shuffle=False, splitRatio=0.7, normalise=False, sep=','):
        self.filePath = filePath
        data = pd.DataFrame(pd.read_csv(filePath, header=None, sep=sep))
        self.np_data_array = data.as_matrix()

        np.random.seed(41)

        if shuffle: np.random.shuffle(self.np_data_array) # this will be done by k-fold eval!
        #features_matx = np_data_array[1,:]

        total_records = self.np_data_array.shape[0]
        
        train_rec = int(splitRatio*total_records)  # approx 70%

        self.X_all = self.np_data_array[:, :-1]
        self.y_all = self.np_data_array[:, -1].astype(int)

        if normalise:
            for j in range(0, self.X_all.shape[1]):
                mean = np.mean(self.X_all[:, j])
                std = np.std(self.X_all[:, j])

            for i in range(0, self.X_all.shape[0]):
                self.X_all[i, j] = (self.X_all[i, j] - mean)/std
            self.isDataNormalised = True
        else:
            self.isDataNormalised = False

        self.X_train = self.np_data_array[:train_rec,:-1]
        self.y_train = self.np_data_array[:train_rec,-1].astype(int)

        self.X_test = self.np_data_array[train_rec:,:-1]
        self.y_test = self.np_data_array[train_rec:,-1].astype(int)


        # import as default

    def getAllData(self):
        return (self.X_all, self.y_all)

    def getTrainData(self):
        return (self.X_train, self.y_train)
    
    def getTestData(self):
        return (self.X_test, self.y_test)