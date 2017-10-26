import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, data_path):
        self.orig_data = pd.read_csv(data_path)
        self.data = self.orig_data
        self.scaled_features = {}
        self.train_features = None
        self.train_targets = None
        self.test_features = None
        self.test_targets = None
        self.test_data = None
        self.val_features = None
        self.val_targets = None

    def show_data(self, plot_by_dteday=False):
        print (self.data.head())
        if plot_by_dteday == True:
            self.data[:24*10].plot(x='dteday', y='cnt', title='Data for the first 10 days')
            plt.show()

    def virtualize(self):
        # Add virtualized data columns
        dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
        for each in dummy_fields:
            dummies = pd.get_dummies(self.data[each], prefix=each, drop_first=False)
            self.data = pd.concat([self.data, dummies], axis=1)

        # Drop scale data columns
        fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                          'weekday', 'atemp', 'mnth', 'workingday', 'hr']
        self.data = self.data.drop(fields_to_drop, axis=1)

    def normalize(self):
        quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
        for each in quant_features:
            mean, std = self.data[each].mean(), self.data[each].std()
            self.scaled_features[each] = [mean, std]
            self.data.loc[:, each] = (self.data[each] - mean) / std

    def split(self):
        # Save data of last 21 days for testing
        self.test_data = self.data[-21 * 24:]
        self.data = self.data[:-21 * 24]

        target_fields = ['cnt', 'casual', 'registered']
        features, targets = self.data.drop(target_fields, axis=1), self.data[target_fields]
        self.test_features, self.test_targets = self.test_data.drop(target_fields, axis=1), self.test_data[target_fields]
        self.train_features, self.train_targets = features[:-60*24], targets[:-60*24]
        self.val_features, self.val_targets = features[-60*24:], targets[-60*24:]

    def get_train_data(self):
        return self.train_features, self.train_targets

    def get_test_data(self):
        return self.test_features, self.test_targets, self.test_data

    def get_val_data(self):
        return self.val_features, self.val_targets

    def get_scaled_features(self):
        return self.scaled_features

    def get_orig_data(self):
        return self.orig_data
