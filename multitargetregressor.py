from random import random
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

class DtPCA:
    def __init__(self, random_state = 0, pca_component = 3, max_leaf_nodes=4):
        self.random_state = random_state
        self.pca_component = pca_component
        self.max_leaf_nodes= max_leaf_nodes
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.pca = PCA(n_components=self.pca_component)

    def fit(self, X, y):
        np.random.seed(0)
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        y_scaled_trans = self.pca.fit_transform(y_scaled)

        self.collected_models = []
        for i in range(y_scaled_trans.shape[1]):
            regressor = GradientBoostingRegressor(random_state=self.random_state, max_leaf_nodes=self.max_leaf_nodes)
            regressor.fit(X_scaled, y_scaled_trans[:, i])
            self.collected_models += [regressor]
    
    def predict(self, X):
        prediction = []
        for md in self.collected_models:
            prediction += [md.predict(X)]

        prediction = np.array(prediction).T
        prediction_inverse = self.scaler_y.inverse_transform(self.pca.inverse_transform(prediction))

        return prediction_inverse

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def evaluate(self, y_true, y_predict):
        return 1/(y_true.shape[1]) * np.sum(
            np.sqrt(
                np.sum((y_predict-y_true)**2) / np.sum((np.mean(y_true)-y_true)**2)
                )
            )