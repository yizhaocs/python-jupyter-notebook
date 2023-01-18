import os
import pickle
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans as _KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from AbstractAlgo import AbstractCluster
from utils.param_utils import parse_params
from utils.const_utils import *


class GMeans(AbstractCluster):

    def __init__(self, options):
        self.kmax_ = 200
        self.random_state = 42
        self.inertia_ = 0
        self.labels_ = []
        self.cluster_centers_ = []
        self.n_clusters = 0
        self.input_params = parse_params(
            options.get('algo_params', {}),
            ints=['n_init', 'max_iter'],
            strs=['init', 'algorithm'],
            floats=['tol']
        )
        self.estimator = _KMeans(n_clusters=min(2, self.kmax_), **self.input_params)
        self.ss_feature = MinMaxScaler()

    def _gen_model(self, X, centers):
        num_centers = len(centers)
        if num_centers == 0:
            kmeans = _KMeans(n_clusters=min(2, self.kmax_), **self.input_params).fit(X)
        else:
            kmeans = _KMeans(n_clusters=num_centers, init=centers, n_init=1, random_state=self.random_state).fit(X)

        return kmeans

    def _adjust_cluster(self, X, labels, centers):
        new_centers = []
        num_additional_clusters = 0
        max_additional_clusters = self.kmax_ - len(centers)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        for cluster_label in unique_labels[np.argsort(-counts)]:
            # Check all clusters from big to small, then decide whether to split
            Y = np.array([X[i] for i in range(len(X)) if labels[i] == cluster_label])
            if len(Y) <= 2:
                new_centers.append(centers[cluster_label])
                continue

            # run k-means for cluster Y
            kmeans = self._gen_model(Y, [])
            if np.array_equal(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1]):
                new_centers.append(centers[cluster_label])
                continue

            # decide to split or not
            split = self._need_split(Y, kmeans.cluster_centers_)
            if split and num_additional_clusters < max_additional_clusters:
                new_centers.extend(kmeans.cluster_centers_)
                num_additional_clusters += 1
            else:
                new_centers.append(centers[cluster_label])

        return np.array(new_centers)

    def _need_split(self, Y, centers_Y, significance_level=1):
        v = np.subtract(centers_Y[0], centers_Y[1])
        normv = np.dot(v, v)
        Y1 = np.divide(np.sum(np.multiply(Y, v), axis=1), normv)
        anderson_statistic, critical_values, significance_levels = stats.anderson(Y1, dist='norm')
        return anderson_statistic >= critical_values[-significance_level]

    def train(self, df, options):
        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]

        # 1. Standardlize the train and test data of features.
        ss_feature_train = self.ss_feature.fit_transform(feature_data)

        # 2. Train the model with GMeans
        if self.kmax_ > 1:
            while len(self.cluster_centers_) < self.kmax_:
                # 2.1 Start the training with the minimal cluster number.
                current_num_centers = len(self.cluster_centers_)
                print("current_num_centers: ", current_num_centers)
                kmeans = self._gen_model(ss_feature_train, self.cluster_centers_)
                # 2.2 Check whether the generated cluster needs to be splitted and generate new cluster centers.
                new_centers = self._adjust_cluster(ss_feature_train, kmeans.labels_, kmeans.cluster_centers_)

                if current_num_centers == len(new_centers):
                    break
                else:
                    self.cluster_centers_ = new_centers

        self.estimator = self._gen_model(ss_feature_train, self.cluster_centers_)
        self.n_clusters = len(self.cluster_centers_)

        # 3. Evaluate the model performance
        y_labels = self.estimator.predict(ss_feature_train)
        metrics = self.evaluate(ss_feature_train, y_labels)
        cluster_centers = list(self.estimator.cluster_centers_)
        centers = {i: list(cluster_centers[i]) for i in range(len(cluster_centers))}
        fitted_parameter = {
            'num_cluster': len(cluster_centers),
            'cluster_centers': centers,
            '_intertia': self.estimator.inertia_
        }
        metrics[FITTED_PARAMS] = fitted_parameter

        # 4. Handle the return value, store the model into cache
        output = pd.concat([df, pd.DataFrame(y_labels, columns=[CLUTER_NAME])], axis=1)

        return {MODEL_TYPE_SINGLE: self.estimator}, output, metrics

    def infer(self, df, options):
        model_file = options['model']
        gmeans_model = model_file[MODEL_TYPE_SINGLE]
        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]
        ss_feature_data = self.ss_feature.fit_transform(feature_data)
        y_pred = gmeans_model.predict(ss_feature_data)
        output = pd.concat([df, pd.DataFrame(y_pred, columns=[CLUTER_NAME])], axis=1)
        return output


if __name__ == '__main__':
    ''' This is used for algorithm level test, should be run at the same dir of this file. 
            python GMeans.py
    '''
    import json

    raw_data = pd.read_csv(BASE_DIR + '/resources/data/host_health.csv')
    raw_data = raw_data.drop('Event Receive Hour', axis=1)

    options = {
        'feature_attrs': ['AVG(CPU Util)', 'AVG(Memory Util)', 'SUM(Sent Bytes64) (Byte)',
                          'SUM(Received Bytes64) (Byte)'],
        'target_attr': '# cluster',
        'train_factor': 0.7,
        'id_attr': "Host Name"
    }
    gmeans = GMeans(options)
    model, output, metrics = gmeans.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    options.clear()
    options = {
        'feature_attrs': ['AVG(CPU Util)', 'AVG(Memory Util)', 'SUM(Sent Bytes64) (Byte)',
                          'SUM(Received Bytes64) (Byte)'],
        'id_attr': "Host Name",
        'model': model
    }
    infer_out = gmeans.infer(raw_data, options)
    print(infer_out)
    print(infer_out.groupby(options.get('id_attr'))[CLUTER_NAME].apply(list).apply(np.unique))
    print(np.unique(infer_out[CLUTER_NAME], return_counts=True))
