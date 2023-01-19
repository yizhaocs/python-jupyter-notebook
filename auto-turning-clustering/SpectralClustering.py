import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering as _SpectralClustering

from AbstractAlgo import AbstractCluster
from utils.param_utils import parse_params
from utils.const_utils import *


class SpectralClustering(AbstractCluster):
    ''' This algorithm takes a similarity matrix between the instances and creates a low-dimensional embedding from it
    (i.e., it reduces its dimensionality), then it uses another clustering algorithm in this low-dimensional space
    (implementation uses K-Means.) Spectral clustering can capture complex cluster structures, and it can also be used
    to cut graphs (e.g., to identify clusters of friends on a social network). It does not scale well to large numbers
    of instances, and it does not behave well when the clusters have very different sizes.
    '''

    def __init__(self, options):
        self.ss_feature = StandardScaler()
        out_params = parse_params(
            options.get('algo_params', {}),
            ints=['n_clusters', 'random_state'],
            strs=['affinity', 'assign_labels'],
            floats=['gamma', 'degree', 'coef0']
        )
        self.estimator = _SpectralClustering(**out_params)

    def train(self, df, options):

        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]

        # 1. Standardlize the train and test data of features.
        ss_feature_train = self.ss_feature.fit_transform(feature_data)

        # 2. Train the model with KMeans
        y_labels = self.estimator.fit_predict(ss_feature_train)

        # 3. Evaluate the model performance
        metrics = self.evaluate(ss_feature_train, y_labels)
        fitted_parameter = {
            'num_cluster': np.unique(y_labels).shape[0],
        }
        metrics[FITTED_PARAMS] = fitted_parameter

        # 4. Handle the return value, store the model into cache
        output = pd.concat([df, pd.DataFrame(y_labels, columns=[CLUTER_NAME])], axis=1)

        return {MODEL_TYPE_SINGLE: self.estimator}, output, metrics

    def infer(self, df, options):
        # This is only a temporary way for the inference.
        # Not sure whether this algo support to predict new data based on training data.
        model_file = options['model']
        spectral_cluster_model = model_file[MODEL_TYPE_SINGLE]
        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]
        ss_feature_data = self.ss_feature.fit_transform(feature_data)
        y_pred = spectral_cluster_model.fit_predict(ss_feature_data)
        output = pd.concat([df, pd.DataFrame(y_pred, columns=['cluster'])], axis=1)
        return output


if __name__ == '__main__':
    ''' This is used for algorithm level test, should be run at the same dir of this file. 
                python KMeans.py
        '''
    import json

    raw_data = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/Resources/host_health.csv')
    raw_data = raw_data.drop('Event Receive Hour', axis=1)

    options = {
        'feature_attrs': ['AVG(CPU Util)', 'AVG(Memory Util)', 'SUM(Sent Bytes64) (Byte)',
                          'SUM(Received Bytes64) (Byte)'],
        'target_attr': '',
        'train_factor': 0.9,
        'id_attr': "Host Name",
        'algo_params': {
            'n_clusters': 5
        }
    }
    spectral_cluster = SpectralClustering(options)
    model, output, metrics = spectral_cluster.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    options.clear()
    options = {
        'feature_attrs': ['AVG(CPU Util)', 'AVG(Memory Util)', 'SUM(Sent Bytes64) (Byte)',
                          'SUM(Received Bytes64) (Byte)'],
        'id_attr': "Host Name",
        'model': model
    }
    infer_out = spectral_cluster.infer(raw_data, options)
    print(infer_out)
    print(infer_out.groupby(options.get('id_attr'))[CLUTER_NAME].apply(list).apply(np.unique))
    print(np.unique(infer_out[CLUTER_NAME], return_counts=True))


