import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch as _Birch

from AbstractAlgo import AbstractCluster
from utils.param_utils import parse_params
from utils.const_utils import *


class BIRCH(AbstractCluster):
    """The BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) algorithm was designed specifically for
    very large datasets, and it can be faster than batch K-Means, with similar results, as long as the number of
    features is not too large (<20). During training, it builds a tree structure containing just enough information to
    quickly assign each new instance to a cluster, without having to store all the instances in the tree: this approach
    allows it to use limited memory, while handling huge datasets.
    """
    def __init__(self, options):
        self.ss_feature = StandardScaler()
        out_params = parse_params(
            options.get('algo_params', {}),
            ints=['n_clusters', 'branching_factor'],
            floats=['threshold']
        )

        self.estimator = _Birch(**out_params)

    def train(self, df, options):
        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]

        # 1. Standardlize the train and test data of features.
        ss_feature_train = self.ss_feature.fit_transform(feature_data)

        # 2. Train the model with KMeans
        self.estimator.fit(ss_feature_train)

        # 3. Evaluate the model performance
        y_labels = self.estimator.predict(ss_feature_train)
        metrics = self.evaluate(ss_feature_train, y_labels)
        sub_cluster_centers = list(self.estimator.subcluster_centers_)
        centers = {i: list(sub_cluster_centers[i]) for i in range(len(sub_cluster_centers))}
        fitted_parameter = {
            'num_sub_cluster': len(sub_cluster_centers),
            'cluster_centers': centers
        }
        metrics[FITTED_PARAMS] = fitted_parameter

        # 4. Handle the return value, store the model into cache
        output = pd.concat([df, pd.DataFrame(y_labels, columns=[CLUTER_NAME])], axis=1)

        return {MODEL_TYPE_SINGLE: self.estimator}, output, metrics

    def infer(self, df, options):
        model_file = options['model']
        birch_model = model_file[MODEL_TYPE_SINGLE]
        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]
        ss_feature_data = self.ss_feature.fit_transform(feature_data)
        y_pred = birch_model.predict(ss_feature_data)
        output = pd.concat([df, pd.DataFrame(y_pred, columns=[CLUTER_NAME])], axis=1)
        return output


if __name__ == '__main__':
    ''' This is used for algorithm level test, should be run at the same dir of this file. 
            python BIRCH.py
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
            'n_clusters': 11,
            'threshold': 0.4
        }
    }
    dbscan = BIRCH(options)
    model, output, metrics = dbscan.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    options.clear()
    options = {
        'feature_attrs': ['AVG(CPU Util)', 'AVG(Memory Util)', 'SUM(Sent Bytes64) (Byte)',
                          'SUM(Received Bytes64) (Byte)'],
        'id_attr': "Host Name",
        'model': model
    }
    infer_out = dbscan.infer(raw_data, options)
    print(infer_out)
    print(infer_out.groupby('Host Name')[CLUTER_NAME].apply(list).apply(np.unique))
    print(np.unique(infer_out[CLUTER_NAME], return_counts=True))