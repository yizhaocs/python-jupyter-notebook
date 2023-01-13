import codecs
import logging
import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as _KMeans
from sklearn.preprocessing import StandardScaler

from AbstractAlgo import AbstractCluster
from utils.param_utils import parse_params
from utils.const_utils import *

from utils.log_utils import get_logger
logger = get_logger(__file__)


class KMeans(AbstractCluster):
    def __init__(self, options):
        # logger.info("Initializing KMeans Algo")
        self.ss_feature = StandardScaler()
        input_params = parse_params(
            options.get('algo_params', {}),
            ints=['n_clusters', 'n_init', 'max_iter', 'random_state'],
            strs=['init', 'algorithm'],
            floats=['tol']
        )
        self.estimator = _KMeans(**input_params)

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
        kmeans_model = model_file[MODEL_TYPE_SINGLE]
        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]
        ss_feature_data = self.ss_feature.fit_transform(feature_data)
        y_pred = kmeans_model.predict(ss_feature_data)
        output = pd.concat([df, pd.DataFrame(y_pred, columns=[CLUTER_NAME])], axis=1)
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
        'id_attr': 'Host Name',
        'target_attr': '',
        'train_factor': 0.9,
        'algo_params': {
            'n_clusters': 5
        }
    }
    kmeans = KMeans(options)
    model, output, metrics = kmeans.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))
    # output.to_csv(BASE_DIR + '/resources/output/kMeans/host_health_predict.csv', index=False)
    # with open(BASE_DIR + '/resources/output/kMeans/host_health_metrics.json', 'w') as metric_output:
    #     json.dump(metrics, metric_output)

    options.clear()
    options = {
        'feature_attrs': ['AVG(CPU Util)', 'AVG(Memory Util)', 'SUM(Sent Bytes64) (Byte)',
                          'SUM(Received Bytes64) (Byte)'],
        'id_attr': "Host Name",
        'model': model
    }
    infer_out = kmeans.infer(raw_data, options)
    print(infer_out)
    print(infer_out.groupby(options.get('id_attr'))[CLUTER_NAME].apply(list).apply(np.unique))
    print(np.unique(infer_out[CLUTER_NAME], return_counts=True))
