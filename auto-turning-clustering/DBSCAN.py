import pickle
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN as _DBSCAN
from scipy.spatial.distance import euclidean

from AbstractAlgo import AbstractCluster
from utils.param_utils import parse_params
from utils.const_utils import *


class DBSCAN(AbstractCluster):
    ''' DBSCAN - Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high density
    and expands clusters from them. Good for data which contains clusters of similar density.
    '''

    def __init__(self, options):
        self.ss_feature = StandardScaler()
        out_params = parse_params(
            options.get('algo_params', {}),
            ints=['min_samples'],
            strs=['algorithm'],
            floats=['eps']
        )

        self.estimator = _DBSCAN(**out_params)

    def train(self, df, options):
        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]

        # 1. Standardlize the train and test data of features.
        ss_feature_train = self.ss_feature.fit_transform(feature_data)

        # 2. Train the model with DBSCAN
        y_labels = self.estimator.fit_predict(ss_feature_train)

        # 3. Evaluate the model performance
        metrics = self.evaluate(ss_feature_train, y_labels)
        cluster_set = set(filter(lambda label : label >= 0, y_labels.tolist()))
        fitted_parameter = {
            'num_cluster': len(cluster_set),
        }
        metrics[FITTED_PARAMS] = fitted_parameter

        # 4. Handle the return value, store the model into cache
        output = pd.concat([df, pd.DataFrame(y_labels, columns=[CLUTER_NAME])], axis=1)
        return {MODEL_TYPE_SINGLE: self.estimator}, output, metrics

    def infer(self, df, options):
        model_file = options['model']
        dbscan_model = model_file[MODEL_TYPE_SINGLE]
        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]
        ss_feature_data = self.ss_feature.fit_transform(feature_data)

        # Result is noise by default
        y_pred = np.ones(shape=len(ss_feature_data), dtype=int) * -1

        # Iterate all input samples for a label
        for j, x_new in enumerate(ss_feature_data):
            # Find a core sample closer than EPS
            for i, x_core in enumerate(dbscan_model.components_):
                if euclidean(x_new, x_core) < dbscan_model.eps:
                    # Assign label of x_core to x_new
                    y_pred[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                    break

        orig_id = options['id_attr'] if 'id_attr' in options and options['id_attr'] is not "" else None
        cluster_name = options['target_attr'] if 'target_attr' in options and options[
            'target_attr'] is not "" else CLUTER_NAME
        output = pd.concat([df[orig_id] if orig_id else None, feature_data,
                            pd.DataFrame(y_pred, columns=[cluster_name])], axis=1)
        return output


if __name__ == '__main__':
    ''' This is used for algorithm level test, should be run at the same dir of this file. 
                python DBSCAN.py
        '''
    import json

    raw_data = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/Resources/host_health.csv')
    raw_data = raw_data.drop('Event Receive Hour', axis=1)

    options = {
        'feature_attrs': ['AVG(CPU Util)', 'AVG(Memory Util)', 'SUM(Sent Bytes64) (Byte)',
                          'SUM(Received Bytes64) (Byte)'],
        'target_attr': '# cluster',
        'train_factor': 0.9,
        'id_attr': "Host Name",
        'algo_params': {
            'eps': 0.5,
            'min_samples': 10
        }
    }
    dbscan = DBSCAN(options)
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
    print(infer_out.groupby('Host Name')['cluster'].apply(list).apply(np.unique))
    print(np.unique(infer_out['cluster'], return_counts=True))

