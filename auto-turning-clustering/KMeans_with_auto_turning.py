import codecs
import logging
import pickle
import os
import sys

import sklearn
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics._scorer import adjusted_rand_scorer, make_scorer

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
from sklearn.model_selection import GridSearchCV

logger = get_logger(__file__)


class KMeansWithAutoTurning(AbstractCluster):
    def __init__(self, options):
        # logger.info("Initializing KMeans Algo")
        self.ss_feature = StandardScaler()

        is_tune = options.get('is_tune', False)
        if not is_tune:
            input_params = parse_params(
                options.get('algo_params', {}),
                ints=['n_clusters', 'n_init', 'max_iter', 'random_state'],
                strs=['init', 'algorithm'],
                floats=['tol']
            )
            self.estimator = _KMeans(**input_params)
        else:
            param_grid = {
                          'n_clusters': range(2, 11),
                          'n_init': [10, 20, 30, 40, 50],
                          'max_iter': [100, 200, 300, 400, 500],
                          'init': ['k-means++', 'random'],
                          'tol': [1e-4, 1e-3, 1e-2]}

            # self.estimator = GridSearchCV(_KMeans(), param_grid)
            # self.estimator = GridSearchCV(_KMeans(), param_grid, cv=5)

            def silhouette_score(estimator, X):
                labels = estimator.fit_predict(X)
                score = sklearn.metrics.silhouette_score(X, labels, metric='euclidean')
                return score
            self.estimator = GridSearchCV(_KMeans(), param_grid, cv=5, scoring=silhouette_score)

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

        if not options.get('is_tune', False):
            cluster_centers = list(self.estimator.cluster_centers_)
            centers = {i: list(cluster_centers[i]) for i in range(len(cluster_centers))}
            fitted_parameter = {
                'num_cluster': len(cluster_centers),
                'cluster_centers': centers,
                '_intertia': self.estimator.inertia_
            }
        else:
            cluster_centers = list(self.estimator.best_estimator_.cluster_centers_)
            centers = {i: list(cluster_centers[i]) for i in range(len(cluster_centers))}
            fitted_parameter = {
                'num_cluster': len(cluster_centers),
                'cluster_centers': centers,
                '_intertia': self.estimator.best_estimator_.inertia_,
                'best_score_': self.estimator.best_score_,
                'best_params_': self.estimator.best_params_
            }

        metrics[FITTED_PARAMS] = fitted_parameter

        # 4. Handle the return value, store the model into cache
        output = pd.concat([df, pd.DataFrame(y_labels, columns=[CLUTER_NAME])], axis=1)

        if not options.get('is_tune', False):
            return {MODEL_TYPE_SINGLE: self.estimator}, output, metrics
        else:
            return {MODEL_TYPE_SINGLE: self.estimator.best_estimator_}, output, metrics

    def infer(self, df, options):
        model_file = options['model']
        kmeans_model = model_file[MODEL_TYPE_SINGLE]
        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]
        ss_feature_data = self.ss_feature.fit_transform(feature_data)
        y_pred = kmeans_model.predict(ss_feature_data)
        output = pd.concat([df, pd.DataFrame(y_pred, columns=[CLUTER_NAME])], axis=1)
        return output

def test_iris(is_tune):
    data = load_iris(as_frame=True)
    raw_data = data.data
    raw_data = pd.concat([raw_data, pd.DataFrame(data=data.target).rename(columns={'target': 'species'})],
                         axis=1).reset_index(drop=True)
    raw_data.fillna(0)
    options = {
        'feature_attrs': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
        'id_attr': 'species',
        'train_factor': 0.9,
        'is_tune': is_tune,
        'algo_params': {
            'n_clusters': 5
        }
    }
    kmeans = KMeansWithAutoTurning(options)
    model, output, metrics = kmeans.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))
    # output.to_csv(BASE_DIR + '/resources/output/kMeans/host_health_predict.csv', index=False)
    # with open(BASE_DIR + '/resources/output/kMeans/host_health_metrics.json', 'w') as metric_output:
    #     json.dump(metrics, metric_output)

    options.clear()
    options = {
        'feature_attrs': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
        'id_attr': 'species',
        'model': model
    }
    infer_out = kmeans.infer(raw_data, options)
    print(infer_out)
    print(infer_out.groupby(options.get('id_attr'))[CLUTER_NAME].apply(list).apply(np.unique))
    print(np.unique(infer_out[CLUTER_NAME], return_counts=True))
def host_health_test(is_tune):
    import json

    raw_data = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/Resources/host_health.csv')
    raw_data = raw_data.drop('Event Receive Hour', axis=1)

    options = {
        'feature_attrs': ['AVG(CPU Util)', 'AVG(Memory Util)', 'SUM(Sent Bytes64) (Byte)',
                          'SUM(Received Bytes64) (Byte)'],
        'id_attr': 'Host Name',
        'target_attr': '',
        'train_factor': 0.9,
        'is_tune': is_tune,
        'algo_params': {
            'n_clusters': 5
        }
    }
    kmeans = KMeansWithAutoTurning(options)
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


if __name__ == '__main__':
    ''' This is used for algorithm level test, should be run at the same dir of this file. 
            python KMeans.py
    '''
    host_health_test(False)
    host_health_test(True)
    # test_iris(False)
    # test_iris(True)
