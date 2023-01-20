import pickle
import os
import sys
import warnings

import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering as _SpectralClustering

from AbstractAlgo import AbstractCluster
from utils.param_utils import parse_params
from utils.const_utils import *

class SpectralClustering_with_auto_turning(AbstractCluster):
    ''' This algorithm takes a similarity matrix between the instances and creates a low-dimensional embedding from it
    (i.e., it reduces its dimensionality), then it uses another clustering algorithm in this low-dimensional space
    (implementation uses K-Means.) Spectral clustering can capture complex cluster structures, and it can also be used
    to cut graphs (e.g., to identify clusters of friends on a social network). It does not scale well to large numbers
    of instances, and it does not behave well when the clusters have very different sizes.
    '''

    def __init__(self, options):
        self.ss_feature = StandardScaler()
        is_tune = options.get('is_tune', False)
        if not is_tune:
            out_params = parse_params(
                options.get('algo_params', {}),
                ints=['n_clusters', 'random_state'],
                strs=['affinity', 'assign_labels'],
                floats=['gamma', 'degree', 'coef0']
            )
            self.estimator = _SpectralClustering(**out_params)
        else:
            param_grid = {'n_clusters': range(2, 11),
                          'affinity': ['nearest_neighbors', 'rbf', 'poly'],
                          'assign_labels': ['kmeans', 'discretize'],
                          'n_neighbors': [5, 10, 15],
                          'gamma': [0.01, 0.1, 1],
                          'degree': [2, 3],
                          'coef0': [0, 1]
                          }

            def silhouette_score(estimator, X):
                estimator.fit(X)
                cluster_labels = estimator.labels_
                num_labels = len(set(cluster_labels))
                num_samples = len(X)
                if num_labels == 1 or num_labels == num_samples:
                    return -1
                else:
                    score = sklearn.metrics.silhouette_score(X, cluster_labels)
                return score

            self.estimator = GridSearchCV(_SpectralClustering(), param_grid, n_jobs=-1, cv=5, scoring=silhouette_score)

    def train(self, df, options):

        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]

        # 1. Standardlize the train and test data of features.
        ss_feature_train = self.ss_feature.fit_transform(feature_data)

        if not options.get('is_tune', False):
            # 2. Train the model with KMeans
            y_labels = self.estimator.fit_predict(ss_feature_train)
            # 3. Evaluate the model performance
            metrics = self.evaluate(ss_feature_train, y_labels)
            fitted_parameter = {
                'num_cluster': np.unique(y_labels).shape[0],
            }
        else:
            # 2. Train the model with KMeans
            self.estimator.fit(ss_feature_train)
            # 3. Evaluate the model performance
            y_labels = self.estimator.best_estimator_.labels_
            # 3. Evaluate the model performance
            metrics = self.evaluate(ss_feature_train, y_labels)
            fitted_parameter = {
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
    }
    algo = SpectralClustering_with_auto_turning(options)
    model, output, metrics = algo.train(raw_data, options)
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
    infer_out = algo.infer(raw_data, options)
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
    }
    algo = SpectralClustering_with_auto_turning(options)
    model, output, metrics = algo.train(raw_data, options)
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
    infer_out = algo.infer(raw_data, options)
    print(infer_out)
    print(infer_out.groupby(options.get('id_attr'))[CLUTER_NAME].apply(list).apply(np.unique))
    print(np.unique(infer_out[CLUTER_NAME], return_counts=True))


if __name__ == '__main__':
    ''' This is used for algorithm level test, should be run at the same dir of this file. 
            python KMeans.py
    '''
    # host_health_test(False)
    # host_health_test(True)
    # test_iris(False)
    test_iris(True)
