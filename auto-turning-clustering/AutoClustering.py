import os
import sys

import sklearn
from sklearn.datasets import load_iris

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from AbstractAlgo import AbstractCluster
from utils.const_utils import *
from utils.log_utils import get_logger
from BIRCH_with_auto_turning import BIRCH_with_auto_turning
from DBSCAN_with_auto_turning import DBSCAN_with_auto_turning
from KMeans_with_auto_turning import KMeans_with_auto_turning
from SpectralClustering_with_auto_turning import SpectralClustering_with_auto_turning
import warnings

# filter out the warnings
warnings.simplefilter("ignore")
logger = get_logger(__file__)


class AutoClustering(AbstractCluster):

    def __init__(self, options):
        self.ss_feature = StandardScaler()
        options.update({'is_tune': True})
        self.estimator_birch = BIRCH_with_auto_turning(options)
        self.estimator_dbscan = DBSCAN_with_auto_turning(options)
        self.estimator_kmeans = KMeans_with_auto_turning(options)
        self.estimator_spectral_clustering = SpectralClustering_with_auto_turning(options)

    def train(self, df, options):
        scores = []
        model_birch, output_birch, metrics_birch = self.estimator_birch.train(df, options)
        scoring = options['algo_params']['scoring']
        if FITTED_ERRORS in metrics_birch:
            if scoring == 'silhouette_score':
                score_birch = metrics_birch[FITTED_ERRORS]['Silhouette Score']
            elif scoring == 'calinski_harabasz_score':
                score_birch = metrics_birch[FITTED_ERRORS]['Calinski Marabasz Score']
            elif scoring == 'davies_bouldin_score':
                score_birch = metrics_birch[FITTED_ERRORS]['Davies Bouldin Score']
            if score_birch:
                scores.append(score_birch)
                print(f'score_birch:{score_birch}')

        model_dbscan, output_dbscan, metrics_dbscan = self.estimator_dbscan.train(df, options)

        if FITTED_ERRORS in metrics_dbscan:
            if scoring == 'silhouette_score':
                score_dbscan = metrics_dbscan[FITTED_ERRORS]['Silhouette Score']
            elif scoring == 'calinski_harabasz_score':
                score_dbscan = metrics_dbscan[FITTED_ERRORS]['Calinski Marabasz Score']
            elif scoring == 'davies_bouldin_score':
                score_dbscan = metrics_dbscan[FITTED_ERRORS]['Davies Bouldin Score']
            if score_dbscan:
                scores.append(score_dbscan)
                print(f'score_dbscan:{score_dbscan}')

        model_kmeans, output_kmeans, metrics_kmeans = self.estimator_kmeans.train(df, options)
        if FITTED_ERRORS in metrics_kmeans:
            if scoring == 'silhouette_score':
                score_kmeans = metrics_kmeans[FITTED_ERRORS]['Silhouette Score']
            elif scoring == 'calinski_harabasz_score':
                score_kmeans = metrics_kmeans[FITTED_ERRORS]['Calinski Marabasz Score']
            elif scoring == 'davies_bouldin_score':
                score_kmeans = metrics_kmeans[FITTED_ERRORS]['Davies Bouldin Score']
            if score_kmeans:
                scores.append(score_kmeans)
                print(f'score_kmeans:{score_kmeans}')

        model_spectral_clustering, output_spectral_clustering, metrics_spectral_clustering = self.estimator_spectral_clustering.train(df, options)

        if FITTED_ERRORS in metrics_spectral_clustering:
            if scoring == 'silhouette_score':
                score_spectral_clustering = metrics_spectral_clustering[FITTED_ERRORS]['Silhouette Score']
            elif scoring == 'calinski_harabasz_score':
                score_spectral_clustering = metrics_spectral_clustering[FITTED_ERRORS]['Calinski Marabasz Score']
            elif scoring == 'davies_bouldin_score':
                score_spectral_clustering = metrics_spectral_clustering[FITTED_ERRORS]['Davies Bouldin Score']
            if score_spectral_clustering:
                scores.append(score_spectral_clustering)
                print(f'score_spectral_clustering:{score_spectral_clustering}')
        max_score = max(scores)

        if max_score == score_birch:
            return model_birch, output_birch, metrics_birch
        elif max_score == score_dbscan:
            return model_dbscan, output_dbscan, metrics_dbscan
        elif max_score == score_kmeans:
            return model_kmeans, output_kmeans, metrics_kmeans
        elif max_score == score_spectral_clustering:
            return model_spectral_clustering, output_spectral_clustering, metrics_spectral_clustering

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
            'scoring': 'davies_bouldin_score'
        }
    }
    algo = AutoClustering(options)
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
        'algo_params': {
            'scoring': 'silhouette_score'
        }
    }
    algo = AutoClustering(options)
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
    # host_health_test(True)
    test_iris(True)
