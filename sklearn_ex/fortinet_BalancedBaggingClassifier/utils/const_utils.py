import json
from enum import Enum

DECIMAL_PRECISION = 2

FITTED_ERRORS = 'errors'
FITTED_CONFUSION = 'confusion'
FITTED_PARAMS = 'fitted_parameter'

PRIDCT_NAME = 'predicted'
CLUTER_NAME = 'clusterId'
ANOMALY_NAME = 'isAnomaly'
ANOMALY_TITLE = 'Anomalies Found'
NOMALY_TITLE = 'Normal Results'
CLUSTERS_TITLE = 'Clusters Found'
TOTAL = 'Total'
DIFF_NAME = 'error'

MODEL_TYPE_SINGLE = 'single'
MODEL_TYPE_MULTI = 'multi'

APPSERVER_TASK_ID_NAME = 'xmlId'
APPSERVER_TASK_TYPE_NAME = 'type'
APPSERVER_TASK_TYPE_TRAIN = 'TrainMachineLearningModel'
APPSERVER_TASK_TYPE_INFER = 'InferMachineLearning'
APPSERVER_TASK_ERROR_PREFIX = 'Error:'

REDIS_CACHE_TASK_NAME = 'parameterString'
REDIS_CACHE_TASK_INPUT = 'input'
REDIS_CACHE_TASK_ALGO = 'algorithm'
REDIS_CACHE_TASK_OUTPUT = 'output'
REDIS_CACHE_TASK_RESULT = 'result'
REDIS_CACHE_TASK_MODEL = 'model'
REDIS_CACHE_TASK_METRICS = 'metrics'
REDIS_CACHE_TASK_CLUSTER_THRESHOLD = 'threshold'

TASK_TYPE_TRAIN = 'train'
TASK_TYPE_INFER = 'infer'


class RunMode(Enum):
    LOCAL = 'Local'
    AWS = 'AWS'
    AWS_AUTO = 'AWS Auto'

class Category(Enum):
    REGRESSION = 'Regression'
    CLASSIFICATION = 'Classification'
    CLUSTERING = 'Clustering'
    FORECASTING = 'Forecasting'
    ANOMALY_DETECTION = 'Anomaly Detection'

    # For AWS Run Mode
    BINARY_CLASSIFICATION = 'BinaryClassification'
    MULTI_CLASSIFICATION = 'MulticlassClassification'

if __name__ == '__main__':
    pass
