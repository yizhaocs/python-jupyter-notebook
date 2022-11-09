import copy
from datetime import datetime

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier as _BalancedBaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn_ex.fortinet_BalancedBaggingClassifier.AbstractAlgo import AbstractClassifier
from sklearn_ex.fortinet_BalancedBaggingClassifier.utils.parsing_utils import data_praser
from sklearn_ex.utils.const_utils import MODEL_TYPE_SINGLE
from sklearn_ex.utils.param_utils import parse_params


class BalancedBaggingClassifier(AbstractClassifier):

    def __init__(self, options):
        self.ss_feature = StandardScaler()
        encoder = OrdinalEncoder(encoded_missing_value=-1)
        super(BalancedBaggingClassifier, self).__init__()
        input_params = parse_params(
            options.get('algo_params', {}),
            ints=['n_estimators', 'verbose'],
            strs=['sampling_strategy'],
            floats=['max_samples', 'max_features'],
            bools=['bootstrap', 'bootstrap_features', 'oob_score', 'warm_start', 'replacement']
        )
        self.estimator = {
            "algorithm": _BalancedBaggingClassifier(**input_params),
            "encoder": encoder
        }

    def train(self, df, options):
        df = data_praser(df, options)
        feature_attrs = options['feature_attrs']
        target_attr = options['target_attr']
        numeric_feature_attrs = []
        categorical_feature_attrs = []
        for attr in feature_attrs:
            e = df[attr][[0]].to_numpy()[0]
            if e is not np.nan and (isinstance(e, np.integer) or isinstance(e, float)):
                numeric_feature_attrs.append(attr)
            else:
                categorical_feature_attrs.append(attr)

        numeric_feature_data = df[numeric_feature_attrs]
        categorical_feature_data = df[categorical_feature_attrs]
        target_data = df[target_attr]

        ####################################################################################################
        encoder = self.estimator['encoder']
        feature_data_with_encoding = encoder.fit_transform(categorical_feature_data)
        feature_data = pd.concat([pd.DataFrame(feature_data_with_encoding), numeric_feature_data], axis=1)

        ####################################################################################################

        print(f'feature_data.shape: {feature_data.shape}')
        # 1. Split the data randomly with 70:30 of train and test.
        feature_train, feature_test, target_train, target_test = \
            train_test_split(feature_data, target_data, random_state=42, shuffle=False,
                             test_size=1 - float(options['train_factor']))
        self.ss_feature = StandardScaler()
        # 2. Standardlize the train and test data of features.
        ss_feature_train = self.ss_feature.fit_transform(feature_train)
        ss_feature_test = self.ss_feature.fit_transform(feature_test)

        # 3. Train the model with LogisticRegression
        self.estimator['algorithm'].fit(ss_feature_train, target_train)

        # 4. Evaluate the model performance
        y_pred = self.estimator['algorithm'].predict(
            pd.concat([
                pd.DataFrame(ss_feature_train),
                pd.DataFrame(ss_feature_test)
            ], axis=0))
        metrics = self.evaluate(target_data, y_pred)

        # 5. Handle the return value

        columns = options['target_attr']
        predict_columns = []
        for index in range(len(columns)):
            predict_columns.append(columns[index] + '_predicted')
        output = pd.concat([df, pd.DataFrame(y_pred, columns=predict_columns)], axis=1).reset_index(drop=True)

        return self.estimator, output, metrics





def fortinet_test_without_text_processing_for_user():
    ''' This is used for algorithm level test, should be run at the same dir of this file.
            python DecisionTreeClassifier.py
    '''
    import json

    raw_data = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/Resources/fortinet_reports/report1666743279291_with_incident_title_with_username.csv')

    raw_options = {
        'algorithm': 'BinaryRelevance',
        # 'encoder_type': 'OneHotEncoder',
        'encoder_type': 'OrdinalEncoder',
        # 'algorithm': 'RandomForestClassifier',
        'feature_attrs': [
            'Incident Category',
            'DayOfWeek(Event Receive Time)',
            'HourOfDay(Event Receive Time)',
            'Event Name',
            'Host IP',
            'Host Name',
            'Incident Source',
            'Incident Reporting Device',
            'Incident Target',
            'Attack Technique',
            'Attack Tactic'
        ],
        # 'target_attr': 'Incident Status',
        'target_attr': ['user_A', 'user_B', 'user_C', 'user_D'],
        'train_factor': 0.7
    }
    options = copy.deepcopy(raw_options)
    ##############################################################################################################

    decisiontree_classification = BalancedBaggingClassifier(options)

    print(f"raw_data[Incident Resolution].value_counts():{raw_data['Incident Resolution'].value_counts()}")

    model, output, metrics = decisiontree_classification.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_training.csv', index=False)

    infer_data = raw_data.iloc[:, :]
    # options.update({'model': pickle.dumps(model)})
    options = raw_options
    options.update({'model': {MODEL_TYPE_SINGLE: model}})

    output = decisiontree_classification.infer(infer_data, options)
    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_inference.csv', index=False)
    # t0 = datetime.now()
    # # x = infer_data.iloc[:1 + 10, :]
    # for i in range(1000):
    #     # output = decisiontree_classification.infer(infer_data.iloc[[i]], options)
    #     output = decisiontree_classification.infer(infer_data.iloc[:i + 10, :], options)
    #     print(i)
    #
    # delta = datetime.now() - t0

    # print(f'delta:{delta}')


def fortinet_test_without_text_processing_for_incident_resolution():
    ''' This is used for algorithm level test, should be run at the same dir of this file.
            python DecisionTreeClassifier.py
    '''
    import json

    raw_data = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/Resources/fortinet_reports/report1666743279291_with_incident_title_with_username.csv')

    raw_options = {
        'algorithm': 'BalancedBaggingClassifier',
        # 'encoder_type': 'OneHotEncoder',
        'encoder_type': 'OrdinalEncoder',
        # 'algorithm': 'RandomForestClassifier',
        'feature_attrs': [
            'Incident Category',
            'DayOfWeek(Event Receive Time)',
            'HourOfDay(Event Receive Time)',
            'Event Name',
            'Host IP',
            'Host Name',
            'Incident Source',
            'Incident Reporting Device',
            'Incident Target',
            'Attack Technique',
            'Attack Tactic'
        ],
        # 'target_attr': 'Incident Status',
        'target_attr': ['Incident Resolution'],
        'train_factor': 0.7
    }
    options = copy.deepcopy(raw_options)

    ##############################################################################################################

    balanced_bagging_classification = BalancedBaggingClassifier(options)

    print(f"raw_data[Incident Resolution].value_counts():{raw_data['Incident Resolution'].value_counts()}")

    model, output, metrics = balanced_bagging_classification.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_training.csv', index=False)

    infer_data = raw_data.iloc[:, :]
    # options.update({'model': pickle.dumps(model)})

    options = raw_options
    options.update({'model': {MODEL_TYPE_SINGLE: model}})

    t0 = datetime.now()

    output = balanced_bagging_classification.infer(infer_data, raw_options)
    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_inference.csv', index=False)
    # x = infer_data.iloc[:1 + 10, :]
    # for i in range(1000):
    #     # output = decisiontree_classification.infer(infer_data.iloc[[i]], options)
    #     output = decisiontree_classification.infer(infer_data.iloc[:i + 10, :], options)
    #     print(i)

    delta = datetime.now() - t0

    print(f'delta:{delta}')


if __name__ == '__main__':
    # fortinet_test_without_text_processing_for_user()
    fortinet_test_without_text_processing_for_incident_resolution()
