import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier
from sklearn_ex.AbstractAlgo import AbstractClassifier
from sklearn_ex.utils.const_utils import MODEL_TYPE_SINGLE
from sklearn_ex.utils.param_utils import parse_params

DIFF_NAME = 'error'
PRIDCT_NAME = 'predicted'
DECIMAL_PRECISION = 2
FITTED_PARAMS = 'fitted_parameter'


class DecisionTreeClassifier_without_OneHotEncoding(AbstractClassifier):

    def __init__(self, options):
        super(DecisionTreeClassifier_without_OneHotEncoding, self).__init__()
        input_params = parse_params(
            options.get('algo_params', {}),
            ints=['min_samples_split', 'min_samples_leaf', 'random_state'],
            strs=['splitter'],
            floats=['min_weight_fraction_leaf', 'min_impurity_decrease', 'ccp_alpha']
        )
        self.estimator = _DecisionTreeClassifier(**input_params)


    def train(self, df, options):
        feature_attrs = options['feature_attrs']
        target_attr = options['target_attr']
        feature_data = df[feature_attrs]
        target_data = df[target_attr]

        # 1. Split the data randomly with 70:30 of train and test.
        feature_train, feature_test, target_train, target_test = \
            train_test_split(feature_data, target_data, random_state=42, shuffle=False,
                             test_size=1 - float(options['train_factor']))
        self.ss_feature = StandardScaler()
        # 2. Standardlize the train and test data of features.
        ss_feature_train = self.ss_feature.fit_transform(feature_train)
        ss_feature_test = self.ss_feature.fit_transform(feature_test)

        # 3. Train the model with LogisticRegression
        self.estimator.fit(ss_feature_train, target_train)

        # 4. Evaluate the model performance
        y_pred = self.estimator.predict(
            pd.concat([
                pd.DataFrame(ss_feature_train),
                pd.DataFrame(ss_feature_test)
            ], axis=0))
        metrics = self.evaluate(target_data, y_pred)
        feature_import = list(self.estimator.feature_importances_.round(DECIMAL_PRECISION))
        fitted_parameter = {feature_attrs[i]: feature_import[i] for i in range(len(feature_attrs))}
        metrics[FITTED_PARAMS] = fitted_parameter

        # 5. Handle the return value
        predict_name = f"{PRIDCT_NAME}({target_attr})"
        output = pd.concat([df, pd.DataFrame(y_pred, columns=[predict_name])], axis=1).reset_index(drop=True)
        output[DIFF_NAME] = output.apply(lambda x: 0 if x[target_attr] == x[predict_name] else 1, axis=1)

        return self.estimator, output, metrics


if __name__ == '__main__':
    ''' This is used for algorithm level test, should be run at the same dir of this file. 
            python DecisionTreeClassifier.py
    '''
    import json

    raw_data = pd.read_csv('../Resources/is_malware.csv')

    options = {
        'feature_attrs': ['bytes_sent', 'bytes_received', 'packets_sent', 'packets_received', 'dest_port', 'src_port',
                          'has_known_vulnerability'],
        'target_attr': 'used_by_malware',
        'train_factor': 0.7
    }
    decisiontree_classification = DecisionTreeClassifier_without_OneHotEncoding(options)
    model, output, metrics = decisiontree_classification.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    infer_data = raw_data.iloc[:, :]
    # options.update({'model': pickle.dumps(model)})
    options.update({'model': {MODEL_TYPE_SINGLE: model}})
    output = decisiontree_classification.infer(infer_data, options)
    print(output)
