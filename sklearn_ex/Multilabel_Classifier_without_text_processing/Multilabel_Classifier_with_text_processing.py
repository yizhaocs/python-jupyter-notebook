import copy
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from skmultilearn.problem_transform import BinaryRelevance

from sklearn_ex.utils.const_utils import MODEL_TYPE_SINGLE, ENCODER


class Classifier_with_text_processing():

    def __init__(self, options):
        self.ss_feature = StandardScaler()
        algorithm = options['algorithm']
        encoder = OrdinalEncoder(encoded_missing_value=-1)
        if algorithm == 'BinaryRelevance':
            self.estimator = {
                "algorithm": BinaryRelevance(GaussianNB()),
                "encoder": encoder
            }
        elif algorithm == 'BalancedBaggingClassifier':
            self.estimator = {
                "algorithm": BalancedBaggingClassifier(base_estimator=RandomForestClassifier(),
                                                       sampling_strategy='auto',
                                                       replacement=False,
                                                       random_state=0),
                "encoder": encoder
            }
        elif algorithm == 'RandomForestClassifier':
            self.estimator = {
                "algorithm": RandomForestClassifier(),
                "encoder": encoder
            }

    def text_preprocessing(self, df, options, mode):
        import neattext as nt
        import neattext.functions as nfx
        tfidf = options['tfidf']
        # Explore For Noise
        col = options['text_processing']
        df[col].apply(lambda x: nt.TextFrame(x).noise_scan())

        # Explore For Noise
        df[col].apply(lambda x: nt.TextExtractor(x).extract_stopwords())

        # Explore For Noise
        df[col].apply(nfx.remove_stopwords)

        corpus = df[col].apply(nfx.remove_stopwords)

        # Build Features
        if mode == 'train':
            Xfeatures = tfidf.fit_transform(corpus).toarray()
        else:
            Xfeatures = tfidf.transform(corpus).toarray()
        df_tfidfvect = pd.DataFrame(data=Xfeatures, columns=tfidf.get_feature_names())
        # df = df.drop(col, axis=1)

        return df_tfidfvect

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

        if 'encoder_type' in options:
            encoder = self.estimator['encoder']
            if options['encoder_type'] == 'OrdinalEncoder':
                feature_data_with_encoding = encoder.fit_transform(categorical_feature_data)
            elif options['encoder_type'] == 'LabelEncoder':
                feature_data_with_encoding = categorical_feature_data.apply(encoder.fit_transform)
            elif options['encoder_type'] == 'OneHotEncoder':
                feature_data_with_encoding = encoder.fit_transform(categorical_feature_data).toarray()

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
        # Convert to Array  To See Result

        if options['algorithm'] == 'BinaryRelevance':
            y_pred = y_pred.toarray()

        metrics = None
        metrics = self.evaluate(self.estimator['algorithm'], pd.concat([
            pd.DataFrame(target_train),
            pd.DataFrame(target_test)
        ], axis=0), y_pred, options)

        # feature_import = list(self.estimator.feature_importances_.round(DECIMAL_PRECISION))
        # fitted_parameter = {feature_attrs[i]: feature_import[i] for i in range(len(feature_attrs))}
        # metrics[FITTED_PARAMS] = fitted_parameter

        # 5. Handle the return value

        columns = options['target_attr']
        predict_columns = []
        for index in range(len(columns)):
            predict_columns.append(columns[index] + '_predicted')
        output = pd.concat([df, pd.DataFrame(y_pred, columns=predict_columns)], axis=1).reset_index(drop=True)

        return self.estimator, output, metrics

    def evaluate(self, model, y_true, y_pred, options=None):
        # Accuracy
        # acc = accuracy_score(y_true.to_numpy(), y_pred)

        # Hamming Loss :Incorrect Predictions
        # The Lower the result the better
        # ham = hamming_loss(y_true.to_numpy(), y_pred)

        if len(y_true.columns) > 1:
            multilabel_classification_report = classification_report(
                y_true,
                y_pred,
                output_dict=False,
                target_names=y_true.columns
            )

            print(multilabel_classification_report)

        from sklearn.metrics import multilabel_confusion_matrix
        confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
        columns = y_true.columns
        confusion_metrix_dict = {}
        for index in range(len(columns)):
            single_confusion = confusion_matrix[index]
            confusion = {
                "True Negative": int(single_confusion[0, 0]),
                "False Positive": int(single_confusion[0, 1]),
                "False Negative": int(single_confusion[1, 0]),
                "True Positive": int(single_confusion[1, 1])
            }
            confusion_metrix_dict[columns[index]] = confusion

        metrics = {"confusion_matrix": confusion_metrix_dict}
        print(f'metrics:{metrics}')

        '''
            Feature ranking
        '''
        if options['algorithm'] == 'RandomForestClassifier':
            importances = model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]

            # Print the feature ranking
            print("Feature ranking:")

            for f in range(0, 27):
                print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        '''
            Each label accuracy
        '''
        y_pred = pd.DataFrame(y_pred, columns=options['target_attr'])
        for label in options['target_attr']:
            print('\n')
            print('... Processing {}'.format(label))

            # Checking overall accuracy
            print('Testing Accuracy is {}'.format(accuracy_score(y_true[label], y_pred[label])))
        return metrics

    def infer(self, df, options):
        df = data_praser(df, options)
        model_file = options['model']
        model = model_file[MODEL_TYPE_SINGLE]
        encoder = model[ENCODER]

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

        if 'encoder_type' in options:
            if options['encoder_type'] == 'OrdinalEncoder':
                feature_data_with_encoding = encoder.transform(categorical_feature_data)
            elif options['encoder_type'] == 'OneHotEncoder':
                feature_data_with_encoding = encoder.transform(categorical_feature_data).toarray()

        # feature_data = pd.concat([pd.DataFrame(feature_data_with_encoding), numeric_feature_data], axis=1)
        feature_data = pd.concat([pd.DataFrame(feature_data_with_encoding), numeric_feature_data], axis=1)

        ss_feature_data = self.ss_feature.fit_transform(feature_data)
        y_pred = model['algorithm'].predict(ss_feature_data)
        target_attr = options['target_attr']
        # output = pd.concat([df, pd.DataFrame(y_pred, columns=[f"{PRIDCT_NAME}({target_attr})"])], axis=1)

        columns = options['target_attr']
        predict_columns = []
        for index in range(len(columns)):
            predict_columns.append(columns[index] + '_predicted')

        if options['algorithm'] == 'BinaryRelevance':
            y_pred = y_pred.toarray()
        output = pd.concat([df, pd.DataFrame(y_pred, columns=predict_columns)], axis=1).reset_index(drop=True)

        return output


def incident_target_column_parsing(raw_data, options):
    feature_attrs = options['feature_attrs']
    if 'Incident Target' not in feature_attrs:
        return raw_data

    feature_attrs.remove('Incident Target')
    feature_attrs.append('incident_target_parsed_hostIpAddr')
    feature_attrs.append('incident_target_parsed_hostName')
    ############################################################################################################################################
    incident_target_parsed = raw_data['Incident Target'].str.split(pat=',', expand=False)
    # print(incident_target_parsed.head())
    # print(incident_target_parsed.iloc[0])
    # print(incident_target_parsed.iloc[0][0])

    hostIpAddr_data = []
    hostName_data = []
    for i in range(len(incident_target_parsed)):
        e = incident_target_parsed.iloc[i]
        if isinstance(e, list):
            # print(f'e:{e}')
            for i in range(0, len(e)):
                s = e[i]
                if 'hostIpAddr' in s:
                    hostIpAddr_data.append(s.partition(':')[2])
                    # print(f'hostIpAddr:{s}')
                elif 'hostName' in s:
                    hostName_data.append(s.partition(':')[2])
                    # print(f'hostName:{s}')
        else:
            hostIpAddr_data.append(pd.np.nan)
            hostName_data.append(pd.np.nan)

    hostIpAddr_data_df = pd.DataFrame(hostIpAddr_data, columns=['incident_target_parsed_hostIpAddr'])
    hostName_data_df = pd.DataFrame(hostName_data, columns=['incident_target_parsed_hostName'])

    raw_data = pd.concat([raw_data, hostIpAddr_data_df], axis=1)
    raw_data = pd.concat([raw_data, hostName_data_df], axis=1)
    return raw_data
    ############################################################################################################################################


def attack_technique_column_parsing(raw_data, options):
    feature_attrs = options['feature_attrs']
    if 'Attack Technique' not in feature_attrs:
        return raw_data
    feature_attrs.remove('Attack Technique')
    feature_attrs.append('techniqueid')
    attack_tactic_parsed = raw_data['Attack Technique'].str.split(pat=',', expand=False)
    techniqueid_data = []

    for i in range(len(attack_tactic_parsed)):
        e = attack_tactic_parsed.iloc[i]
        if e and isinstance(e, list):
            # print(f'e:{e}')
            for i in range(0, len(e)):
                s = e[i]
                if 'techniqueid' in s:
                    import re

                    techniqueid = s.partition(':')[2]

                    list_of_char = ['\"', '}', '\]', ' ']
                    pattern = '[' + ''.join(list_of_char) + ']'
                    techniqueid = re.sub(pattern, '', techniqueid)
                    techniqueid_data.append(techniqueid)
                    break
                    # print(f'techniqueid:{s}')
        else:
            techniqueid_data.append(pd.np.nan)

    techniqueid_data_df = pd.DataFrame(techniqueid_data, columns=['techniqueid'])
    raw_data = pd.concat([raw_data, techniqueid_data_df], axis=1)
    print(f'raw_data.columns:{raw_data.columns}')
    return raw_data

def data_praser(raw_data, options):
    raw_data = incident_target_column_parsing(raw_data, options)
    raw_data = attack_technique_column_parsing(raw_data, options)
    return raw_data

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

    decisiontree_classification = Classifier_with_text_processing(options)

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

    decisiontree_classification = Classifier_with_text_processing(options)

    print(f"raw_data[Incident Resolution].value_counts():{raw_data['Incident Resolution'].value_counts()}")

    model, output, metrics = decisiontree_classification.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_training.csv', index=False)

    infer_data = raw_data.iloc[:, :]
    # options.update({'model': pickle.dumps(model)})

    options = raw_options
    options.update({'model': {MODEL_TYPE_SINGLE: model}})

    t0 = datetime.now()

    output = decisiontree_classification.infer(infer_data, raw_options)
    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_inference.csv', index=False)
    # x = infer_data.iloc[:1 + 10, :]
    # for i in range(1000):
    #     # output = decisiontree_classification.infer(infer_data.iloc[[i]], options)
    #     output = decisiontree_classification.infer(infer_data.iloc[:i + 10, :], options)
    #     print(i)

    delta = datetime.now() - t0

    print(f'delta:{delta}')


if __name__ == '__main__':
    fortinet_test_without_text_processing_for_user()
    # fortinet_test_without_text_processing_for_incident_resolution()
