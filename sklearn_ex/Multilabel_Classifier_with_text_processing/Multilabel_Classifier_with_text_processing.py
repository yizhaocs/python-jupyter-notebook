import pickle
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from skmultilearn.problem_transform import BinaryRelevance

from sklearn_ex.Multilabel_Classifier_with_text_processing.AbstractAlgo import AbstractClassifier
from sklearn_ex.utils.const_utils import MODEL_TYPE_SINGLE, FITTED_PARAMS, PRIDCT_NAME, DIFF_NAME, DECIMAL_PRECISION, ENCODER


class Classifier_with_text_processing(AbstractClassifier):

    def __init__(self, options):
        super(Classifier_with_text_processing, self).__init__()

        algorithm = options['algorithm']

        if algorithm == 'BinaryRelevance':
            '''
                According to my understanding of NB algorithm:
    
                    1.Gaussian NB: It should be used for features in decimal form. GNB assumes features to follow a normal distribution.
                    
                    2.MultiNomial NB: It should be used for the features with discrete values like word count 1,2,3...
                    
                    3.Bernoulli NB: It should be used for features with binary or boolean values like True/False or 0/1.
            '''
            self.estimator = BinaryRelevance(GaussianNB())
        else:
            self.estimator = RandomForestClassifier()

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
        categorical_feature_attrs = options['categorical_feature_attrs']
        numeric_feature_attrs = options['numeric_feature_attrs']


        target_attr = options['target_attr']
        if 'text_processing' in options:
            text_feature_data = self.text_preprocessing(df, options, 'train')
            categorical_feature_data = df[categorical_feature_attrs]
            numeric_feature_data = df[numeric_feature_attrs]

        else:
            categorical_feature_data = df[categorical_feature_attrs]
            numeric_feature_data = df[numeric_feature_attrs]


        target_data = df[target_attr]

        ####################################################################################################

        if 'encoder_type' in options:
            encoder = options['encoder']
            if options['encoder_type'] == 'OrdinalEncoder':
                feature_data_with_encoding = encoder.fit_transform(categorical_feature_data)
            elif options['encoder_type'] == 'LabelEncoder':
                feature_data_with_encoding = categorical_feature_data.apply(encoder.fit_transform)
            elif options['encoder_type'] == 'OneHotEncoder':
                feature_data_with_encoding = encoder.fit_transform(categorical_feature_data).toarray()

        feature_data = pd.concat([pd.DataFrame(feature_data_with_encoding), numeric_feature_data], axis=1)
        if 'text_processing' in options:
            feature_data = pd.concat([feature_data, text_feature_data], axis=1)

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
        self.estimator.fit(ss_feature_train, target_train)

        # 4. Evaluate the model performance
        y_pred = self.estimator.predict(
            pd.concat([
                pd.DataFrame(ss_feature_train),
                pd.DataFrame(ss_feature_test)
            ], axis=0))
        # Convert to Array  To See Result

        if options['algorithm'] == 'BinaryRelevance':
            y_pred = y_pred.toarray()

        metrics = None
        metrics = self.evaluate(self.estimator, pd.concat([
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


def fortinet_report_preprocessing(raw_data):
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
    ############################################################################################################################################
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
    ##############################################################################################################
    incident_status = raw_data['Incident Status']
    incident_resolution = raw_data['Incident Resolution']
    raw_data['Incident_Status_with_Incident_Resolution'] = incident_status.astype(str) + incident_resolution.astype(str)
    return raw_data


def real_data_test():
    ''' This is used for algorithm level test, should be run at the same dir of this file.
            python DecisionTreeClassifier.py
    '''
    import json

    raw_data = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/Resources/Multi-Label_Classification_Dataset/train.csv')

    options = {
        'algorithm': 'BinaryRelevance',
        # 'algorithm': 'RandomForestClassifier',
        'text_processing': 'TITLE',
        'feature_attrs': [
            'ABSTRACT'
        ],
        # 'target_attr': 'Incident Status',
        'target_attr': ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'],
        'train_factor': 0.7
    }

    ##############################################################################################################

    decisiontree_classification = Classifier_with_text_processing(options)

    model, output, metrics = decisiontree_classification.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/real_data_training.csv', index=False)

    infer_data = raw_data.iloc[:, :]
    # options.update({'model': pickle.dumps(model)})
    options.update({'model': {MODEL_TYPE_SINGLE: model}})
    output = decisiontree_classification.infer(infer_data, options)
    print(output)

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/real_data_inference.csv', index=False)


def fortinet_test_with_text_processing_for_user():
    ''' This is used for algorithm level test, should be run at the same dir of this file.
            python DecisionTreeClassifier.py
    '''
    import json

    raw_data = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/Resources/fortinet_reports/report1666743279291_with_incident_title_with_username.csv')

    options = {
        'algorithm': 'BinaryRelevance',
        'encoder_type': 'OneHotEncoder',
        'encoder': OneHotEncoder(handle_unknown='ignore'),
        # 'encoder_type': 'OrdinalEncoder',
        # 'encoder': OrdinalEncoder(encoded_missing_value=-1),
        # 'encoder_type': 'LabelEncoder',
        # 'encoder': LabelEncoder(),
        # 'algorithm': 'RandomForestClassifier',
        # 'text_processing': 'TITLE',
        'text_processing': 'Incident Title',
        'tfidf': TfidfVectorizer(analyzer='word', stop_words='english'),
        'numeric_feature_attrs': [
            'Incident Category',
            'DayOfWeek(Event Receive Time)',
            'HourOfDay(Event Receive Time)'
        ],
        'categorical_feature_attrs': [
            'Event Name',
            'Host IP',
            'Host Name',
            'Incident Source',
            'Incident Reporting Device',
            'incident_target_parsed_hostName',
            'incident_target_parsed_hostIpAddr',
            'techniqueid'
        ],
        # 'target_attr': 'Incident Status',
        'target_attr': ['user_A', 'user_B', 'user_C', 'user_D'],
        'train_factor': 0.7
    }

    raw_data = fortinet_report_preprocessing(raw_data)
    ##############################################################################################################

    decisiontree_classification = Classifier_with_text_processing(options)

    print(f"raw_data[Incident Resolution].value_counts():{raw_data['Incident Resolution'].value_counts()}")

    model, output, metrics = decisiontree_classification.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_training.csv', index=False)

    infer_data = raw_data.iloc[:, :]
    # options.update({'model': pickle.dumps(model)})
    options.update({'model': {MODEL_TYPE_SINGLE: model, ENCODER: options['encoder']}})

    t0 = datetime.now()

    output = decisiontree_classification.infer(infer_data, options)

    # x = infer_data.iloc[:1 + 10, :]
    # for i in range(1000):
    #     # output = decisiontree_classification.infer(infer_data.iloc[[i]], options)
    #     output = decisiontree_classification.infer(infer_data.iloc[:i + 10, :], options)
    #     print(i)

    delta = datetime.now() - t0

    print(f'delta:{delta}')

def fortinet_test_without_text_processing_for_user():
    ''' This is used for algorithm level test, should be run at the same dir of this file.
            python DecisionTreeClassifier.py
    '''
    import json

    raw_data = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/Resources/fortinet_reports/report1666743279291_with_incident_title_with_username.csv')

    options = {
        'algorithm': 'BinaryRelevance',
        'encoder_type': 'OneHotEncoder',
        'encoder': OneHotEncoder(handle_unknown='ignore'),
        # 'encoder_type': 'OrdinalEncoder',
        # 'encoder': OrdinalEncoder(encoded_missing_value=-1),
        # 'encoder_type': 'LabelEncoder',
        # 'encoder': LabelEncoder(),
        # 'algorithm': 'RandomForestClassifier',
        # 'text_processing': 'TITLE',
        # 'text_processing': 'Incident Title',
        # 'tfidf': TfidfVectorizer(analyzer='word', stop_words='english'),
        'numeric_feature_attrs': [
            'Incident Category',
            'DayOfWeek(Event Receive Time)',
            'HourOfDay(Event Receive Time)'
        ],
        'categorical_feature_attrs': [
            'Event Name',
            'Host IP',
            'Host Name',
            'Incident Source',
            'Incident Reporting Device',
            'incident_target_parsed_hostName',
            'incident_target_parsed_hostIpAddr',
            'techniqueid'
        ],
        # 'target_attr': 'Incident Status',
        'target_attr': ['user_A', 'user_B', 'user_C', 'user_D'],
        'train_factor': 0.7
    }

    raw_data = fortinet_report_preprocessing(raw_data)
    ##############################################################################################################

    decisiontree_classification = Classifier_with_text_processing(options)

    print(f"raw_data[Incident Resolution].value_counts():{raw_data['Incident Resolution'].value_counts()}")

    model, output, metrics = decisiontree_classification.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_training.csv', index=False)

    infer_data = raw_data.iloc[:, :]
    # options.update({'model': pickle.dumps(model)})
    options.update({'model': {MODEL_TYPE_SINGLE: model, ENCODER: options['encoder']}})

    t0 = datetime.now()
    # x = infer_data.iloc[:1 + 10, :]
    for i in range(1000):
        # output = decisiontree_classification.infer(infer_data.iloc[[i]], options)
        output = decisiontree_classification.infer(infer_data.iloc[:i + 10, :], options)
        print(i)

    delta = datetime.now() - t0

    print(f'delta:{delta}')



def fortinet_test_with_text_processing_for_incident_resolution():
    ''' This is used for algorithm level test, should be run at the same dir of this file.
            python DecisionTreeClassifier.py
    '''
    import json

    raw_data = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/Resources/fortinet_reports/report1666743279291_with_incident_title_with_username.csv')

    options = {
        'algorithm': 'BinaryRelevance',
        'encoder_type': 'OneHotEncoder',
        'encoder': OneHotEncoder(handle_unknown='ignore'),
        # 'encoder_type': 'OrdinalEncoder',
        # 'encoder': OrdinalEncoder(encoded_missing_value=-1),
        # 'encoder_type': 'LabelEncoder',
        # 'encoder': LabelEncoder(),
        # 'algorithm': 'RandomForestClassifier',
        # 'text_processing': 'TITLE',
        'text_processing': 'Incident Title',
        'tfidf': TfidfVectorizer(analyzer='word', stop_words='english'),
        'numeric_feature_attrs': [
            'Incident Category',
            'DayOfWeek(Event Receive Time)',
            'HourOfDay(Event Receive Time)'
        ],
        'categorical_feature_attrs': [
            'Event Name',
            'Host IP',
            'Host Name',
            'Incident Source',
            'Incident Reporting Device',
            'incident_target_parsed_hostName',
            'incident_target_parsed_hostIpAddr',
            'techniqueid'
        ],
        # 'target_attr': 'Incident Status',
        'target_attr': ['Incident Resolution'],
        'train_factor': 0.7
    }

    raw_data = fortinet_report_preprocessing(raw_data)
    ##############################################################################################################

    decisiontree_classification = Classifier_with_text_processing(options)

    print(f"raw_data[Incident Resolution].value_counts():{raw_data['Incident Resolution'].value_counts()}")

    model, output, metrics = decisiontree_classification.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_training.csv', index=False)

    infer_data = raw_data.iloc[:, :]
    # options.update({'model': pickle.dumps(model)})
    options.update({'model': {MODEL_TYPE_SINGLE: model, ENCODER: options['encoder']}})

    t0 = datetime.now()

    output = decisiontree_classification.infer(infer_data, options)

    # x = infer_data.iloc[:1 + 10, :]
    # for i in range(1000):
    #     # output = decisiontree_classification.infer(infer_data.iloc[[i]], options)
    #     output = decisiontree_classification.infer(infer_data.iloc[:i + 10, :], options)
    #     print(i)

    delta = datetime.now() - t0

    print(f'delta:{delta}')


def fortinet_test_without_text_processing_for_incident_resolution():
    ''' This is used for algorithm level test, should be run at the same dir of this file.
            python DecisionTreeClassifier.py
    '''
    import json

    raw_data = pd.read_csv('/Users/yzhao/PycharmProjects/python-jupyter-notebook/Resources/fortinet_reports/report1666743279291_with_incident_title_with_username.csv')

    options = {
        'algorithm': 'BinaryRelevance',
        'encoder_type': 'OneHotEncoder',
        'encoder': OneHotEncoder(handle_unknown='ignore'),
        # 'encoder_type': 'OrdinalEncoder',
        # 'encoder': OrdinalEncoder(encoded_missing_value=-1),
        # 'encoder_type': 'LabelEncoder',
        # 'encoder': LabelEncoder(),
        # 'algorithm': 'RandomForestClassifier',
        # 'text_processing': 'TITLE',
        # 'text_processing': 'Incident Title',
        # 'tfidf': TfidfVectorizer(analyzer='word', stop_words='english'),
        'numeric_feature_attrs': [
            'Incident Category',
            'DayOfWeek(Event Receive Time)',
            'HourOfDay(Event Receive Time)'
        ],
        'categorical_feature_attrs': [
            'Event Name',
            'Host IP',
            'Host Name',
            'Incident Source',
            'Incident Reporting Device',
            'incident_target_parsed_hostName',
            'incident_target_parsed_hostIpAddr',
            'techniqueid'
        ],
        # 'target_attr': 'Incident Status',
        'target_attr': ['Incident Resolution'],
        'train_factor': 0.7
    }

    raw_data = fortinet_report_preprocessing(raw_data)
    ##############################################################################################################

    decisiontree_classification = Classifier_with_text_processing(options)

    print(f"raw_data[Incident Resolution].value_counts():{raw_data['Incident Resolution'].value_counts()}")

    model, output, metrics = decisiontree_classification.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_training.csv', index=False)

    infer_data = raw_data.iloc[:, :]
    # options.update({'model': pickle.dumps(model)})
    options.update({'model': {MODEL_TYPE_SINGLE: model, ENCODER: options['encoder']}})

    t0 = datetime.now()

    output = decisiontree_classification.infer(infer_data, options)

    # x = infer_data.iloc[:1 + 10, :]
    # for i in range(1000):
    #     # output = decisiontree_classification.infer(infer_data.iloc[[i]], options)
    #     output = decisiontree_classification.infer(infer_data.iloc[:i + 10, :], options)
    #     print(i)

    delta = datetime.now() - t0

    print(f'delta:{delta}')

if __name__ == '__main__':
    # fortinet_test_with_text_processing_for_user()
    # fortinet_test_without_text_processing_for_user()
    # fortinet_test_with_text_processing_for_incident_resolution()
    fortinet_test_without_text_processing_for_incident_resolution()
    # real_data_test()
