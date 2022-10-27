import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skmultilearn.problem_transform import BinaryRelevance

from sklearn_ex.AbstractAlgo import AbstractClassifier
from sklearn_ex.utils.const_utils import MODEL_TYPE_SINGLE, FITTED_PARAMS, PRIDCT_NAME, DIFF_NAME, DECIMAL_PRECISION
from sklearn_ex.utils.param_utils import parse_params
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier


class Classifier_with_text_processing(AbstractClassifier):

    def __init__(self, options):
        super(Classifier_with_text_processing, self).__init__()

        algorithm = options['algorithm']
        if algorithm == 'DecisionTreeClassifier':
            input_params = parse_params(
                options.get('algo_params', {}),
                ints=['min_samples_split', 'min_samples_leaf', 'random_state'],
                strs=['splitter'],
                floats=['min_weight_fraction_leaf', 'min_impurity_decrease', 'ccp_alpha']
            )
            '''
                ref:
                    https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb
                That way, you can train a classifier that will handle the imbalance without having to undersample or oversample manually before training.
            '''

            self.estimator = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(**input_params),
                                                       sampling_strategy='auto',
                                                       replacement=False,
                                                       random_state=0)


        elif algorithm == 'RandomForestClassifier':
            input_params = parse_params(
                options.get('algo_params', {}),
                ints=['n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state'],
                strs=['criterion', 'splitter'],
                floats=['min_weight_fraction_leaf', 'min_impurity_decrease', 'ccp_alpha']
            )
            self.estimator = BalancedBaggingClassifier(base_estimator=_RandomForestClassifier(**input_params),
                                                       sampling_strategy='auto',
                                                       replacement=False,
                                                       random_state=0)
        elif algorithm == 'multilabel':
            self.estimator = BinaryRelevance(GaussianNB())

    def text_preprocessing(self, df):
        import neattext as nt
        import neattext.functions as nfx

        # Explore For Noise
        df['Incident Title'].apply(lambda x: nt.TextFrame(x).noise_scan())

        # Explore For Noise
        df['Incident Title'].apply(lambda x: nt.TextExtractor(x).extract_stopwords())

        # Explore For Noise
        df['Incident Title'].apply(nfx.remove_stopwords)

        corpus = df['Incident Title'].apply(nfx.remove_stopwords)

        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
        # Build Features
        Xfeatures = tfidf.fit_transform(corpus).toarray()
        df_tfidfvect = pd.DataFrame(data=Xfeatures, columns=tfidf.get_feature_names())
        df = df.drop('Incident Title', axis=1)
        df = pd.concat([df, df_tfidfvect], axis=1)
        return df

    def train(self, df, options, is_text_preprocessing):
        feature_attrs = options['feature_attrs']
        target_attr = options['target_attr']
        if is_text_preprocessing:
            feature_data = self.text_preprocessing(df[feature_attrs])
        else:
            feature_data = df[feature_attrs]

        target_data = df[target_attr]

        ####################################################################################################
        ohe = OneHotEncoder()
        feature_data_with_one_hot_encoding = ohe.fit_transform(feature_data).toarray()
        ####################################################################################################

        # 1. Split the data randomly with 70:30 of train and test.
        feature_train, feature_test, target_train, target_test = \
            train_test_split(feature_data_with_one_hot_encoding, target_data, random_state=42, shuffle=False,
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
        metrics = None
        if not is_text_preprocessing:
            metrics = self.evaluate(target_data, y_pred, options)
        # feature_import = list(self.estimator.feature_importances_.round(DECIMAL_PRECISION))
        # fitted_parameter = {feature_attrs[i]: feature_import[i] for i in range(len(feature_attrs))}
        # metrics[FITTED_PARAMS] = fitted_parameter

        # 5. Handle the return value
        predict_name = f"{PRIDCT_NAME}({target_attr})"
        output = pd.concat([df, pd.DataFrame(y_pred, columns=[predict_name])], axis=1).reset_index(drop=True)

        if not is_text_preprocessing:
            output[DIFF_NAME] = output.apply(lambda x: 0 if x[target_attr] == x[predict_name] else 1, axis=1)

        return self.estimator, output, metrics


if __name__ == '__main__':
    ''' This is used for algorithm level test, should be run at the same dir of this file. 
            python DecisionTreeClassifier.py
    '''
    import json

    raw_data = pd.read_csv('../Resources/fortinet_reports/report1666743279291_with_incident_title_with_username.csv')

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


    ##############################################################################################################
    options = {
        'algorithm': 'multilabel',
        'feature_attrs': [
            'Event Name',
            'Host IP',
            'Host Name',
            'Incident Title',
            'Incident Source',
            'Incident Reporting Device',
            'incident_target_parsed_hostName',
            'incident_target_parsed_hostIpAddr',
            'Incident Category',
            'DayOfWeek(Event Receive Time)',
            'HourOfDay(Event Receive Time)',
            'techniqueid'
        ],
        # 'target_attr': 'Incident Status',
        'target_attr': ['user_A', 'user_B', 'user_C', 'user_D'],
        'train_factor': 0.7
    }

    is_text_preprocessing = False
    if options['algorithm'] == 'multilabel':
        is_text_preprocessing = True


    ##############################################################################################################
    if not is_text_preprocessing:
        labels = raw_data["Incident_Status_with_Incident_Resolution"].unique()
        print(f'labels:{labels}')

    decisiontree_classification = Classifier_with_text_processing(options)
    model, output, metrics = decisiontree_classification.train(raw_data, options, is_text_preprocessing)
    print(output)
    print(json.dumps(metrics, indent=2))

    # manually_cleared_incidents_rows = output.loc[(
    #                                                  # False Positive
    #                                                      output['Incident ID'] == 9705)
    #                                              | (output['Incident ID'] == 7782)
    #                                              | (output['Incident ID'] == 9525)
    #                                              | (output['Incident ID'] == 9738)
    #                                              | (output['Incident ID'] == 7780)
    #                                              # True Positive
    #                                              | (output['Incident ID'] == 9461)
    #                                              | (output['Incident ID'] == 7779)
    #                                              | (output['Incident ID'] == 9090)
    #                                              | (output['Incident ID'] == 9740)
    #                                              | (output['Incident ID'] == 9143)
    #                                              ]
    # manually_cleared_incidents_rows.to_csv('/Users/yzhao/Documents/ai_for_operational_management/manually_cleared_incidents_rows.csv', index=False)

    if not is_text_preprocessing:
        for label_col in range(len(labels)):
            label = labels[label_col]
            '''
                good and error for the lable is 11
            '''
            error = output.loc[(output['error'] == 0) & (output['Incident_Status_with_Incident_Resolution'] == label)]
            error.to_csv('/Users/yzhao/Documents/ai_for_operational_management/true_positive_' + label + '.csv', index=False)
            good = output.loc[(output['error'] == 1) & (output['Incident_Status_with_Incident_Resolution'] == label)]
            good.to_csv('/Users/yzhao/Documents/ai_for_operational_management/false_negative_' + label + '.csv', index=False)

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_training.csv', index=False)

    infer_data = raw_data.iloc[:, :]
    # options.update({'model': pickle.dumps(model)})
    options.update({'model': {MODEL_TYPE_SINGLE: model}})
    output = decisiontree_classification.infer(infer_data, options, is_text_preprocessing)
    print(output)

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_inference.csv', index=False)
