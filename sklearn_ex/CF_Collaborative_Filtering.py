import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_ex.AbstractAlgo import AbstractClassifier
from sklearn_ex.utils.const_utils import MODEL_TYPE_SINGLE, FITTED_PARAMS, PRIDCT_NAME, DIFF_NAME, DECIMAL_PRECISION
from sklearn_ex.utils.param_utils import parse_params
from sklearn.neighbors import NearestNeighbors



class CF_Collaborative_Filtering(AbstractClassifier):

    def __init__(self, options):
        super(CF_Collaborative_Filtering, self).__init__()
        input_params = parse_params(
            options.get('algo_params', {}),
            strs=['metric', 'algorithm']
        )
        '''
            ref:
                https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb
            That way, you can train a classifier that will handle the imbalance without having to undersample or oversample manually before training.
        '''
        self.estimator = NearestNeighbors(**input_params)


    def train(self, df, options):
        number_neighbors =  options['number_neighbors']
        n_neighbors = number_neighbors
        feature_attrs = options['feature_attrs']
        target_attr = options['target_attr']
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
        distances, indices = self.estimator.kneighbors(
            pd.concat([
                pd.DataFrame(ss_feature_train),
                pd.DataFrame(ss_feature_test)
            ], axis=0),  n_neighbors=number_neighbors)
        print(f'distances:{distances}')
        print(f'indices:{indices}')

        # convert user_name to user_index
        print(f'df.columns.tolist():{df.columns.tolist()}')
        user_index = df.columns.tolist().index('User')
        print(f'user_index:{user_index}')

        df1 = df.copy()
        # t: movie_title, m: the row number of t in df
        for m, t in list(enumerate(df.index)):

            # find movies without ratings by user_4
            if df.iloc[m, user_index] == 0:
                sim_movies = indices[m].tolist()
                movie_distances = distances[m].tolist()

                # Generally, this is the case: indices[3] = [3 6 7]. The movie itself is in the first place.
                # In this case, we take off 3 from the list. Then, indices[3] == [6 7] to have the nearest NEIGHBORS in the list.
                if m in sim_movies:
                    id_movie = sim_movies.index(m)
                    sim_movies.remove(m)
                    movie_distances.pop(id_movie)

                    # However, if the percentage of ratings in the dataset is very low, there are too many 0s in the dataset.
                # Some movies have all 0 ratings and the movies with all 0s are considered the same movies by NearestNeighbors().
                # Then,even the movie itself cannot be included in the indices.
                # For example, indices[3] = [2 4 7] is possible if movie_2, movie_3, movie_4, and movie_7 have all 0s for their ratings.
                # In that case, we take off the farthest movie in the list. Therefore, 7 is taken off from the list, then indices[3] == [2 4].
                else:
                    sim_movies = sim_movies[:n_neighbors - 1]
                    movie_distances = movie_distances[:n_neighbors - 1]

                # movie_similarty = 1 - movie_distance
                movie_similarity = [1 - x for x in movie_distances]
                movie_similarity_copy = movie_similarity.copy()
                nominator = 0

                # for each similar movie
                for s in range(0, len(movie_similarity)):

                    # check if the rating of a similar movie is zero
                    if df.iloc[sim_movies[s], user_index] == 0:

                        # if the rating is zero, ignore the rating and the similarity in calculating the predicted rating
                        if len(movie_similarity_copy) == (number_neighbors - 1):
                            movie_similarity_copy.pop(s)

                        else:
                            movie_similarity_copy.pop(s - (len(movie_similarity) - len(movie_similarity_copy)))

                    # if the rating is not zero, use the rating and similarity in the calculation
                    else:
                        nominator = nominator + movie_similarity[s] * df.iloc[sim_movies[s], user_index]

                # check if the number of the ratings with non-zero is positive
                if len(movie_similarity_copy) > 0:

                    # check if the sum of the ratings of the similar movies is positive.
                    if sum(movie_similarity_copy) > 0:
                        predicted_r = nominator / sum(movie_similarity_copy)

                    # Even if there are some movies for which the ratings are positive, some movies have zero similarity even though they are selected as similar movies.
                    # in this case, the predicted rating becomes zero as well
                    else:
                        predicted_r = 0

                # if all the ratings of the similar movies are zero, then predicted rating should be zero
                else:
                    predicted_r = 0

                # place the predicted rating into the copy of the original dataset
                df1.iloc[m, user_index] = predicted_r
        '''
            metrics = self.evaluate(target_data, y_pred)
            # feature_import = list(self.estimator.feature_importances_.round(DECIMAL_PRECISION))
            # fitted_parameter = {feature_attrs[i]: feature_import[i] for i in range(len(feature_attrs))}
            # metrics[FITTED_PARAMS] = fitted_parameter
    
            # 5. Handle the return value
            predict_name = f"{PRIDCT_NAME}({target_attr})"
            output = pd.concat([df, pd.DataFrame(y_pred, columns=[predict_name])], axis=1).reset_index(drop=True)
            output[DIFF_NAME] = output.apply(lambda x: 0 if x[target_attr] == x[predict_name] else 1, axis=1)
        '''
        return self.estimator, output, metrics


if __name__ == '__main__':
    ''' This is used for algorithm level test, should be run at the same dir of this file. 
            python DecisionTreeClassifier.py
    '''
    import json


    raw_data = pd.read_csv('../Resources/fortinet_reports/report1665512454703_resolution_as_user.csv')

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
    options = {
        'algo_params': json.dumps({
            "metric": "cosine",
            "algorithm": "brute"
        }),
        'number_neighbors': 3,
        'feature_attrs': [
            'Event Name',
            'Host IP',
            'Host Name',
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
        'target_attr': 'User',
        'train_factor': 0.7
    }
    decisiontree_classification = CF_Collaborative_Filtering(options)
    model, output, metrics = decisiontree_classification.train(raw_data, options)
    print(output)
    print(json.dumps(metrics, indent=2))

    manually_cleared_incidents_rows = output.loc[(
                                                      # False Positive
                                                      output['Incident ID'] == 9705)
                                                 | (output['Incident ID'] == 7782)
                                                 | (output['Incident ID'] == 9525)
                                                 | (output['Incident ID'] == 9738)
                                                 | (output['Incident ID'] == 7780)
                                                 #True Positive
                                                 | (output['Incident ID'] == 9461)
                                                 | (output['Incident ID'] == 7779)
                                                 | (output['Incident ID'] == 9090)
                                                 | (output['Incident ID'] == 9740)
                                                 | (output['Incident ID'] == 9143)
                                                 ]
    manually_cleared_incidents_rows.to_csv('/Users/yzhao/Documents/ai_for_operational_management/manually_cleared_incidents_rows.csv', index=False)

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
    output = decisiontree_classification.infer(infer_data, options)
    print(output)

    output.to_csv('/Users/yzhao/Documents/ai_for_operational_management/ai_for_operational_management_inference.csv', index=False)
