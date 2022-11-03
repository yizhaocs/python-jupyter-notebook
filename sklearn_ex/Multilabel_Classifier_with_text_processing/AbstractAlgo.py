import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, \
    recall_score, precision_score, roc_auc_score, hamming_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn_ex.Multilabel_Classifier_with_text_processing.utils.const_utils import MODEL_TYPE_SINGLE
from sklearn_ex.Multilabel_Classifier_with_text_processing.utils.excp_utils import phMLNotImplError
from sklearn_ex.utils.const_utils import ENCODER


class AbstractAlgo(object):
    """The AbstractAlgo class defines the interface for all phML algorithms."""

    def __init__(self, options):
        """The initialization function.

        This method is **required**. The __init__ method provides the chance to
        check grammar, convert parameters passed into the search, and initialize
        additional objects or imports needed by the algorithm. If none of these
        things are needed, a simple pass or return is sufficient.

        This will be called before processing any data.

        The `options` argument passed to this method is closely related to the
        algo implementations. Here is one example with LinearRegression algo:
            {
                'algo_name': u'LinearRegression',
                "algo_params" : {
                    "feature_attrs" : ["bedrooms", "bathrooms", "sqft_living", "sqft_lot"],
                    "target_attr" : "price",
                    "train_factor" : 70
                }
            }
        """
        self.feature_attrs = []
        self.target_attr = None
        msg = 'The {} algorithm cannot be initialized.'
        msg = msg.format(self.__class__.__name__)
        raise phMLNotImplError(msg)

    def train(self, df, options):
        """The train method creates and updates a model - it may make predictions.

        If the algorithm necessarily makes predictions while training, return
        the output DataFrame here. Additionally, if the algorithm cannot be
        saved, make predictions and return them. Otherwise, make predictions in
        the infer method and do not return anything here.

        The `df` argument is a pandas DataFrame from the search or reqport results.
        Note that modification to `df` within this method will also modify the
        dataframe to be used in the subsequent infer method.

        The `options` argument is the same as those described in the __init__
        method.
        """
        msg = 'The {} algorithm does not support train.'
        msg = msg.format(self.__class__.__name__)
        raise phMLNotImplError(msg)

    def infer(self, df, options):
        """The infer method creates predictions.

        The `df` argument is a pandas DataFrame from the search results.

        The `options` argument is the same as those described in the __init__
        method.
        """
        msg = 'The {} algorithm does not support apply.'
        msg = msg.format(self.__class__.__name__)
        raise phMLNotImplError(msg)

    def evaluate(self, X, y_pred):
        """The metrics method defines how to evaluate the model.
        The metrics method is only necessary with a saved model. This method
        must return dict.
        """
        msg = 'The {} algorithm does not support metrics.'
        msg = msg.format(self.__class__.__name__)
        raise phMLNotImplError(msg)


class AbstractClassifier(AbstractAlgo):
    def __init__(self):
        self.ss_feature = StandardScaler()

    def evaluate(self, model, y_true, y_pred, options=None):
        # Accuracy
        acc = accuracy_score(y_true, y_pred)

        # Hamming Loss :Incorrect Predictions
        # The Lower the result the better
        ham = hamming_loss(y_true, y_pred)

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

        metrics = {"accuracy:": acc, "hamming_score": ham, "confusion_matrix": confusion_metrix_dict}
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
        model_file = options['model']
        model = model_file[MODEL_TYPE_SINGLE]
        encoder = model_file[ENCODER]
        feature_attrs = options['feature_attrs']
        target_attr = options['target_attr']
        if 'text_processing' in options:
            text_processing_attr = options['text_processing']
            df_tfidfvect = self.text_preprocessing(df, options, 'infer')
            df = df.drop(text_processing_attr, axis=1)
            feature_data = pd.concat([df, df_tfidfvect], axis=1)
            feature_data.drop(target_attr, axis=1)
        else:
            feature_data = df[feature_attrs]


        if 'encoder' in options:
            if options['encoder'] == 'OrdinalEncoder':
                feature_data_with_encoding = encoder.transform(feature_data)
            elif options['encoder'] == 'LabelEncoder':
                feature_data_with_encoding = feature_data.apply(encoder.fit_transform)
            elif options['encoder'] == 'OneHotEncoder':
                feature_data_with_encoding = encoder.transform(feature_data).toarray()


        ss_feature_data = self.ss_feature.fit_transform(feature_data_with_encoding)
        y_pred = model.predict(ss_feature_data)
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
