import pickle

import numpy as np
import pandas as pd
from numpy import array
from sklearn.metrics import max_error, r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, \
    recall_score, precision_score, roc_auc_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

from sklearn_ex.utils.const_utils import ENCODER
from utils.const_utils import *
from utils.excp_utils import phMLNotImplError


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


class AbstractRegressor(AbstractAlgo):

    def __init__(self):
        self.ss_feature = StandardScaler()

    def train(self, df, options):
        model = {}
        output = None
        metrics = {}
        if 'id_attr' not in options or options['id_attr'] is "":
            single_model, single_output, single_metrics = self.train_single(df, options)
            model.update({MODEL_TYPE_SINGLE: single_model})
            output = single_output
            metrics = single_metrics
        else:
            id_attr = options['id_attr']
            id_list = list(df[id_attr].unique())
            multi_models = {}
            for id in id_list:
                single_model, single_output, single_metrics = \
                    self.train_single(df[df[id_attr] == id].reset_index(drop=True), options)
                multi_models.update({id: single_model})
                metrics.update({id: single_metrics})
                single_output[id_attr] = id
                output = single_output if output is None else pd.concat([output, single_output])
            model.update({MODEL_TYPE_MULTI: multi_models})
        return model, output, metrics

    def train_single(self, df, options):
        pass

    def infer(self, df, options):
        model_file = options['model']
        df = df.reset_index(drop=True)
        output = None
        if MODEL_TYPE_SINGLE in model_file:
            model = model_file[MODEL_TYPE_SINGLE]
            output = self.infer_single(df, options, model)
        elif MODEL_TYPE_MULTI in model_file:
            id_attr = options['id_attr']
            models = model_file[MODEL_TYPE_MULTI]
            for id in models.keys():
                model = models[id]
                single_pred = self.infer_single(df[df[id_attr] == id].reset_index(drop=True), options, model)
                single_pred[id_attr] = id
                output = single_pred if output is None else pd.concat([output, single_pred])
        else:
            raise Exception(f"The model was not valid! The model has to have a key of {MODEL_TYPE_SINGLE} "
                            f"or {MODEL_TYPE_MULTI}. The give model key is: {model_file.keys()}")

        return output.reset_index(drop=True)

    def infer_single(self, df, options, model):
        feature_attrs = options['feature_attrs']
        target_attr = options['target_attr']
        ss_feature_data = self.ss_feature.fit_transform(df[feature_attrs])
        y_pred = model.predict(df[feature_attrs])
        y_pred_df = pd.DataFrame(y_pred, columns=[f"{PRIDCT_NAME}({target_attr})"]).round(DECIMAL_PRECISION)
        output = pd.concat([df, y_pred_df], axis=1).reset_index(drop=True)

        return output

    def evaluate(self, y_true, y_pred):
        errors = {
            'Max Error': round(max_error(y_true, y_pred), DECIMAL_PRECISION),
            'R2 Score': round(r2_score(y_true, y_pred), DECIMAL_PRECISION),
            'Mean Absolute Error': round(mean_absolute_error(y_true, y_pred), DECIMAL_PRECISION),
            'Mean Squared Error': round(mean_squared_error(y_true, y_pred), DECIMAL_PRECISION),
            'Root Mean Squared Error': round(mean_squared_error(y_true, y_pred, squared=False), DECIMAL_PRECISION)
        }

        metrics = {FITTED_ERRORS: errors}

        return metrics


class AbstractClassifier(AbstractAlgo):
    def __init__(self):
        self.ss_feature = StandardScaler()

    def evaluate(self, y_true, y_pred):
        labels = y_true.iloc[:, 0].unique()
        if len(labels) == 2:  # true if binary classification
            confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
            confusion = {
                "True Negative": int(confusion_matrix.iloc[0, 0]),
                "False Positive": int(confusion_matrix.iloc[0, 1]),
                "False Negative": int(confusion_matrix.iloc[1, 0]),
                "True Positive": int(confusion_matrix.iloc[1, 1])
            }

            errors = {
                'Accuracy': round(accuracy_score(y_true, y_pred), DECIMAL_PRECISION),
                'F1 Score': round(f1_score(y_true, y_pred), DECIMAL_PRECISION),
                'Recall': round(recall_score(y_true, y_pred), DECIMAL_PRECISION),
                'Precision': round(precision_score(y_true, y_pred), DECIMAL_PRECISION),
                'ROC AUC': round(roc_auc_score(y_true, y_pred), DECIMAL_PRECISION),
                'Confusion': confusion
            }

            metrics = {FITTED_ERRORS: errors}

            return metrics
        else:
            from sklearn.metrics import multilabel_confusion_matrix, classification_report
            confusion_matrix_with_labels = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
            classification_report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
            # print(classification_report)

            confusion_metrix_dict = {}
            for label_col in range(len(labels)):
                confusion_matrix = confusion_matrix_with_labels[label_col]
                confusion = {
                    "True Negative": int(confusion_matrix[0, 0]),
                    "False Positive": int(confusion_matrix[0, 1]),
                    "False Negative": int(confusion_matrix[1, 0]),
                    "True Positive": int(confusion_matrix[1, 1])
                }
                confusion_metrix_dict[int(labels[label_col])] = confusion

            errors = {
                'Labels': labels.tolist(),
                'Classification Report': classification_report,
                'Confusion': confusion_metrix_dict
            }
            metrics = {FITTED_ERRORS: errors}
            return metrics

    def infer(self, df, options):
        labels = df[options['target_attr']].iloc[:, 0].unique()
        if len(labels) == 2:  # true if binary classification
            model_file = options['model']
            model = model_file[MODEL_TYPE_SINGLE]
            feature_attrs = options['feature_attrs']
            feature_data = df[feature_attrs]
            ss_feature_data = self.ss_feature.fit_transform(feature_data)
            y_pred = model.predict(ss_feature_data)
            target_attr = options['target_attr']
            output = pd.concat([df, pd.DataFrame(y_pred, columns=[f"{PRIDCT_NAME}({target_attr})"])], axis=1)
            return output
        else:
            model_file = options['model']
            model = model_file[MODEL_TYPE_SINGLE]
            encoder = model[ENCODER]

            feature_attrs = options['feature_attrs']
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

            feature_data_with_encoding = encoder.transform(categorical_feature_data)

            feature_data = pd.concat([pd.DataFrame(feature_data_with_encoding), numeric_feature_data], axis=1)

            ss_feature_data = self.ss_feature.fit_transform(feature_data)
            y_pred = model['algorithm'].predict(ss_feature_data)

            columns = options['target_attr']
            predict_columns = []
            for index in range(len(columns)):
                predict_columns.append(columns[index] + '_predicted')

            if options['algorithm'] == 'BinaryRelevance':
                y_pred = y_pred.toarray()
            output = pd.concat([df, pd.DataFrame(y_pred, columns=predict_columns)], axis=1).reset_index(drop=True)

            return output


class AbstractCluster(AbstractAlgo):

    def evaluate(self, X, labels):
        cluster_set = set(filter(lambda label: label >= 0, labels.tolist()))
        errors = {
            'Silhouette Score': round(silhouette_score(X, labels), DECIMAL_PRECISION),
            'Calinski Marabasz Score': round(calinski_harabasz_score(X, labels), DECIMAL_PRECISION),
            'Davies Bouldin Score': round(davies_bouldin_score(X, labels), DECIMAL_PRECISION),
            CLUSTERS_TITLE: len(cluster_set)
        }

        metrics = {FITTED_ERRORS: errors}

        return metrics


class AbstractAnomalyDetection(AbstractAlgo):

    def train(self, df, options):
        model = {}
        output = None
        metrics = OrderedDict()
        if 'id_attr' not in options or options['id_attr'] is "":
            single_model, single_output, single_metrics = self.train_single(df, options)
            model.update({MODEL_TYPE_SINGLE: single_model})
            output = single_output
            metrics.update({TOTAL: single_metrics})
        else:
            id_attr = options['id_attr']
            id_list = list(df[id_attr].unique())
            multi_models = {}
            for id in id_list:
                single_model, single_output, single_metrics = \
                    self.train_single(df[df[id_attr] == id].reset_index(drop=True), options)
                multi_models.update({id: single_model})
                metrics.update({id: single_metrics})
                single_output[id_attr] = id
                output = single_output if output is None else pd.concat([output, single_output])
            model.update({MODEL_TYPE_MULTI: multi_models})
            metrics.update({
                TOTAL: {
                    ANOMALY_TITLE: int(output[ANOMALY_NAME][output[ANOMALY_NAME] == 1].count()),
                    NOMALY_TITLE: int(output[ANOMALY_NAME][output[ANOMALY_NAME] == 0].count())
                }
            })
            metrics.move_to_end(TOTAL, last=False)
        return model, output, metrics

    def train_single(self, df, options):
        pass

    def infer(self, df, options):
        model_file = options['model']
        output = None
        df = df.reset_index(drop=True)
        if MODEL_TYPE_SINGLE in model_file:
            model = model_file[MODEL_TYPE_SINGLE]
            output = self.infer_single(df, options, model)
        elif MODEL_TYPE_MULTI in model_file:
            id_attr = options['id_attr']
            models = model_file[MODEL_TYPE_MULTI]
            for id in models.keys():
                model = models[id]
                single_pred = self.infer_single(df[df[id_attr] == id].reset_index(drop=True), options, model)
                output = single_pred if output is None else pd.concat([output, single_pred])
        else:
            raise Exception(f"The model was not valid! The model has to have a key of {MODEL_TYPE_SINGLE} "
                            f"or {MODEL_TYPE_MULTI}. The give model key is: {model_file.keys()}")

        return output.reset_index(drop=True)

    def infer_single(self, df, options, model):
        feature_attrs = options['feature_attrs']
        y_pred = model.predict(df[feature_attrs])
        y_pred_df = pd.DataFrame(y_pred, columns=[ANOMALY_NAME])
        y_pred_df[ANOMALY_NAME][y_pred_df[ANOMALY_NAME] == 1] = 0
        y_pred_df[ANOMALY_NAME][y_pred_df[ANOMALY_NAME] == -1] = 1
        output = pd.concat([df, y_pred_df], axis=1).reset_index(drop=True)

        return output
