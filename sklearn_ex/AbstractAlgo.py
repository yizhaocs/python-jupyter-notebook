import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, \
    recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_ex.utils.const_utils import *
from sklearn_ex.utils.excp_utils import phMLNotImplError


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

    def evaluate(self, y_true, y_pred, is_one_hot_encoding_on):
        confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
        confusion = {
            "True Negative": int(confusion_matrix.iloc[0, 0]),
            "False Positive": int(confusion_matrix.iloc[0, 1]),
            "False Negative": int(confusion_matrix.iloc[1, 0]),
            "True Positive": int(confusion_matrix.iloc[1, 1])
        }


        if is_one_hot_encoding_on:
            errors = {
                'Accuracy': round(accuracy_score(y_true, y_pred), DECIMAL_PRECISION),
                'F1 Score': round(f1_score(y_true, y_pred,
                                               pos_label='positive',
                                               average='micro'), DECIMAL_PRECISION),
                'Recall': round(recall_score(y_true, y_pred,
                                               pos_label='positive',
                                               average='micro'), DECIMAL_PRECISION),
                'Precision': round(precision_score(y_true, y_pred,
                                               pos_label='positive',
                                               average='micro'), DECIMAL_PRECISION),
                # 'ROC AUC': round(roc_auc_score(y_true, y_pred, multi_class='ovr'), DECIMAL_PRECISION),
                'Confusion': confusion
            }
        else:
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

    def infer(self, df, options):
        model_file = options['model']
        model = model_file[MODEL_TYPE_SINGLE]
        feature_attrs = options['feature_attrs']
        feature_data = df[feature_attrs]
        ohe = OneHotEncoder()
        feature_data_with_one_hot_encoding = ohe.fit_transform(feature_data).toarray()
        ss_feature_data = self.ss_feature.fit_transform(feature_data_with_one_hot_encoding)
        y_pred = model.predict(ss_feature_data)
        target_attr = options['target_attr']
        output = pd.concat([df, pd.DataFrame(y_pred, columns=[f"{PRIDCT_NAME}({target_attr})"])], axis=1)
        return output