import os
import sys
import copy

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from sklearn_ex.utils.const_utils import Category, RunMode
from sklearn_ex.utils.excp_utils import phMLPreCheckError
from sklearn_ex.utils.log_utils import get_logger

logger = get_logger(__file__)


def parse_params(algo_params, floats=None, ints=None, strs=None, bools=None):
    def _get_default(obj):
        if obj is None:
            return []
        return obj

    def unquote_arg(arg):
        if len(arg) > 0 and (arg[0] == "'" or arg[0] == '"') and arg[0] == arg[-1]:
            return arg[1:-1]
        return arg

    floats = _get_default(floats)
    ints = _get_default(ints)
    strs = _get_default(strs)
    bools = _get_default(bools)
    input_params = {}

    for p in algo_params:
        if p in floats:
            try:
                input_params[p] = float(algo_params[p])
            except:
                raise RuntimeError("Invalid value for %s: must be a float" % p)
        elif p in ints:
            try:
                input_params[p] = int(algo_params[p])
            except:
                raise RuntimeError("Invalid value for %s: must be an int" % p)
        elif p in strs:
            input_params[p] = str(unquote_arg(algo_params[p]))
            if len(input_params[p]) == 0:
                raise RuntimeError("Invalid value for %s: must be a non-empty string" % p)
        elif p in bools:
            try:
                input_params[p] = algo_params[p]
            except RuntimeError:
                raise RuntimeError("Invalid value for %s: must be a boolean" % p)

    return input_params


def check_input(category=Category.REGRESSION.value, df_input=None, options=None):
    logger.info(f"Checking input data: Columns-{df_input.columns.values.tolist()},Types-{df_input.dtypes.to_dict()}")
    df_input.fillna(0, inplace=True)
    check_msg = ""

    if df_input.shape[0] == 0:
        check_msg += "The input is empty!"
        return check_msg

    missing_msg = check_missing_attrs(df_input, options)
    if missing_msg:
        check_msg += missing_msg
        return check_msg

    datatype_msg = check_datatype(category, df_input, options)
    if datatype_msg:
        check_msg += datatype_msg
        return check_msg


def check_missing_attrs(df_input, options):
    required_attrs = []
    if 'feature_attrs' in options and options['feature_attrs']:
        required_attrs = copy.deepcopy(options['feature_attrs'])
    if 'target_attr' in options and options['target_attr'] :
        required_attrs.append(options['target_attr'])
    if 'datetime_attr' in options and options['datetime_attr'] and df_input.index.name != options['datetime_attr']:
        required_attrs.append(options['datetime_attr'])

    logger.debug(f"Checking missing attrs for input data: {required_attrs}")
    missing_attr = set(required_attrs).difference(df_input.columns)

    if len(missing_attr) > 0:
        missing_msg = f"Missing attrs-{missing_attr}"
        logger.warn(missing_msg)
        return missing_msg
    return None


def check_datatype(category, df_input, options):
    datatype_msg = ""

    required_attrs = []
    if 'feature_attrs' in options and options['feature_attrs']:
        required_attrs = copy.deepcopy(options['feature_attrs'])
    if 'target_attr' in options and options['target_attr']:
        required_attrs.append(options['target_attr'])

    logger.debug(f"Checking attrs data type for features and target: {required_attrs}")

    def is_numberic_type(df_column):
        return df_column.dtype == "int64" or df_column.dtype == "float64"

    if category in [Category.REGRESSION.value, Category.FORECASTING.value, Category.ANOMALY_DETECTION.value]:
        not_numberic_attrs = [col for col in required_attrs if not is_numberic_type(df_input[col])]
        if len(not_numberic_attrs) > 0:
            datatype_msg += f"The attrs {not_numberic_attrs} are not numberic!"
            logger.warn(datatype_msg)
            return datatype_msg

    return None


if __name__ == '__main__':
    import numpy as np
    # data = {'name': ['Oliver', 'Harry', 'George', np.nan],
    #         'percentage': [90, 99, 50, 65],
    #         'grade': [88, np.nan, 95, np.nan]}
    #
    # df = pd.DataFrame(data)
    # print(df.columns.values.tolist())
    # print(df.dtypes.to_dict())
    # options = {
    #     "feature_attrs": ['name', 'percentage'],
    #     'target_attr': None
    # }
    #
    # try:
    #     check_msg = check_input(Category.CLUSTERING.value, df_input = df, options=options)
    #     if check_msg is not None:
    #         raise phMLPreCheckError(check_msg)
    #     print(df)
    # except phMLPreCheckError as e:
    #     print(e)

    pd.set_option('display.max_columns', None)
    df = pd.read_csv(BASE_DIR + "/resources/data/report_null.csv", na_values="\\N")
    print(df)
    df.fillna(0, inplace=True)
    print(df.columns.values.tolist())
    print(df.dtypes.to_dict())
    print(df)

