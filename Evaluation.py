import os
import sys
import time
from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from Utility import _train_test_splitting

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from DataGenerator import Generator
from Taxonomy import EngineTaxonomy
import numpy as np
import pandas as pd
from pymfe.mfe import MFE
import sklearn.metrics as skm
from fairlearn.metrics import MetricFrame
from functools import partial
from fairlearn.metrics import count
import warnings

warnings.filterwarnings('ignore')


def extract_mfes(X, y, meta_feature_set, summary=["mean"], groups=["all"]):
    mfe = MFE(groups=groups, features=meta_feature_set,
              summary=summary)
    mfe.fit(X, y)
    ft = mfe.extract()
    return ft


def elements(array):
    return array.ndim and array.size


def extract_statistics_from_data(X, y):
    stats = {}

    # Basic stats (#instances etc.)
    stats["avg #n classes+groups"] = df.groupby(['group', 'target'])["target"].count().mean()
    stats["min #n classes+groups"] = df.groupby(['group', 'target'])["target"].count().min()
    stats["max #n classes+groups"] = df.groupby(['group', 'target'])["target"].count().max()

    stats["avg #n groups"] = df.groupby(['group']).size().mean()
    stats["min #n groups"] = df.groupby(['group']).size().min()
    stats["max #n groups"] = df.groupby(['group']).size().max()

    stats["avg #c groups"] = df.groupby(['group'])["target"].nunique().mean()
    stats["min #c groups"] = df.groupby(['group'])["target"].nunique().min()
    stats["max #c groups"] = df.groupby(['group'])["target"].nunique().max()

    #####################################
    ### Complexity metrics  from PyMFE ###
    X = KNNImputer().fit_transform(X)
    ft = extract_mfes(X, y, complexity_metrics, groups=["complexity"])

    for metric, value in zip(ft[0], ft[1]):
        print(f"{metric} (C): {value}")
        stats[f"{metric} (C)"] = value

    ### On Groups ###
    for group in groups:
        group_df = df[df["group"] == group]
        group_X, group_y = group_df[[f"F{i}" for i in range(f)]].to_numpy(), group_df["target"].to_numpy()
        group_X = KNNImputer().fit_transform(group_X)
        ft = extract_mfes(group_X, group_y, complexity_metrics)

        for metric, value in zip(ft[0], ft[1]):
            stats[f"{metric} (G)"] = value

    for key, value in stats.items():
        if "(G)" in key:
            stats[key] = [np.nanmean(value)]
    #####################################

    # ######################################
    # ### Gini #############################
    stats[f"Gini (C)"] = generator.gini(y)
    stats[f"Gini (G)"] = generator.gini(df["group"])
    # ######################################

    for key, value in stats.items():
        if not isinstance(value, list):
            stats[key] = [value]
    return stats


def calculate_accuracy():
    df_train, df_test = _train_test_splitting(df)
    ### Prediction part ###
    X_train, y_train = df_train[[f"F{i}" for i in range(f)]], df_train["target"]
    X_test, y_test = df_test[[f"F{i}" for i in range(f)]], df_test["target"]
    train_imputer = KNNImputer()
    X_train = train_imputer.fit_transform(X_train)
    X_test = train_imputer.transform(X_test)

    model_X = RandomForestClassifier()
    model_X.fit(X_train, y_train)
    y_pred_X = model_X.predict(X_test)
    acc_X = skm.accuracy_score(y_pred_X, y_test)

    mf_X = MetricFrame({'accuracy': skm.accuracy_score,
                        'F1': partial(skm.f1_score, average='weighted'),
                        'prec': partial(skm.precision_score, average='weighted'),
                        'recall': partial(skm.recall_score, average='weighted'),
                        'count': count},
                       y_true=y_test,
                       y_pred=y_pred_X,
                       sensitive_features=df_test['group'])

    ## For each group ##
    model_repo = {}
    for group in df_train["group"]:
        group_df = df_train[df_train["group"] == group]
        group_X = group_df[[f"F{i}" for i in range(f)]].to_numpy()
        model = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                          ("forest", RandomForestClassifier())])
        group_y = group_df["target"].to_numpy()
        # model = RandomForestClassifier()
        model.fit(group_X, group_y)
        model_repo[group] = model

    y_group_pred = df_test.apply(
        lambda row: model_repo[row["group"]].predict(row[[f"F{i}" for i in range(f)]].to_numpy().reshape(1, -1))[0],
        axis=1)

    group_accuracy = skm.accuracy_score(y_group_pred, y_test)

    mf = MetricFrame({'accuracy': skm.accuracy_score,
                      'F1': partial(skm.f1_score, average='weighted'),
                      'prec': partial(skm.precision_score, average='weighted'),
                      'recall': partial(skm.recall_score, average='weighted'),
                      'count': count}, y_true=df_test['target'], y_pred=y_group_pred,
                     sensitive_features=df_test['group'])

    # Store Predictions for whole Data and Groups in predictions.csv
    mf_g = mf.by_group
    mf_g["Model"] = "G"
    mf_x = mf_X.by_group
    mf_x["Model"] = "X"
    dataset_predictions = pd.concat([mf.by_group, mf_X.by_group])
    return acc_X, group_accuracy, dataset_predictions


def _update_data_informations(dfs, data_config):
    for df in dfs:
        for config_name, config_value in data_config.items():
            df[config_name] = config_value


def _result_available(dfs, data_config):
    try:
        # Check for all given data Frames if they contain the data config
        return all([df.shape[0] > 0 for df in dfs]) \
               and all([(df[list(data_config)] == pd.Series(data_config)).all(axis=1).any() for df in dfs])
    except KeyError as e:
        # column is not in dataframe
        print(e)
        return False


if __name__ == '__main__':
    # Here you can also select only a subset of the complexity metrics from pymfe.
    # See https://pymfe.readthedocs.io/en/latest/auto_pages/meta_features_description.html
    complexity_metrics = "all"

    # We use the taxonomy from a real-world use case
    taxonomy = EngineTaxonomy().create_taxonomy()

    ####################################################################
    ## Define the possible configurations for the generated Datasets ###
    # Varying parameters are the ones that we varied for our evaluation
    varying_parameters = {
        "sC": list(range(0, 6)),
        "sG": [0, 0.5, 1.0, 1.5, 2],
        "gs": [0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0],
        "cf": [1, 5, 10, 15, 20, 25, 30, 40],
    }

    # These ones are fixed
    default_parameter_setting = {
        "n": 1000,
        "n_features": 50,
        "c": 20,
        "class_overlap": 1.5,
        "root": taxonomy,
    }
    #####################################################################

    # Iterate over all possible combinations of the varying_parameters
    for data_config_values in product(*varying_parameters.values()):

        data_config = default_parameter_setting.copy()
        varied_parameters = dict(zip(varying_parameters, data_config_values))
        data_config.update(varied_parameters)

        print('---------------------------------')
        print(f"Running with config:")
        print(f'{data_config}')

        start = time.time()
        f = data_config["n_features"]
        generator = Generator(hardcoded=False,
                              **data_config)
        df = generator.generate_data_from_taxonomy()
        if "index" in df.columns:
            df = df.drop("index", axis=1)
        df = df.dropna(how="any")

        X, y = df[[f"F{i}" for i in range(f)]].to_numpy(), df["target"].to_numpy()
        groups = np.unique(df["group"].to_numpy())

        # Statistics and Complexity Measures
        stats = extract_statistics_from_data(X, y)

        # accuracy
        predictions = pd.DataFrame()
        accuracy_x, accuracy_groups, predictions = calculate_accuracy()
        stats["Acc (X)"] = accuracy_x
        stats["Acc (G)"] = accuracy_groups
        stats["Acc (G - X)"] = accuracy_groups - accuracy_x

        # update data config to dataframes. Also store the parameters and values in the csv file.
        _update_data_informations([stats, predictions], data_config)

        data_information = ""
        for parameter, parameter_value in varied_parameters.items():
            data_information += f"_{parameter}{parameter_value}"

        # To CSV
        stats_df = pd.DataFrame(columns=stats.keys(), data=stats)
        stats_df.to_csv(f'evaluation/stats{data_information}.csv', sep=';', decimal=',', index=True)
        predictions.to_csv(f'evaluation/predictions{data_information}.csv', sep=';', decimal=',', index=True)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Took {time.time() - start}s for f={f}")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
