import os
import sys
import time
from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from DataGenerator import ImbalanceGenerator
from sklearn.model_selection import train_test_split
from Hierarchy import EngineTaxonomy
from anytree import RenderTree
import numpy as np
import random
from pathlib import Path
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

    complexity_metrics = ["f1", "f1v", "n1", "n3",
                          #  "f2", "n2", "l3", "l1", "t1", "density", "c1", "c2",
                          ]
    CVI_s = ["sil", "ch", "vdu", "int", "pb", "vdb"]
    #####################################
    ### Complexity metrics  from PyMFE ###
    X = KNNImputer().fit_transform(X)
    ft = extract_mfes(X, y, complexity_metrics)

    for metric, value in zip(ft[0], ft[1]):
        print(f"{metric_mapping[metric]} (C): {value}")
        stats[f"{metric_mapping[metric]} (C)"] = value

    ### On Groups ###
    for group in groups:
        group_df = df[df["group"] == group]
        group_X, group_y = group_df[[f"F{i}" for i in range(f)]].to_numpy(), group_df["target"].to_numpy()
        group_X = KNNImputer().fit_transform(group_X)
        ft = extract_mfes(group_X, group_y, complexity_metrics)

        for metric, value in zip(ft[0], ft[1]):
            stats[f"{metric_mapping[metric]} (G)"] = value

    for key, value in stats.items():
        if "(G)" in key:
            stats[key] = [np.nanmean(value)]
    ######

    # ######################################
    # ### Gini #############################
    stats[f"Gini (C)"] = generator.gini(y)
    stats[f"Gini (G)"] = generator.gini(df["group"])
    # ######################################

    for key, value in stats.items():
        if not isinstance(value, list):
            stats[key] = [value]
    return stats


def _train_test_splitting(df, train_size=0.7):
    try:
        df_train, df_test = train_test_split(df, train_size=train_size, stratify=df[["group", "target"]])
    except ValueError as e:
        print(e)
        try:
            df_train, df_test = train_test_split(df, train_size=train_size, stratify=df["group"])
        except ValueError as e:
            print(e)
            try:
                df_train, df_test = train_test_split(df, train_size=train_size, stratify=df["target"])
            except ValueError as e:
                print(e)
                df_train, df_test = train_test_split(df, train_size=train_size)
    return df_train, df_test


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
        # group_X = KNNImputer().fit_transform(group_X)
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
    # Check for all given data Frames if they contain the data config
    return all([df.shape[0] > 0 for df in dfs]) \
           and all([(df[list(data_config)] == pd.Series(data_config)).all(axis=1).any() for df in dfs])


# Complexity metrics to measure
metric_mapping = {
    'c1': "Class entropy",
    "c2": "Imbalance Degree",
    "f1.mean": "Fishers DR",
    "f1v.mean": "Fishers DRv",
    "f2.mean": "Class Overlap",
    "n1": "Border Points",
    "n2.mean": "Inter/Intra Class Dist",
    "n3.mean": "NN Error",
    "l3.mean": "NL LC",
    "l1.mean": "Sum Error LP",
    "t1.mean": "Hyperspheres",
    "density": "Density",
    "int": "INT",
    "pb": "PB",
    "vdb": "VDB",
    "sil": "SIL",
    "vdu": "Dunn Index",
    "ch": "CHI",
}

if __name__ == '__main__':
    ####################################################################
    ## Define the possible configurations for the generated Datasets ###
    # Makes it a lot easier
    # --> Do not need that much for loops and automatically store configurations in csv files.
    varying_parameters = {
        "gs": [0, 0.05, 0.1, 0.15,
                0.25,
                0.5,
             0.75, 1.0
               ],
        "n_group_features": [1, 5,
                             10,
            15, 20, 25, 30, 40
                             ],
        "group_imbalance": [
            #'very_balanced', 'balanced',
            'medium',
            #'imbalanced',
            #'very_imbalanced'
        ],
        "cls_imbalance": [
            #'very_balanced', 'balanced',
            'medium',
            #'imbalanced', 'very_imbalanced'
        ]
    }

    default_parameter_setting = {
        "n": 1000,
        "n_features": 50,
        "c": 20,
        "gs": 0.5,
        "features_remove_percent": 0,
        "n_group_features": 10,
        "group_imbalance":
            'imbalanced',
        "cls_imbalance": 'medium',
    }
    #####################################################################

    #####################################################################
    ### Check for existing results ######################################
    if Path("stats.csv").is_file() and Path("predictions.csv").is_file():
        stats_df = pd.read_csv("stats.csv", sep=';', decimal=',', index_col=0)
        pred_df = pd.read_csv("predictions.csv", sep=';', decimal=',', index_col=0)
    else:
        stats_df = pd.DataFrame()
        pred_df = pd.DataFrame()
    #####################################################################

    gini_df = pd.DataFrame()
    # Do this if we want to iterate over all possible combinations
    # for data_config_values in product(*data_configs.values()):
    for parameter, parameter_values in varying_parameters.items():
        for parameter_value in parameter_values:
            data_config = default_parameter_setting.copy()
            data_config.update({parameter: parameter_value})
            print('---------------------------------')
            if _result_available([stats_df, pred_df], data_config):
                print(f"skipping config {data_config}")
                continue

            print('---------------------------------')
            print(f"Running with config:")
            print(f'{data_config}')
            start = time.time()
            f = data_config["n_features"]
            generator = ImbalanceGenerator(hardcoded=False,
                                           class_overlap=1.5,
                                           root=EngineTaxonomy().create_taxonomy(),
                                           **data_config)
            df = generator.generate_data_with_product_hierarchy()
            df = df.drop("index", axis=1)
            df = df.dropna(how="any")

            print(f"Rows with missing values: {df.shape[0] - df.dropna().shape[0]}")
            X, y = df[[f"F{i}" for i in range(f)]].to_numpy(), df["target"].to_numpy()
            groups = np.unique(df["group"].to_numpy())

            # dataset_gini_df = pd.DataFrame()
            # gini_groups = generator.gini(df["group"])
            # print(gini_groups)
            # dataset_gini_df["Gini (G)"] = [gini_groups]
            # print(dataset_gini_df)
            # _update_data_informations([dataset_gini_df], data_config)
            # gini_df = pd.concat([gini_df, dataset_gini_df])
            # print(gini_df)
            # gini_df.to_csv("gini_groups.csv", sep=';', decimal=',', index=True)

            # Statistics
            stats = extract_statistics_from_data(X, y)
            predictions = pd.DataFrame()
            # accuracy
            accuracy_x, accuracy_groups, predictions = calculate_accuracy()
            stats["Acc (X)"] = accuracy_x
            stats["Acc (G)"] = accuracy_groups
            stats["Acc (G - X)"] = accuracy_groups - accuracy_x

            # update data config to dataframes
            _update_data_informations([stats,
                                       predictions
                                       ], data_config)

            # To CSV
            stats_df = pd.concat([stats_df, pd.DataFrame(columns=stats.keys(), data=stats)])
            stats_df.to_csv('stats.csv', sep=';', decimal=',', index=True)
            # pred_df = pd.concat([pred_df, predictions])
            pred_df.to_csv(f'predictions.csv', sep=';', decimal=',', index=True)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Took {time.time() - start}s for f={f}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

