import argparse
import os
import random
import warnings
from itertools import product

from ClassificationPartitioningMethods import SPHandCPI, SPH, RandomForestClassMethod, RandomForestBorutaMethod, CPI, \
    random_forest_parameters

import pandas as pd
import numpy as np

from DataGenerator import ImbalanceGenerator
from Utility import train_test_splitting, update_data_and_training_data
from Hierarchy import HardCodedHierarchy


def get_train_test_X_y(df_train, df_test, n_features=100):
    X_train = df_train[[f"F{i}" for i in range(n_features)]].to_numpy()
    y_train = df_train["target"].to_numpy()

    X_test = df_test[[f"F{i}" for i in range(n_features)]].to_numpy()
    y_test = df_test["target"].to_numpy()
    return X_train, X_test, y_train, y_test


def store_data_to_csv(df_train, df_test, data_output_directory):
    df_train.to_csv(data_output_directory + "/train.csv")
    df_test.to_csv(data_output_directory + "/test.csv")


if __name__ == '__main__':
    from sklearn.exceptions import UndefinedMetricWarning

    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    ###############################################################
    ######################## Default Arguments ####################
    gini_thresholds = [
        # 0.2,
        # 0.25,
        # 0.3,
        # 0.35,
        0.4
    ]

    p_quantile = [
        # 0.7,
        # 0.75,
        # 0.8,
        # 0.85,
        0.9
    ]

    max_info_loss_values = [
        # 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
        0.4]

    # Machine learning algorithms to execute
    ###############################################################
    METHODS = [
        SPHandCPI,
        SPH,
        RandomForestClassMethod,
        RandomForestBorutaMethod,
        CPI,
    ]

    ###############################################################
    ######################## Parse Arguments from CMD##############
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpi_free", help="Runs the cpi version that does not require additional parameters",
                        const=True, default=False, action='store_const')
    parser.add_argument("-imbalance", help="Degree of imbalance. This should either be 'low', 'normal' or 'high'.",
                        default='normal', choices=ImbalanceGenerator.imbalance_degrees + ['all'])
    parser.add_argument('-max_info_loss', type=float,
                        help="Percentage of information loss to use. Default is 25 percent",
                        nargs='*',
                        required=False, default=max_info_loss_values)

    parser.add_argument('-gini_thresholds', type=float,
                        help='Percentage of the threshold for the gini index. Per default, multiple values from 25 '
                             'to 40 in 5th steps are executed.',
                        nargs='*',
                        required=False, default=gini_thresholds)

    parser.add_argument('-p_thresholds', type=float,
                        help='Percentage of the thresholds for the p_quantile. Per default, multiple values from 70 '
                             'to 95 in 5th steps are executed',
                        nargs='*',
                        required=False, default=p_quantile)

    parser.add_argument('-runs', type=int, help='Number of runs to perform. The runs differ in different seed values.',
                        required=False, default=1)
    parser.add_argument('-methods', type=str, help="Methods to execute (SPH, CPI, SPHandCPI, RF, RF+B).",
                        default=METHODS)
    args = parser.parse_args()

    imbalance_degree = args.imbalance
    if imbalance_degree == 'all':
        imbalance_degrees = ImbalanceGenerator.imbalance_degrees
    else:
        imbalance_degrees = [imbalance_degree]

    run_cpi_no_parameters = args.cpi_free

    max_info_loss_values = args.max_info_loss

    gini_thresholds = args.gini_thresholds

    p_quantile = args.p_thresholds

    total_runs = args.runs
    runs = range(1, total_runs + 1, 1)
    ###############################################################

    for imbalance_degree in imbalance_degrees:
        ##############################################################################################
        ################ Setting up output directories based on imbalance degree #####################
        # Default for directories
        output_directory = f"test/imbalance_degree/{imbalance_degree}/"
        data_output_directory = f"{output_directory}/data"
        result_output_directory = f"{output_directory}/result"

        if not os.path.exists(data_output_directory):
            os.makedirs(data_output_directory)

        if not os.path.exists(result_output_directory):
            os.makedirs(result_output_directory)
        ##############################################################################################

        acc_result_df = pd.DataFrame()
        detailed_predictions_df = pd.DataFrame()
        stats_result_df = pd.DataFrame()

        for run_id in runs:
            # maybe change back to 5?
            np.random.seed(run_id * 10)
            random.seed(run_id * 10)

            root_node = HardCodedHierarchy().create_hardcoded_hierarchy()
            data_df = ImbalanceGenerator().generate_data_with_product_hierarchy(root=root_node,
                                                                                imbalance_degree=imbalance_degree)

            # Train/Test split and update data in the hierarchy
            df_train, df_test = train_test_splitting(data_df)
            store_data_to_csv(df_train, df_test, data_output_directory)
            root_node = update_data_and_training_data(root_node, df_train, data_df, n_features=100)
            X_train, X_test, y_train, y_test = get_train_test_X_y(df_train, df_test, n_features=100)

            # Dictionary of parameters for the different methods
            methods_to_parameters = {SPH.name(): {"max_info_loss": max_info_loss_values, "hierarchy": [root_node]},
                                     SPHandCPI.name(): {"max_info_loss": max_info_loss_values,
                                                        "gini_threshold": gini_thresholds, "p_threshold": p_quantile,
                                                        "hierarchy": [root_node]},
                                     CPI.name(): {"gini_threshold": gini_thresholds, "p_threshold": p_quantile},
                                     RandomForestClassMethod.name(): {"classifier_params": [random_forest_parameters]},
                                     RandomForestBorutaMethod.name(): {"classifier_params": [random_forest_parameters]}}

            for method in METHODS:
                # Dictionary of parameters to use for each method, retrieve the one for this method
                parameter_dicts = methods_to_parameters[method.name()]
                for parameter_vals in product(*parameter_dicts.values()):
                    # 1.) Instantiate method to execute (SPH, SPHandCPI, ...)
                    method_instance = method(**dict(zip(parameter_dicts, parameter_vals)))

                    # 2.) Fit Method
                    method_instance.fit(X_train, y_train)

                    # 3.) Predict the test samples; No need to use the return value as we use the method_instance object
                    # to retrieve the results in a prettier format
                    method_instance.predict_test_samples(df_test)

                    # 4.) Retrieve accuracy Results (A@e and RA@e)
                    accuracy_per_e_df = method_instance.get_accuracy_per_e_df(run_id)
                    print(accuracy_per_e_df)
                    acc_result_df = pd.concat([acc_result_df, accuracy_per_e_df],
                                              ignore_index=True)

                    # 5.) We also get the detailed predictions if we want to make further analysis
                    predictions_df = method_instance.get_predictions_df(run_id)
                    detailed_predictions_df = pd.concat([detailed_predictions_df, predictions_df], ignore_index=True)

                    # 6.) Save statistics about the dataset and the methods
                    # track statistics
                    method_instance.track_stats()
                    # retrieve method stats and store parameter values
                    method_stats_df = method_instance.get_stats_df()
                    # Keep track of all statistics
                    stats_result_df = pd.concat([stats_result_df, method_stats_df], ignore_index=True)

                    print(accuracy_per_e_df)
                    print(method_stats_df)

        # Todo: Maybe store each method run separately?! Instead of large csv files?
        acc_result_df.to_csv(result_output_directory + "/accuracy.csv", index=False)
        stats_result_df.to_csv(result_output_directory + "/stats.csv", index=False)
        stats_result_df.to_csv(result_output_directory + "/predictions.csv", index=False)
