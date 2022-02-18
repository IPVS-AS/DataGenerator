import argparse
import os
import random
import warnings
from itertools import product

from anytree import PreOrderIter, RenderTree

from SPH_CPI import SPHandCPI, SPH, RandomForestClassMethod, RandomForestBorutaMethod, CPI, \
    random_forest_parameters, KMeansClassification, GMMClassification, BirchClassification, RFperGroup

import pandas as pd
import numpy as np

from DataGenerator import ImbalanceGenerator

from Utility import train_test_splitting, update_data_and_training_data, get_train_test_X_y
from Hierarchy import HardCodedHierarchy, HIERARCHY_TYPES

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def store_data_to_csv(df_train, df_test, data_output_directory, run_id):
    df_train.to_csv(data_output_directory + f"/train_n_{run_id}.csv")
    df_test.to_csv(data_output_directory + f"/test_{run_id}.csv")


def run_machine_learning(gini_thresholds: list, p_quantile: list, max_info_loss_values: list, total_runs: int,
                         imbalance_degree: str = "normal", output_dir: str = "",
                         features_remove_percent_list=[0.2],):
    if imbalance_degree == 'all':
        imbalance_degrees = ImbalanceGenerator.imbalance_degrees
    else:
        imbalance_degrees = [imbalance_degree]
    runs = range(1, total_runs + 1, 1)
    ###############################################################

    print(n_features_list)

    for imbalance_degree in imbalance_degrees:
        # Results to capture as files
        acc_result_df = pd.DataFrame()
        detailed_predictions_df = pd.DataFrame()
        stats_result_df = pd.DataFrame()
        surrogates_df = pd.DataFrame()

        ##############################################################################################
        ################ Setting up output directories based on imbalance degree #####################
        # Default for directories, append the output_directory
        data_output_directory = f"data/"
        result_output_directory = f"results/"

        if not os.path.exists(data_output_directory):
            os.makedirs(data_output_directory)

        if not os.path.exists(result_output_directory):
            os.makedirs(result_output_directory)
        ##############################################################################################

        for n_samples in n_samples_list:
            n_train_samples = n_samples * (750 / 1050)
            for n_features in n_features_list:
                for features_remove_percent in features_remove_percent_list:
                    print(features_remove_percent)
                    for run_id in runs:
                        # Random seed for reproducibility
                        np.random.seed(run_id * 100)
                        random.seed(run_id * 100)

                        root_node = HardCodedHierarchy().create_hardcoded_hierarchy()

                        generator = ImbalanceGenerator(root=root_node,
                                                       imbalance_degree=imbalance_degree,
                                                       n_features=n_features,
                                                       n_samples_total=n_samples,
                                                       features_remove_percent=features_remove_percent)
                        data_df = generator.generate_data_with_product_hierarchy()
                        root_node = generator.root

                        # Train/Test split and update data in the hierarchy
                        df_train, df_test = train_test_splitting(data_df, n_train_samples=n_train_samples)
                        store_data_to_csv(df_train, df_test, data_output_directory, run_id)
                        root_node = update_data_and_training_data(root_node, df_train, data_df, n_features=n_features)
                        X_train, X_test, y_train, y_test = get_train_test_X_y(df_train, df_test, n_features=n_features)

                        # Dictionary of parameters for the different methods
                        methods_to_parameters = {
                            RFperGroup.name(): {"hierarchy": [root_node]},
                            RandomForestClassMethod.name(): {"classifier_params": [random_forest_parameters]},
                            RandomForestBorutaMethod.name(): {"classifier_params": [random_forest_parameters]},
                            KMeansClassification.name(): {"n_clusters": n_clusters_values},
                            GMMClassification.name(): {"n_components": n_clusters_values},
                            BirchClassification.name(): {"n_clusters": n_clusters_values},
                            SPH.name(): {"max_info_loss": max_info_loss_values, "hierarchy": [root_node]},
                            SPHandCPI.name(): {"max_info_loss": max_info_loss_values, "hierarchy": [root_node],
                                               "gini_threshold": gini_thresholds, "p_threshold": p_quantile, },
                            CPI.name(): {"gini_threshold": gini_thresholds, "p_threshold": p_quantile,
                                         "hierarchy": [root_node]},
                        }

                        for method in METHODS:
                            # Dictionary of parameters to use for each method, retrieve the one for this method
                            parameter_dicts = methods_to_parameters[method.name()]
                            for parameter_vals in product(*parameter_dicts.values()):
                                # 1.) Instantiate method to execute (SPH, SPHandCPI, ...)
                                method_instance = method(**dict(zip(parameter_dicts, parameter_vals)))

                                # 2.) Fit Method
                                method_instance.fit(X_train, y_train)

                                # 3.) Predict the test samples;
                                # No need to use the return value as we use the method_instance object to retrieve
                                # the results in a prettier format
                                method_instance.predict_test_samples(df_test)

                                # 4.) Retrieve accuracy Results (A@e and RA@e)
                                accuracy_per_e_df = method_instance.get_accuracy_per_e_df()
                                print(accuracy_per_e_df)

                                # 5.) We also get the detailed predictions if we want to make further analysis
                                predictions_df = method_instance.get_predictions_df()

                                # 6.) Save statistics about the dataset and the methods
                                # track statistics
                                method_instance.track_stats()
                                # retrieve method stats and store parameter values
                                method_stats_df = method_instance.get_stats_df()

                                surrogates = method_instance.get_surrogates_df()

                                for df in [predictions_df, method_stats_df, surrogates, accuracy_per_e_df]:
                                    # add n instances and features to each dataframe
                                    df["n"] = n_samples
                                    df["f"] = n_features
                                    df["mf"] = features_remove_percent

                                # Keep track of all statistics
                                detailed_predictions_df = pd.concat([detailed_predictions_df, predictions_df],
                                                                    ignore_index=True)
                                stats_result_df = pd.concat([stats_result_df, method_stats_df], ignore_index=True)
                                surrogates_df = pd.concat([surrogates_df, surrogates], ignore_index=True)
                                acc_result_df = pd.concat([acc_result_df, accuracy_per_e_df],
                                                          ignore_index=True)

                                print(accuracy_per_e_df)
                                print(method_stats_df)
                                print(surrogates_df)

                                # Todo: Maybe store each method run separately?! Instead of large csv files?
                                #  also store instances, features and hierarchy name in filename/directory?
                                acc_result_df.to_csv(result_output_directory + f"/{imbalance_degree}_accuracy.csv",
                                                     index=False)
                                stats_result_df.to_csv(result_output_directory + f"/{imbalance_degree}_stats.csv", index=False)
                                detailed_predictions_df.to_csv(result_output_directory + f"/{imbalance_degree}_predictions.csv",
                                                               index=False)

if __name__ == '__main__':

    ###############################################################
    ######################## Default Arguments ####################
    gini_thresholds = [
        0.2,
        0.25,
        0.3,
        0.35,
        0.4
    ]

    p_quantile = [
        0.7,
        0.75,
        0.8,
        0.85,
        0.9
    ]

    max_info_loss_values = [
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
        0.4]

    n_clusters_values = [10, 15, 20, 25, 26, 30, 35, 40, 45, 50]

    n_samples_list = [1050, 5000, 10000]
    n_features_list = [50, 100, 200]
    features_remove_percent_list = [0.2]

    # Machine learning algorithms to execute
    ###############################################################
    METHODS = [
        RFperGroup,
        RandomForestClassMethod,
        SPH,
        RandomForestBorutaMethod,
        KMeansClassification,
        GMMClassification,
        BirchClassification,
        SPHandCPI,
        CPI,
    ]

    ###############################################################
    ######################## Parse Arguments from CMD##############
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpi_free", help="Runs the cpi version that does not require additional parameters",
                        const=True, default=False, action='store_const')
    parser.add_argument("-imbalance", help="Degree of imbalance. This should either be 'low', 'normal' or 'high'.",
                        default='all', choices=ImbalanceGenerator.imbalance_degrees + ['all'])
    parser.add_argument('-info_loss', type=float,
                        help="Percentage of information loss to use. Default is 25 percent",
                        nargs='*',
                        required=False, default=max_info_loss_values)

    parser.add_argument('-gini', type=float,
                        help='Percentage of the threshold for the gini index. Per default, multiple values from 25 '
                             'to 40 in 5th steps are executed.',
                        nargs='*',
                        required=False, default=gini_thresholds)

    parser.add_argument('-p', type=float,
                        help='Percentage of the thresholds for the p_quantile. Per default, multiple values from 70 '
                             'to 95 in 5th steps are executed',
                        nargs='*',
                        required=False, default=p_quantile)

    parser.add_argument('-runs', type=int, help='Number of runs to perform. The runs differ in different seed values.',
                        required=False, default=1)

    parser.add_argument('-samples', type=int, nargs='*', help='Number of samples to generate with the data generation.',
                        required=False, default=n_samples_list)
    parser.add_argument('-features', type=int, nargs='*',
                        help='Number of samples to generate with the data generation.',
                        required=False, default=n_features_list)
    parser.add_argument('-methods', type=str, nargs='*', help="Methods to execute (SPH, CPI, SPHandCPI, RF, RF+B).",
                        default=METHODS)
    parser.add_argument('-hierarchy', type=str, help="Methods to execute (SPH, CPI, SPHandCPI, RF, RF+B).",
                        default="hardcoded", choices=HIERARCHY_TYPES)
    parser.add_argument('-output_dir', type=str, help="Name of the output directory where the results will be stored.",
                        default="")
    parser.add_argument('-missing_features', default=features_remove_percent_list, type=float, required=False,
                        nargs='*',
                        help="Fraction (0 to 1 ) of features to remove. The missing features percentage will be "
                             "in this are but will be a bit higher (~5%).")

    args = parser.parse_args()

    run_cpi_no_parameters = args.cpi_free

    max_info_loss_values = args.info_loss

    gini_thresholds = args.gini

    p_quantile = args.p

    n_samples_list = args.samples
    n_features_list = args.features
    hierarchy_type = args.hierarchy

    features_remove_percent = args.missing_features
    out_dir = args.output_dir
    imbalance_degree = args.imbalance
    total_runs = args.runs

    if imbalance_degree == 'all':
        imbalance_degrees = ImbalanceGenerator.imbalance_degrees
    else:
        imbalance_degrees = [imbalance_degree]
    runs = range(1, total_runs + 1, 1)
    ###############################################################

    ###############################################################
    ######### Run Machine Learning Magic ##########################
    run_machine_learning(
        gini_thresholds=gini_thresholds,
        p_quantile=p_quantile,
        max_info_loss_values=max_info_loss_values,
        total_runs=total_runs,
        imbalance_degree=imbalance_degree,
        output_dir=out_dir,
        features_remove_percent_list=features_remove_percent
    )
    ###############################################################
