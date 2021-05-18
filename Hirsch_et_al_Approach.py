import os
import random
from collections import Counter

import pandas as pd
import numpy as np
from anytree import PreOrderIter, RenderTree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import jaccard_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from DataGenerator import ImbalanceGenerator

import matplotlib.pyplot as plt
import concentrationMetrics as cm

# matplotlib.use('Agg')
from Hierarchy import HardCodedHierarchy

np.random.seed(5)


def theil(x):
    my_index = cm.Index()
    class_frequencies = np.array(list(Counter(x).values()))
    return my_index.theil(class_frequencies)


def gini(x):
    my_index = cm.Index()
    class_frequencies = np.array(list(Counter(x).values()))
    return my_index.gini(class_frequencies)


def SPH(root_node, min_samples_per_class=1, max_info_loss=0.25):
    """
    Runs the SPH step of Hirsch et al.
    First, we start at all leave nodes (product groups).
    Here, all classes that have equal or less than min_samples_per_class are removed.
    After this, it is checked if more than max_info_loss percent of samples are removed.
    If yes, we use the surrogate subset, i.e., we use the parent node and apply again our checks on them.
    :param root_node: The root node of the hierarchy that represent the whole tree (the whole tree can be navigated with
     the root node).
    :param min_samples_per_class: number of minimum samples per class. If a class has less than this samples,
     the class is removed with its samples.
    :param max_info_loss: maximum information loss that is still be ok when removing classes and samples.
    If we remove more than this threshold, we go up in the hierarchy.
    :return: partitions_for_node, i.e., a dictionary that has the node_id as key and the actual node as value. The data
    that is used after sph is stored in the nodes (actually the data is still stored in the leave nodes).
    """

    # outputs are whole_dataset partitions
    # -> we just return partitions of the nodes, i.e., a dictionary with node id as key and node as value
    # the node will contain the data for sph
    partitions_for_node = {}

    # get leave nodes
    product_group_nodes = [node for node in PreOrderIter(root_node) if not node.children]
    sph_executed = 0

    while len(product_group_nodes) > 0:
        group_node = product_group_nodes.pop(0)
        node_id = group_node.node_id
        check_passed = False

        # node to traverse, if checks passed we go up in the hierarchy
        traverse_node = group_node
        while not check_passed:
            # check that SPH
            group_labels = traverse_node.training_labels
            group_data = traverse_node.training_data
            check_passed, sph_data, sph_labels, info_loss = SPH_checks(group_data, group_labels,
                                                                       min_samples_per_class,
                                                                       max_info_loss)
            if not check_passed:
                sph_executed += 1
            if not check_passed and traverse_node.parent:
                print(f"Using surrogate for {traverse_node.node_id} with info loss {info_loss}")
                traverse_node = traverse_node.parent
            else:
                # if  np.unique(group_sample_labels):
                group_node.sph_data = sph_data
                group_node.sph_labels = sph_labels
                partitions_for_node[node_id] = group_node
    return partitions_for_node, sph_executed


def SPH_checks(group_data, group_labels, min_samples_per_class=1, max_info_loss=0.25):
    ##############################################################################
    ######## First step: Remove samples, where the class occurs only once ########
    # get samples, lables for this product group
    group_samples = np.array(group_data)
    group_labels = np.array(group_labels)
    original_n_samples = len(group_samples)

    # get for each class how often it occurs in this product group
    class_count = Counter(group_labels)
    indices = []
    for index, label in enumerate(group_labels):
        if class_count[label] > min_samples_per_class:
            indices.append(index)
    min_c_group_labels = np.take(group_labels, indices, axis=0)
    min_c_group_samples = np.take(group_samples, indices, axis=0)

    assert len(min_c_group_samples) == len(indices)
    assert len(min_c_group_labels) == len(indices)

    ####### First step finished ###################################################
    ###############################################################################

    ###############################################################################
    ###### Second step: Check that two classes available ##########################
    # check that two classes are available after class removal
    two_class_avail = len(np.unique(min_c_group_labels)) >= min_samples_per_class
    ###### Second step Finished ###################################################
    ###############################################################################

    ###############################################################################
    ###### Third step: Check that info loss less than 25% #########################
    info_loss = 1 - (len(min_c_group_samples) / original_n_samples)
    less_info_loss = info_loss < max_info_loss
    ####### Third step Finished ###################################################
    ##############################################################################

    check_passed = two_class_avail and less_info_loss
    return check_passed, min_c_group_samples, min_c_group_labels, info_loss


def run_cpi_without_threshold(partition_labels, concentration_function=gini):
    """
    Runs a version of the Class Partitioning according to Imbalance (CPI) that does not require any parameters.
    :param partition_labels: class labels of the partitions.
    :param concentration_function: function for measuring class imbalance degree. per default, gini is used.
    :return: indices of the data/labels that should be used after applying the partitioning. This is either a list of
    indices, meaning there is no partitionign done, or it is a tuple, where the first entry is the
     minority class indices and the second entry are the majortiy class indices.
    """
    cm_no_cpi = concentration_function(partition_labels)
    print(f"gini index with no further partitioning is: {cm_no_cpi}")

    indices_per_partitioning = []
    avg_cms_per_partitioning = []
    weighted_avg_cms_per_partitioning = []

    old_gini = cm_no_cpi

    # we do not want to take the last two minority classes!
    for i in range(0, n_classes - 2):

        class_counter = Counter(partition_labels)

        # find the next highest class --> To make this more efficient we could do this only for minority classes
        n_max_classes = sorted(class_counter.values(), reverse=True)[i]

        # divide according to max class number
        minority_indices = [ind for ind, label in enumerate(partition_labels) if
                            class_counter[label] < n_max_classes]

        # should be the same as all data without minority classes
        majority_indices = [ind for ind, label in enumerate(partition_labels)
                            if class_counter[label] >= n_max_classes]

        if len(minority_indices) == 0:
            break

        minority_labels = partition_labels[minority_indices]
        print(f"minority_labels with counts are: {Counter(minority_labels)}")
        majority_labels = partition_labels[majority_indices]
        print(f"majority_labels with counts are: {Counter(majority_labels)}")

        if len(np.unique(majority_labels)) == 1:
            # add minority set to majority set again and relabel minority to "-1"
            # majority_partition = np.concatenate([majority_partition, minority_partition])
            new_minority_labels = np.array([-1 for x in minority_labels])
            cm_majority = concentration_function(np.concatenate([majority_labels, new_minority_labels]))
        else:
            cm_majority = concentration_function(majority_labels)

        cm_minority = concentration_function(minority_labels)
        avg_cm = (cm_minority + cm_majority) / 2

        """
        # Greedy procedure, do not consider it now
        if avg_cm > old_gini:
            print(f'Gini index increas from {old_gini} to {avg_cm}, so break!')
            print('----------------------------------------------------------')
            break
            
        old_gini = avg_cm
        """

        weighted_avg_cm = (cm_minority * len(minority_labels) + cm_majority * len(majority_labels)) / len(
            partition_labels)

        print(f"gini index for minority is: {cm_minority}")
        print(f"gini index for majority is: {cm_majority}")
        print(f"average gini index is: {avg_cm}")
        print(f"weighted average gini index is: {weighted_avg_cm}")
        print('---------------------------------------------------')

        avg_cms_per_partitioning.append(avg_cm)
        weighted_avg_cms_per_partitioning.append(weighted_avg_cm)
        indices_per_partitioning.append((minority_indices, majority_indices))

    final_indices = []
    min_cm_value = cm_no_cpi

    for i, avg_cm in enumerate(avg_cms_per_partitioning):
        if avg_cm < min_cm_value:
            final_indices = indices_per_partitioning[i]
            min_cm_value = avg_cm
    return final_indices


def run_cpi_free_in_isolation(X_train, y_train, df_test):
    """
    Runs the CPI version without parameters on the whole dataset.
    :param X_train: Training data (should be a np.array)
    :param y_train: Training labels, should be a 1d numpy array with len(y) == X.shape[0]
    :param df_test: dataframe that contains the test data
    :return: Dataframe that contains the accuracy for different lengths of the reccomendation list.
    """
    final_indices = run_cpi_without_threshold(y_train)

    if isinstance(final_indices, tuple):
        minority_indices = final_indices[0]
        majority_indices = final_indices[1]

        # take minority, majority data based on indices. Note that axis=0 specifies to take the row indices!
        minority_partition = np.take(X_train, minority_indices, axis=0)  # minority_data
        majority_partition = np.take(X_train, majority_indices, axis=0)  # majority data

        print("class counter Minority: {}".format(Counter(np.take(y_train, minority_indices, axis=0))))
        print("class counter Majority: {}".format(Counter(np.take(y_train, majority_indices, axis=0))))

        minority_labels = np.take(y_train, minority_indices, axis=0)  # minority labels
        majority_labels = np.take(y_train, majority_indices, axis=0)  # majority labels

        if len(np.unique(majority_labels)) == 1:
            majority_partition = np.concatenate([minority_partition, majority_partition])
            majority_labels = np.concatenate([majority_labels, [-1 for x in minority_labels]])

        minority_estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                       ("forest", RandomForestClassifier(random_state=1234,
                                                                         n_estimators=100))])

        minority_estimator.fit(minority_partition, minority_labels)

        majority_estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                       ("forest", RandomForestClassifier(random_state=1234,
                                                                         n_estimators=100))])
        majority_estimator.fit(majority_partition, majority_labels)

        # pseudo model_repo for each group! Makes it easier to get accuracy etc.
        model_repo = {group: (minority_estimator, majority_estimator) for group in set(df_test['group'].values)}

        # classifiy new samples based on repository
        predicted_probabilities, score = classifiy_new_samples(model_repo, df_test, n_features=n_features)

        # get ranked list
        ranked_list = obtain_ranked_list_R(predicted_probabilities)

        # get accuracy by length e, default e=10
        cpi_isolation_accuracy, cpi_isolation_accuracy_df = accuracy_for_e(ranked_list)

    else:
        # same result as RF --> but should actually not appear
        cpi_isolation_accuracy_df = rf_df
        cpi_isolation_accuracy_df['Method'] = 'CPI free'

    return cpi_isolation_accuracy_df


def CPI_no_threshold(node_per_node_id):
    """
    Runs CPI without required parameters for each of the product groups, i.e., for each leave node in the tree.
    This should be containd in the node_per_node_id dictionary.
    :param node_per_node_id: Dictionary that contains the node_id as key and the corresponding product group/
    leave node as value.
    :return: A dictionary that contains the node_id as keys and the modified product group nodes as values. Here,
    the product group nodes now have cpi_data and cpi_labels set.
    """
    cpi_partitions_per_node_id = {}
    for node_id, node in node_per_node_id.items():
        print('---------------------------------------------------')
        print(f'Node: {node_id}')
        node = node_per_node_id[node_id]

        partition_labels = node.sph_labels

        final_indices = run_cpi_without_threshold(partition_labels)

        # check if partitioned into min/majority
        if isinstance(final_indices, tuple):
            # if yes, upadate the data
            min_indices = final_indices[0]
            maj_indices = final_indices[1]

            node.cpi_data = (
                np.take(node.sph_data, min_indices, axis=0),  # minority_data
                np.take(node.sph_data, maj_indices, axis=0)  # majority data
            )

            node.cpi_labels = (
                np.take(node.sph_labels, min_indices, axis=0),  # minority labels
                np.take(node.sph_labels, maj_indices, axis=0)  # majority labels
            )
        else:
            # if not, use the data as cpi data
            node.cpi_data = node.sph_data
            node.cpi_labels = node.sph_labels

        cpi_partitions_per_node_id[node_id] = node
    return cpi_partitions_per_node_id


def run_cpi(partition_labels, concentration_function, cm_threshold, p_threshold):
    ############# Detector: Check if gini threshold reached ###################################################
    # We now call this concentration measure (cm), as it could also be theil, shannon, ... index
    cm_value = concentration_function(partition_labels)

    if cm_value > cm_threshold:
        ######### If yes then divsior is executed, i.e., partition to minority and majority sampels ###########
        class_counter = Counter(partition_labels)
        print("partition min/majority")
        print(class_counter)
        class_freq = np.array(list(class_counter.values()))
        # calculate q quantile of classes
        class_threshold = np.quantile(class_freq, p_threshold)

        # divide according to q quantile value the partition into minority and majority samples
        minority_indices = [ind for ind, label in enumerate(partition_labels) if
                            class_counter[label] <= class_threshold]
        # should be the same as all data without minority classes
        majority_indices = [ind for ind, _ in enumerate(partition_labels) if ind not in minority_indices]
        print("class threshold: {}".format(class_threshold))

        # assert that we still have the same amount of samples and labels
        assert len(minority_indices) + len(majority_indices) == len(partition_labels)
        return minority_indices, majority_indices

    else:
        return partition_labels


def CPI(node_per_node_id, cm_threshold=0.3, p_threshold=0.7, concentration_function=gini):
    """
    Runs CPI with the given thresholds for each of the product groups, i.e., for each leave node in the tree.
    This should be containd in the node_per_node_id dictionary.
    :param node_per_node_id: Dictionary that contains the node_id as key and the corresponding product group/
    leave node as value.
    :param cm_threshold: Threshold for the concentration function
    :param p_threshold: Threshold of the p-quantile.
    :param concentration_function: Concentration function to use, default is gini index.
    :return: A dictionary that contains the node_id as keys and the modified product group nodes as values. Here,
    the product group nodes now have cpi_data and cpi_labels set.
    """
    cpi_partitions_per_node_id = {}
    for node_id, node in node_per_node_id.items():
        node = node_per_node_id[node_id]

        partition_labels = node.sph_labels

        cpi_indices = run_cpi(partition_labels, concentration_function, cm_threshold, p_threshold)

        if isinstance(cpi_indices, tuple):
            minority_indices = cpi_indices[0]
            majority_indices = cpi_indices[1]

            if minority_indices and majority_indices:
                # take minority, majority data based on indices. Note that axis=0 specifies to take the row indices!
                node.cpi_data = (
                    np.take(node.sph_data, minority_indices, axis=0),  # minority_data
                    np.take(node.sph_data, majority_indices, axis=0)  # majority data
                )

                print("class counter Minority: {}".format(Counter(np.take(node.sph_labels, minority_indices, axis=0))))
                print("class counter Majority: {}".format(Counter(np.take(node.sph_labels, majority_indices, axis=0))))

                node.cpi_labels = (
                    np.take(node.sph_labels, minority_indices, axis=0),  # minority labels
                    np.take(node.sph_labels, majority_indices, axis=0)  # majority labels
                )

            else:
                node.cpi_data = node.sph_data
                node.cpi_labels = node.sph_labels
        else:
            node.cpi_data = node.sph_data
            node.cpi_labels = node.sph_labels

        cpi_partitions_per_node_id[node_id] = node

    return cpi_partitions_per_node_id


def preprocessing(cpi_partitions):
    """
    Preprocesses the data according to hirsch et al.
    In particular, this means OvA is applied for partitions where we only have one class.

    :param cpi_partitions:  Dictionary that contains the node_id as key and the corresponding product group/
    leave node as value.
    :return: A dictionary that contains the node_id as keys and the modified product group nodes as values. Here,
    the cpi_data and labels may be modified with the OvA binarization.
    """
    resulting_partitions = {}
    # structural manipulation
    for key, node in cpi_partitions.items():
        if len(node.cpi_data) == 2 and isinstance(node.cpi_data, tuple):
            # we have division into minority and majority classes
            minority_partition = node.cpi_data[0]

            majority_partition = node.cpi_data[1]

            minority_labels = node.cpi_labels[0]
            majority_labels = node.cpi_labels[1]
            majority_label_count = len(np.unique(majority_labels))

            # check if majority set contains only one class
            if majority_label_count == 1:
                # add minority set to majority set again and relabel minority to "-1"
                majority_partition = np.concatenate([majority_partition, minority_partition])
                new_minority_labels = np.array([-1 for x in minority_labels])
                majority_labels = np.concatenate([majority_labels, new_minority_labels])

            # update cpi data and labels accordingly
            node.cpi_data = (minority_partition, majority_partition)
            node.cpi_labels = (minority_labels, majority_labels)
            resulting_partitions[key] = node

        else:
            node.cpi_data = np.array(node.cpi_data)
            resulting_partitions[key] = node
    return resulting_partitions


def train_test_splitting(df, n_train_samples=750, at_least_two_samples=True):
    """
    Performs our custom train test splitting.
    This highly affects the accuracy of this approach.

    :param df: Dataframe of the whole data.
    :param n_train_samples: number of samples to use for training
    :param at_least_two_samples: check if we should have at least one sample in the training data for each group.
    :return: train, test: dataframes that contain training and test data.
    """
    # split in 750 training samples
    # 300 test samples
    train_percent = n_train_samples / len(df)

    n_classes = len(np.unique(df["target"].to_numpy()))

    # split with stratify such that each class occurs in train and test set
    train, test = train_test_split(df, train_size=train_percent, random_state=1234,
                                   stratify=df["target"]
                                   )
    n_not_in_train = 1

    while at_least_two_samples and n_not_in_train > 0:
        test['freq'] = test.groupby('group')['target'].transform('count')
        train['freq'] = train.groupby('group')['target'].transform('count')

        print(f"Test data contains: {Counter(zip(test['target'].to_numpy(), test['group'].values))}")
        # check which classes occur once and do not occur on training data!
        # Use counter for this of group and target, in both training and test set
        train_counter = Counter(zip(train['target'].to_numpy(), train['group'].values))

        print(f"test of length before: {len(test)}")
        # mark which ones occur not in training set (group and target) but occur in test set
        test['marker'] = test.apply(lambda row: train_counter[(row['target'], row['group'])] == 0, axis=1)
        test_in_train = test[~test['marker']]
        test_not_in_train = test[test['marker']]
        print(f"length of test afterwards: {len(test_in_train)}")

        n_not_in_train = len(test_not_in_train)
        print(n_not_in_train)

        # check if there is still a class that occurs in test but not in training
        if n_not_in_train > 0:
            # if yes, replace these samples with random samples from training set
            print(f"Classes that occur once in test but not in training data: {n_not_in_train}")
            drop_indices = np.random.choice(train.index, n_not_in_train)
            train_subset = train.loc[drop_indices]
            train = train.drop(drop_indices)

            train = train.append(test_not_in_train)
            test = test_in_train.append(train_subset)

        test.drop(['freq'], axis=1)
        train.drop(['freq'], axis=1)

    y_train_classes = len(np.unique(train["target"].to_numpy()))
    y_test_classes = len(np.unique(test["target"].to_numpy()))

    print(y_train_classes)
    print(y_test_classes)
    # make sure train_classes = test_classes = n_classes
    if y_test_classes < n_classes:
        print(f"Classes that do no occur in test set: {[x for x in range(84) if x not in test['target'].to_numpy()]}")
    #    assert y_test_classes == n_classes and y_train_classes == n_classes
    # if len(test[test['freq'] == 1]) == 0:
    return train, test


def update_data_and_training_data(root_node, df_train, data_df, n_features=100):
    product_group_nodes = [node for node in PreOrderIter(root_node) if not node.children]

    for group_node in product_group_nodes:
        # we have leave node so set training data
        node_id = group_node.node_id

        # get training data and training labels as numpy arrays
        train_data = df_train[df_train["group"] == node_id][[f"F{i}" for i in range(n_features)]].to_numpy()
        training_labels = df_train[df_train["group"] == node_id]["target"].to_numpy()

        data = data_df[data_df["group"] == node_id][[f"F{i}" for i in range(n_features)]].to_numpy()
        labels = data_df[data_df["group"] == node_id]["target"].to_numpy()

        group_node.training_data = train_data
        group_node.training_labels = training_labels

        # also set the "normal" data and labels
        # These are more than the training data and in case we already created a dataset, we want to set them as well
        group_node.data = data
        group_node.target = labels

        # print(f"Length of training data: {len(train_data)} and length of group node training data: {len(group_node.training_data)}")

        # pass training data upwards the whole tree
        traverse_node = group_node
        while traverse_node.parent:
            traverse_node = traverse_node.parent

            if traverse_node.training_data is not None:
                traverse_node.training_data = np.concatenate([traverse_node.training_data, train_data])
                traverse_node.training_labels = np.concatenate([traverse_node.training_labels, training_labels])
            else:
                traverse_node.training_data = train_data
                traverse_node.training_labels = training_labels
    return root_node


def create_ensemble(cpi_partitions):
    model_repository = {}
    # use random forest on product groups
    for key, node in cpi_partitions.items():

        if len(node.cpi_data) == 2 and isinstance(node.cpi_data, tuple):
            # minority and majority sets
            minority_partition = node.cpi_data[0]
            majority_partition = node.cpi_data[1]

            minority_labels = node.cpi_labels[0]
            majority_labels = node.cpi_labels[1]
            # train majority and minority learners
            minority_estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                           ("forest", RandomForestClassifier(random_state=1234,
                                                                             n_estimators=100))])

            minority_estimator.fit(minority_partition, minority_labels)

            majority_estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                           ("forest", RandomForestClassifier(random_state=1234,
                                                                             n_estimators=100))])
            majority_estimator.fit(majority_partition, majority_labels)
            model_repository[node.node_id] = (minority_estimator, majority_estimator)
        else:
            estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                  ("forest", RandomForestClassifier(random_state=1234,
                                                                    n_estimators=100))])
            estimator.fit(node.cpi_data, node.cpi_labels)
            model_repository[node.node_id] = estimator
    # return a model_repository (key is node id, value is model)
    return model_repository


def classifiy_new_samples(model_repo, test_data_df, n_features=100, method="SPH+CPI"):
    actual_labels = test_data_df["target"].to_numpy()
    predicted_labels = []

    # we store the probabilities for each sample in a list (same order as the samples) of dictionaries
    # each dictionary contains either the probabilies or the probabilities for the minority and for majority classes
    # also, the classes and the actual class of the sample are stored
    predicted_probabilities = []
    test_data = test_data_df[[f"F{i}" for i in range(n_features)]].to_numpy()
    # test_data = KNNImputer(missing_values=np.nan).fit_transform(test_data)

    for i, (_, row) in enumerate(test_data_df.iterrows()):
        y_target = row["target"]

        sample = test_data[i].reshape(1, -1)
        clf = model_repo[row["group"]]

        if isinstance(clf, tuple):
            minority_clf = clf[0]
            majority_clf = clf[1]
            minority_probas = minority_clf.predict_proba(sample)[0]

            majority_probas = majority_clf.predict_proba(sample)[0]
            majority_probas = [x for x in majority_probas if x != -1]
            majority_classes = majority_clf["forest"].classes_
            # filter out probs and class for -1
            maj_prob_classes = list(zip(majority_probas, majority_classes))
            maj_prob_classes = [x for x in maj_prob_classes if x[1] != -1]

            majority_probas = [x[0] for x in maj_prob_classes]
            majority_classes = [x[1] for x in maj_prob_classes]

            max_minority = max(minority_probas)
            max_majority = max(majority_probas)
            #print(f"For target class {y_target} the maximum is")
            if max_minority > max_majority:
                predicted_label = minority_clf.predict(sample)
                #print(f"minority with {max_minority} and predicted {predicted_label}")
            else:
                predicted_label = majority_clf.predict(sample)
                #print(f"minority with {max_majority} and predicted {predicted_label}")

            predicted_probabilities.append({"minority_probabilities": minority_probas,
                                            "minority_classes": minority_clf["forest"].classes_,
                                            "majority_probabilities": majority_probas,
                                            "majority_classes": majority_classes,
                                            "target": y_target})
            predicted_labels.append(predicted_label)
        else:

            probabilities = clf.predict_proba(sample)[0]
            predicted_label = clf.predict(sample)
            predicted_labels.append(predicted_label)
            predicted_probabilities.append(
                {"probabilities": probabilities, "classes": clf["forest"].classes_, "target": y_target})

    df_test["predicted"] = predicted_labels
    # check if predicted == equal
    df_test["equal"] = df_test["predicted"] == df_test["target"]
    df_test["equal"] = df_test.apply(lambda row: int(row["equal"]), axis=1)

    acc_per_group_df = pd.DataFrame()
    if os.path.isfile("acc_per_group.csv"):
        acc_per_group_df = pd.concat([pd.read_csv("acc_per_group.csv", sep=';', decimal=',', index_col=None),
                                      acc_per_group_df])

    print("--------------------------------------------------------------------------------------------------------")
    print("Classes per group:")
    counter = Counter(zip(df_test['target'].to_numpy(), df_test['group'].values))
    #dict = {"group": [k[1] for k in counter.keys()], "target": [k[0] for k in counter.keys()],
    #        "count": [counter[k] for k in counter.keys()]}
    #class_per_group_df = pd.DataFrame.from_dict(dict)
    #class_per_group_df = class_per_group_df.sort_values(["group", "target"])
    #class_per_group_df = class_per_group_df.reset_index()
    #print(class_per_group_df)

    train_class_counter = Counter(df_train["target"].to_numpy())
    print("accuracy per group:")
    accuracy_per_group = df_test.groupby(['target'])["equal"].mean()
    accuracy_per_group = accuracy_per_group.reset_index()
    accuracy_per_group = accuracy_per_group.sort_values(["target"])

    #accuracy_per_group["count"] = class_per_group_df["count"]
    accuracy_per_group["train count"] = accuracy_per_group["target"].apply(lambda x: train_class_counter[x])
    accuracy_per_group["Method"] = method
    print(accuracy_per_group)
    acc_per_group_df = pd.concat([accuracy_per_group, acc_per_group_df])
    acc_per_group_df.to_csv("acc_per_group.csv", sep=';', decimal=',', index=False)

    print("--------------------------------------------------------------------------------------------------------")

    # exit()
    correct_classified = [pred_l[0] for act_l, pred_l in zip(actual_labels, predicted_labels) if act_l == pred_l[0]]
    print(f"Correctly classified: {Counter(correct_classified)}")

    return predicted_probabilities, accuracy_score(actual_labels, predicted_labels)


def obtain_ranked_list_R(predicted_probabilities):
    # predicted labels is a list of dictionaries
    # each dict either contains the probabilities or the probabilities for min/majority separately
    # it also contains the classes (should be same order as probabilties) and the actual target value (y_true)
    result_list = []
    for dic in predicted_probabilities:
        if "minority_probabilities" in dic:
            # so there are minority probabilities and we have to merge minority and majority lists
            min_probas = dic["minority_probabilities"]
            majority_probas = dic["majority_probabilities"]
            minority_classes = dic["minority_classes"]
            majority_classes = dic["majority_classes"]

            minority_prob_clas = list(zip(min_probas, minority_classes))
            majority_prob_clas = list(zip(majority_probas, majority_classes))

            merged_list = minority_prob_clas + majority_prob_clas
            # sort merge list by probabilties
            merged_list = sorted(merged_list, key=lambda tuple: tuple[
                0])  # lowest probability first! Want to iterate from lowest to highest

            # iterate through the confidence value list together with the class
            for index in range(len(merged_list) - 1):

                tup = merged_list[index]
                prob = tup[0]
                cls = tup[1]

                next_tup = merged_list[index + 1]
                next_prob = next_tup[0]
                next_cls = tup[1]

                # difference in confidence value less than 1.5%?
                if next_prob - prob < 0.015:
                    # minority class higher confidence than majority class? (lists are sorted)
                    if cls in majority_classes and next_cls in minority_classes:

                        # only take into account if majority and minority confidence are higher than uniform random
                        # probability
                        if prob > 1 / len(majority_classes) and next_prob > 1 / len(minority_classes):
                            # change confidence values
                            prob = prob * 1.015
                            next_prob = next_prob - next_prob * 0.015

                            tup = (prob, cls)
                            next_tup = (next_prob, next_cls)

                            # swap order of the tuples (majority class should now be before minority)
                            merged_list[index] = next_tup
                            merged_list[index + 1] = tup

            # sort list such that highest probability is first now
            merged_list = sorted(merged_list, key=lambda tup: tup[0], reverse=True)
            dic = {"probabilities": [tup[0] for tup in merged_list], "classes": [tup[1] for tup in merged_list],
                   "target": dic["target"]}
            result_list.append(dic)
        else:
            probs = dic["probabilities"]
            classes = dic["classes"]
            sorted_merged_list = sorted(list(zip(probs, classes)), key=lambda tup: tup[0], reverse=True)

            # no min/majority so just append dic
            dic["probabilities"] = [tup[0] for tup in sorted_merged_list]
            dic["classes"] = [tup[1] for tup in sorted_merged_list]
            result_list.append(dic)

    return result_list


def accuracy_for_e(ranked_list, e=10):
    correctly_classified_per_e = {x: 0 for x in range(1, e + 1)}

    for entry in ranked_list:
        probabilities = entry["probabilities"]
        classes = entry["classes"]
        y_true = entry["target"]
        class_predicted = False
        for ind, (prob, cls) in enumerate(zip(probabilities, classes)):
            if cls == y_true:
                class_predicted = True

            if class_predicted and ind < e:
                correctly_classified_per_e[ind + 1] = correctly_classified_per_e[ind + 1] + 1

        # maybe list is not as long as it should --> still add correctly_classified
        if len(probabilities) < len(range(e)):
            if class_predicted:
                for i in [x for x in list(range(1, e + 1)) if x > len(probabilities)]:
                    correctly_classified_per_e[i] = correctly_classified_per_e[i] + 1

    accuracy_per_e = {i: correctly_classified_per_e[i] / len(ranked_list) for i in range(1, e + 1)}
    result_dic = {"R_e": list(range(1, e + 1)), "A@e": [accuracy_per_e[i] for i in range(1, e + 1)],
                  "Method": ["Tailored Approach" for x in range(1, e + 1)]}
    result_df = pd.DataFrame(result_dic)
    return accuracy_per_e, result_df


def accuracy_e_baselines(clf, X_test, y_test, e=10, method="RF+B"):
    # returns list of probabilities for each point
    probabilities = clf.predict_proba(X_test)
    classes = clf.classes_

    correctly_classified_per_e = {x: 0 for x in range(1, e + 1)}

    # sort probabilities, together with classes so the order is the same!
    probabilities_classes_sorted = [sorted(list(zip(probs, classes)), key=lambda tup: tup[0], reverse=True) for probs in
                                    probabilities]

    for probs_classes, y_true in zip(probabilities_classes_sorted, y_test):
        # now we have probabilities for one test sample
        class_predicted = False
        for ind, (prob, cls) in enumerate(probs_classes):
            if cls == y_true:
                class_predicted = True
            if class_predicted and ind <= 9:
                correctly_classified_per_e[ind + 1] = correctly_classified_per_e[ind + 1] + 1

    accuracy_per_e = {i: correctly_classified_per_e[i] / len(probabilities_classes_sorted) for i in range(1, e + 1)}
    result_dic = {"R_e": list(range(1, e + 1)), "A@e": [accuracy_per_e[i] for i in range(1, e + 1)],
                  "Method": [method for x in range(1, e + 1)]}
    result_df = pd.DataFrame(result_dic)
    return accuracy_per_e, result_df


def calc_stats_SPH(root_node, concentration_function=gini):
    product_group_nodes = [node for node in PreOrderIter(root_node) if not node.children]
    sph_cm = sum([concentration_function(node.sph_labels) for node in product_group_nodes])
    sph_cm = sph_cm / len(product_group_nodes)

    missing_values = []
    for node in product_group_nodes:
        data = np.array(node.sph_data)
        df = pd.DataFrame(data=data, columns=[f"F{i}" for i in range(data.shape[1])])
        df = df.dropna(axis=1, how='all')
        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_value_df = pd.DataFrame({'column_name': df.columns,
                                         'percent_missing': percent_missing})

        missing = missing_value_df.mean(axis=0)
        missing_values.append(missing)

    n_classes_per_group = [len(np.unique(node.sph_labels)) for node in product_group_nodes]
    n_classes_per_group = sum(n_classes_per_group) / len(product_group_nodes)

    n_samples_per_group = [len(node.sph_data) for node in product_group_nodes]
    n_samples_per_group = sum(n_samples_per_group) / len(n_samples_per_group)

    return sph_cm, n_classes_per_group, n_samples_per_group, sum(missing_values) / len(missing_values)


def calc_stats_CPI(root_node, concentration_function=gini):
    product_group_nodes = [node for node in PreOrderIter(root_node) if not node.children]
    resulting_labels = []
    samples = []
    missing_values = []
    for node in product_group_nodes:
        labels = node.cpi_labels

        if isinstance(labels, tuple):
            min_labels = labels[0]
            maj_labels = labels[1]
            resulting_labels.append(min_labels)
            indices = []
            new_maj_labels = []
            for ind, label in enumerate(maj_labels):
                if label != -1:
                    new_maj_labels.append(label)
                    indices.append(ind)
            resulting_labels.append(new_maj_labels)
            maj_data = node.cpi_data[1]
            maj_data = np.take(maj_data, indices, axis=0)

            data = np.concatenate((node.cpi_data[0], maj_data))
            df = pd.DataFrame(data=data, columns=[f"F{i}" for i in range(node.cpi_data[0].shape[1])])
            df = df.dropna(axis=1, how='all')

            percent_missing = df.isnull().sum() * 100 / len(df)
            missing_value_df = pd.DataFrame({'column_name': df.columns,
                                             'percent_missing': percent_missing})

            missing = missing_value_df.mean(axis=0)
            missing_values.append(missing)
            missing_values.append(missing)

            samples.append(len(node.cpi_data[0]))
            samples.append(len(node.cpi_data[1]))

        else:
            df = pd.DataFrame(data=node.cpi_data, columns=[f"F{i}" for i in range(node.cpi_data.shape[1])])
            df = df.dropna(axis=1, how='all')

            resulting_labels.append(labels)
            samples.append(len(node.cpi_data))

            percent_missing = df.isnull().sum() * 100 / len(df)
            missing_value_df = pd.DataFrame({'column_name': df.columns,
                                             'percent_missing': percent_missing})
            missing = missing_value_df.mean(axis=0)
            missing_values.append(missing)
    cpi_cm_value = sum([concentration_function(labels) for labels in resulting_labels])
    cpi_cm_value = cpi_cm_value / len(resulting_labels)

    n_classes_per_group = [len(np.unique(lables)) for lables in resulting_labels]
    n_classes_per_group = sum(n_classes_per_group) / len(resulting_labels)

    n_samples = sum(samples) / len(samples)

    return cpi_cm_value, n_classes_per_group, n_samples, sum(missing_values) / len(missing_values)


def run_baseline_rf(df_train, df_test):
    X_train = df_train[[f"F{i}" for i in range(n_features)]].to_numpy()
    y_train = df_train["target"].to_numpy()

    X_test = df_test[[f"F{i}" for i in range(n_features)]].to_numpy()
    y_test = df_test["target"].to_numpy()

    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    imp = KNNImputer(missing_values=np.nan)
    X_train = imp.fit_transform(X_train)

    # feat_selector = BorutaPy(rf, n_estimators='auto', max_iter=100, verbose=2, random_state=1)
    # feat_selector.fit(X_train,y_train)
    # X_filtered = feat_selector.transform(X)
    # X_train = X_train[:, feat_selector.support_]
    print(f"selected feature shape: {X_train.shape}")

    rf.fit(X_train, y_train)
    X_test = imp.transform(X_test)
    # X_test = X_test[:, feat_selector.support_]
    acc_e, rf_df = accuracy_e_baselines(rf, X_test, y_test)

    score = rf.score(X_test, y_test)
    # scores = cross_val_score(rf, X, y, cv=10)
    print(f"Random Forest score: {score}")
    return rf_df, acc_e


def calculate_sph_accuracy(resulting_nodes_per_node_id):
    cpi_nodes = {}
    for node_id, node in resulting_nodes_per_node_id.items():
        node.cpi_data = node.sph_data
        node.cpi_labels = node.sph_labels
        cpi_nodes[node_id] = node
    # create model repository
    model_repo = create_ensemble(cpi_nodes)

    # classifiy new samples based on repository
    predicted_probabilities, score = classifiy_new_samples(model_repo, df_test, n_features=n_features, method="SPH")

    # get ranked list
    ranked_list = obtain_ranked_list_R(predicted_probabilities)

    # get accuracy by length e, default e=10
    sph_acc, sph_df = accuracy_for_e(ranked_list)

    return sph_acc, sph_df


def run_cpi_in_isolation(X_train, y_train, X_test, y_test, concentration_function, cm_val, p_val):
    # Run CPI on whole data
    cpi_isolation_result = run_cpi(partition_labels=y_train, concentration_function=concentration_function,
                                   cm_threshold=cm_val,
                                   p_threshold=p_val)

    if isinstance(cpi_isolation_result, tuple):

        minority_indices = cpi_isolation_result[0]
        majority_indices = cpi_isolation_result[1]

        minority_partition = np.take(X_train, minority_indices, axis=0)
        minority_labels = np.take(y_train, minority_indices, axis=0)

        majority_partition = np.take(X_train, majority_indices, axis=0)
        majority_labels = np.take(y_train, majority_indices, axis=0)

        if len(np.unique(majority_labels)) == 1:
            majority_partition = np.concatenate([minority_partition, majority_partition])
            majority_labels = np.concatenate([majority_labels, [-1 for x in minority_labels]])

        minority_estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                       ("forest", RandomForestClassifier(random_state=1234,
                                                                         n_estimators=100))])

        minority_estimator.fit(minority_partition, minority_labels)

        majority_estimator = Pipeline([("imputer", KNNImputer(missing_values=np.nan)),
                                       ("forest", RandomForestClassifier(random_state=1234,
                                                                         n_estimators=100))])
        majority_estimator.fit(majority_partition, majority_labels)

        # pseudo model_repo for each group! Makes it easier to get accuracy etc.
        model_repo = {group: (minority_estimator, majority_estimator) for group in set(df_test['group'].values)}

        # classifiy new samples based on repository
        predicted_probabilities, score = classifiy_new_samples(model_repo, df_test, n_features=n_features, method="CPI")

        # get ranked list
        ranked_list = obtain_ranked_list_R(predicted_probabilities)

        # get accuracy by length e, default e=10
        cpi_isolation_accuracy, cpi_isolation_accuracy_df = accuracy_for_e(ranked_list)

    else:
        # same result as RF --> but should actually not appear
        cpi_isolation_accuracy_df = rf_df
        cpi_isolation_accuracy_df['Method'] = 'CPI'
    return cpi_isolation_accuracy_df


def plot_data_distribution(df):
    print(df.head)
    df.plot(kind="bar", x="Class")
    plt.title("Class Frequency")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.yticks(list(range(0, 100, 5)))

    plt.savefig(f"{data_output_directory}/class_dist_{run_id}.png")
    plt.close()


def plot_accuracy_result():
    fig, ax = plt.subplots()
    plt.ioff()
    ax1 = ax.plot(vitali_df["R_e"], vitali_df["A@e"], marker='o', label='SPH+CPI')
    ax2 = ax.plot(rf_df["R_e"], rf_df["A@e"], label='RF+B', marker='o')
    ax3 = ax.plot(sph_df["R_e"], sph_df["A@e"], label='SPH', marker='o')
    ax4 = ax.plot(cpi_isolation_df["R_e"], cpi_isolation_df["A@e"], label="CPI", marker='o')

    for _, df in zip([ax1, ax2,
                      ax3,
                      ax4
                      ], [vitali_df, rf_df,
                          sph_df,
                          cpi_isolation_df
                          ]):

        for i, txt in enumerate(df['A@e'].values):
            txt = txt * 100
            txt = "%.0f" % txt
            txt = txt + "%"
            move_Accuracy = 0.03
            # text of rf underneath the curve
            if df["Method"].values[0] == "RF" or df["Method"].values[0] == "CPI":
                move_Accuracy = move_Accuracy \
                                * -1
            ax.annotate(txt, (df["R_e"].values[i] + 0.05, df["A@e"].values[i] + move_Accuracy))

    plt.xticks([i for i in range(1, 11)])
    plt.ylabel("Accuracy A@e")
    plt.ylim(0, 1)
    plt.xlabel(r"Length of Recommendation List R_e")
    plt.legend()
    plt.savefig(
        f"{result_output_directory}/accuracy_{concentration_measure}{cm_val * 100}_q{p_val * 100}_dataset{run_id}.png")
    plt.close()


if __name__ == '__main__':
    import seaborn as sns

    sns.set_style()

    import argparse

    ###############################################################
    ######################## Default Arguments ####################
    cm_vals = [
        # 0.1, 0.15, 0.2,
        0.25,
        0.3,
        0.35, 0.4
    ]
    p_quantile = [
        0.7, 0.75,
        0.8, 0.85,
        0.9, 0.95
    ]
    ###############################################################

    ###############################################################
    ######################## Parse Arguments from CMD##############
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpi_free", help="Runs the cpi version that does not require additional parameters",
                        const=True, default=False, action='store_const')

    parser.add_argument("-imbalance", help="Degree of imbalance. This should either be 'low', 'normal' or 'high'.",
                        default='normal', choices=ImbalanceGenerator.imbalance_degrees)

    parser.add_argument('-max_info_loss', type=float, help="Percentage of information loss to use. Default is 25 percent",
                        nargs='*',
                        required=False, default=[0.25])

    parser.add_argument('-gini_thresholds', type=float,
                        help='Percentage of the threshold for the gini index. Per default, multiple values from 25 '
                             'to 40 in 5th steps are executed.',
                        nargs='*',
                        required=False, default=cm_vals)

    parser.add_argument('-p_thresholds', type=float,
                        help='Percentage of the thresholds for the p_quantile. Per default, multiple values from 70 '
                             'to 95 in 5th steps are executed',
                        nargs='*',
                        required=False, default=p_quantile)

    parser.add_argument('-runs', type=int, help='Number of runs to perform. The runs differ in different seed values.',
                        required=False, default=10)
    args = parser.parse_args()

    imbalance_degree = args.imbalance
    run_cpi_no_parameters = args.cpi_free

    max_info_loss_values = args.max_info_loss

    cm_vals = args.gini_thresholds

    p_quantile = args.p_thresholds

    total_runs = args.runs
    runs = range(1, total_runs + 1)
    ###############################################################

    # todo: set parameters here, like generating datasets, concentration function, which cpi etc.
    generate_data = True
    concentration_measure = 'gini'

    # Default for directories
    if imbalance_degree == "normal":
        data_output_directory = "data_split"
        result_output_directory = "result_split"
    else:
        data_output_directory = f"imbalance_degree/{imbalance_degree}/data_split"
        result_output_directory = f"imbalance_degree/{imbalance_degree}/result_split"

    if not os.path.exists(data_output_directory):
        os.makedirs(data_output_directory)

    if not os.path.exists(result_output_directory):
        os.makedirs(result_output_directory)

    # concentration_measure = 'theil'
    concentration_function = theil if concentration_measure == 'theil' else gini

    runs_data = {'run_id': [], concentration_measure: [], 'p-quantile': []}
    result_df = pd.DataFrame()

    for run_id in runs:
        # for different runs, we use different random seeds
        np.random.seed(run_id * 5)
        random.seed(run_id * 10)

        runs_data['run_id'].append(run_id)

        for max_info_loss in max_info_loss_values:
            print(
                "-------------------------------------------------------------------------------------------------------------------")
            print(
                f"---------------------------Running SPH with info_loss={max_info_loss}----------------------------------------------")

            cm_q_parameter_list = [(x, y) for x in cm_vals for y in p_quantile]
            print(cm_q_parameter_list)

            n_features = 100
            n_samples = 1050
            generator = ImbalanceGenerator()

            generation_mechanism = "hardcoded"
            # generation_mechanism = "default"

            sph_executed = 100
            # sph should only be executed 5 times according to the paper
            sph_A_at_one = 1
            rf_at_one = 1

            if os.path.isfile(
                    f"{data_output_directory}/train_{generation_mechanism}_{run_id}.csv") and not generate_data:
                # load existing data. I don't think we need this for the journal as we make everything pseudo random
                # and data generation is very fast
                df_train = pd.read_csv(f"{data_output_directory}/train_{generation_mechanism}_{run_id}.csv",
                                       sep=';')
                df_test = pd.read_csv(f"{data_output_directory}/test_{generation_mechanism}_{run_id}.csv", sep=';')
                data_df = pd.concat([df_train, df_test])
                root_node = HardCodedHierarchy().create_hardcoded_hierarchy()

            else:
                ###############################################################
                ######################## Generate Data ########################
                if generation_mechanism == "hardcoded":
                    root_node = HardCodedHierarchy().create_hardcoded_hierarchy()
                else:
                    root_node = None

                data_df = generator.generate_data_with_product_hierarchy(root=root_node,
                                                                         imbalance_degree=imbalance_degree)
                ###############################################################

                # split in training and test data
                df_train, df_test = train_test_splitting(data_df)

                print(f"Test data contains: {Counter(zip(df_test['target'].to_numpy(), df_test['group'].values))}")

                root_node = generator.root

                # create hierarchy (samples) solely with df_train
                # update training data and labels in product hierarchy
                # Important! We distinguish training_data, training_labels and data, target!
                root_node = update_data_and_training_data(root_node, df_train, data_df, n_features=n_features)

                print("--------------------------------------------------")
                print("----------------Training Hierarchy----------------")
                print(RenderTree(root_node))
                # DotExporter(root_node, nodenamefunc= lambda n: n.node_id).to_picture("data/test.png")

                # run SPH
                resulting_nodes_per_node_id, sph_executed = SPH(root_node, max_info_loss=max_info_loss)

                counter = Counter(data_df['target'].to_numpy())
                n_classes = len(np.unique(data_df['target'].to_numpy()))
                sensor_df = data_df[[f"F{i}" for i in range(n_features)]]

                # Calculate missing data for dataset
                percent_missing = sensor_df.isnull().sum() * 100 / len(data_df)
                missing_value_df = pd.DataFrame({'column_name': sensor_df.columns,
                                                 'percent_missing': percent_missing})
                missing_value_df.sort_values('percent_missing', inplace=True)
                missing_whole_data = float(missing_value_df.mean(axis=0))

                df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
                df = df.rename(columns={'index': 'Class', 0: 'count'})
                df = df.sort_values(by=['count'])

                labels = data_df["target"].to_numpy()
                cm_index = concentration_function(labels)

                print(f'executed sph: {sph_executed}')
                sph_acc, sph_df = calculate_sph_accuracy(resulting_nodes_per_node_id)
                sph_cm_value, sph_n_classes, sph_n_samples, sph_missing = calc_stats_SPH(root_node,
                                                                                         concentration_function)

                sph_A_at_one = sph_acc[1]
                print(sph_A_at_one)

                rf_df, acc_rf = run_baseline_rf(df_train, df_test)
                rf_at_one = acc_rf[1]
                training_data_gini = concentration_function((df_train['target'].to_numpy()))
                print(f"{concentration_measure} index for dataset is: {cm_index}")
                print(f"{concentration_measure} index for training dataset is: {training_data_gini}")

                print("-------------------Statistics------------------------")
                print(f"--------Missing Values")
                print(f"    for whole dataset: {missing_whole_data}")
                print(f"    for SPH: {float(sph_missing)}")

                print(f"--------{concentration_measure} index")
                print(f"    for whole dataset: {cm_index}")
                print(f"    for SPH: {sph_cm_value}")

                print(f"--------Classes")
                print(f"    for whole dataset: {n_classes}")
                print(f"    for SPH: {sph_n_classes}")

                print(f"--------Samples")
                print(f"    for whole dataset: {n_samples}")
                print(f"    for SPH: {sph_n_samples}")

                print(f"--------Accuracy")
                print(f"    for RF+B: {acc_rf}")
                print(f"    after SPH: {sph_acc}")

                print("-----------------------")

            # checks passed, so we can save training/test data and the hierarchy specification
            df_train.to_csv(f"{data_output_directory}/train_{generation_mechanism}_{run_id}.csv", index=False, sep=';')
            df_test.to_csv(f"{data_output_directory}/test_{generation_mechanism}_{run_id}.csv", index=False, sep=';')

            plot_data_distribution(df)

            X_train = df_train[[f"F{i}" for i in range(n_features)]].to_numpy()
            y_train = df_train["target"].to_numpy()

            X_test = df_test[[f"F{i}" for i in range(n_features)]].to_numpy()
            y_test = df_test["target"].to_numpy()

            # run baseline
            rf_df.to_csv(f"{result_output_directory}/Baseline_{run_id}.csv", sep=';')

            sph_df[f"{concentration_measure}"] = -1
            sph_df['p value'] = -1
            sph_df["Method"] = "SPH"
            sph_df["Run"] = run_id
            sph_df["SPH Executed"] = sph_executed
            sph_df["info loss"] = max_info_loss

            sph_A_at_one = sph_acc[1]

            rf_df["Run"] = run_id
            rf_df[f"{concentration_measure}"] = -1
            rf_df['p value'] = -1
            rf_df["SPH Executed"] = sph_executed
            result_df = pd.concat([result_df, sph_df, rf_df])
            stats_df = pd.DataFrame()

            """
            Try new approach --> CPI without parameter thresholds!
            """
            if run_cpi_no_parameters:
                # todo: basically cpi_free and cpi are very similar --> cpi executed in a loop with the resulting
                #  accuracy and so on
                result_file_name = "w_cpi_free_{concentration_measure}_accuracy_all_runs.xlsx"
                # Run CPI_free in isolation
                cpi_isolation_df = run_cpi_free_in_isolation(X_train=X_train, y_train=y_train, df_test=df_test)

                p_val = -1
                cm_val = -1

                cpi_partitions_per_node = CPI_no_threshold(resulting_nodes_per_node_id)
                # create model repository
                model_repo = create_ensemble(cpi_partitions_per_node)

                # classifiy new samples based on repository
                predicted_probabilities, score = classifiy_new_samples(model_repo, df_test, n_features=n_features,
                                                                       method="CPI free")

                # get ranked list
                ranked_list = obtain_ranked_list_R(predicted_probabilities)

                # get accuracy by length e, default e=10
                acc_vitali, vitali_df = accuracy_for_e(ranked_list)

                # get accuracy by length e, default e=10
                acc_vitali, vitali_df = accuracy_for_e(ranked_list)
                vitali_df[concentration_measure] = cm_val
                vitali_df["p value"] = p_val
                vitali_df["Run"] = run_id
                vitali_df["SPH A@1"] = sph_A_at_one
                vitali_df["RF+B A@1"] = acc_rf[1]
                vitali_df["info loss"] = max_info_loss

                vitali_df[concentration_measure] = cm_val
                vitali_df['p value'] = p_val
                vitali_df["SPH Executed"] = sph_executed
                vitali_df["Run"] = run_id
                result_df = pd.concat([result_df, vitali_df])

                #####################plot ######################################################
                plot_accuracy_result()
                ################################################################################

                cpi_cm_value, cpi_n_classes, cpi_n_samples, cpi_missing_values = calc_stats_CPI(root_node,
                                                                                                concentration_function)

                statistics = {
                    f"CM Index": [concentration_measure],
                    f"{concentration_measure}_parameter": [cm_val],
                    "p_parameter": [p_val],
                    concentration_measure: [cm_index],
                    f"Training {concentration_measure}": [training_data_gini],
                    f"{concentration_measure} SPH": [sph_cm_value],
                    f"{concentration_measure} CPI": [cpi_cm_value],
                    "Missing": [missing_whole_data], "Missing SPH": [float(sph_missing)],
                    "Missing CPI": [float(cpi_missing_values)],
                    "#Classes": [n_classes], "#Classes SPH": [sph_n_classes], "#Classes CPI": [cpi_n_classes],
                    "#Samples": [n_samples], "#Samples SPH": [sph_n_samples], "#Samples CPI": [cpi_n_samples]
                }

                stats_df = pd.concat([stats_df, pd.DataFrame(data=statistics)])
                stats_df.to_csv(f"{result_output_directory}/w_cpi_free_{concentration_measure}_stats_{run_id}.csv",
                                sep=';',
                                decimal=',')

                print(f"{concentration_measure} index for datasets is: {cm_index}")
                print(f"{concentration_measure} index for training dataset is: {training_data_gini}")

                print("-------------------Statistics------------------------")
                print(f"--------Missing Values")
                print(f"    for whole dataset: {missing_whole_data}")
                print(f"    for SPH: {float(sph_missing)}")

                print(f"--------{concentration_measure} index")
                print(f"    for whole dataset: {cm_index}")
                print(f"    for SPH: {sph_cm_value}")

                print(f"--------Classes")
                print(f"    for whole dataset: {n_classes}")
                print(f"    for SPH: {sph_n_classes}")

                print(f"--------Samples")
                print(f"    for whole dataset: {n_samples}")
                print(f"    for SPH: {sph_n_samples}")

                print(f"--------Accuracy")
                print(f"    for RF+B: {acc_rf}")
                print(f"    after SPH: {sph_acc}")

                print("-----------------------")
                #result_df.to_csv( f"{result_output_directory}/w_cpi_free_{concentration_measure}_accuracy_all_runs.csv",
                #                   index=False)

            else:

                for cm_val, p_val in cm_q_parameter_list:
                    print(
                        "--------------------------------------------------------------------------------------------------------------------")
                    print(
                        f"------------------Running with gini={cm_val} and p-value={p_val}------------------------------------------------------")

                    # Run CPI in isolation
                    cpi_isolation_df = run_cpi_in_isolation(X_train=X_train, X_test=X_test, y_train=y_train,
                                                            y_test=y_test,
                                                            concentration_function=concentration_function,
                                                            cm_val=cm_val,
                                                            p_val=p_val)

                    result_df = pd.concat([result_df, cpi_isolation_df])
                    # Run CPI
                    cpi_partitions_per_node = CPI(resulting_nodes_per_node_id, cm_threshold=cm_val, p_threshold=p_val,
                                                  concentration_function=concentration_function)
                    cpi_partitions_per_node = preprocessing(cpi_partitions_per_node)

                    # create model repository
                    model_repo = create_ensemble(cpi_partitions_per_node)

                    # classifiy new samples based on repository
                    predicted_probabilities, score = classifiy_new_samples(model_repo, df_test, n_features=n_features,
                                                                           method="SPH+CPI")

                    # get ranked list
                    ranked_list = obtain_ranked_list_R(predicted_probabilities)

                    # get accuracy by length e, default e=10
                    acc_vitali, vitali_df = accuracy_for_e(ranked_list)
                    vitali_df[concentration_measure] = cm_val
                    vitali_df["p value"] = p_val
                    vitali_df["Run"] = run_id
                    vitali_df["SPH A@1"] = sph_A_at_one
                    vitali_df["RF+B A@1"] = acc_rf[1]
                    vitali_df["CPI A@1"] = cpi_isolation_df[cpi_isolation_df["R_e"] == 1]["A@e"].values[0]
                    vitali_df["info loss"] = max_info_loss
                    vitali_df[concentration_measure] = cm_val
                    vitali_df['p value'] = p_val
                    vitali_df["SPH Executed"] = sph_executed
                    vitali_df["Run"] = run_id

                    acc_values = list(acc_vitali.values())
                    vitali_df["Avg Acc SPH+CPI"] = sum(acc_values)/len(acc_values)
                    result_df = pd.concat([result_df, vitali_df])

                    #####################plot ######################################################
                    plot_accuracy_result()
                    ################################################################################

                    cpi_cm_value, cpi_n_classes, cpi_n_samples, cpi_missing_values = calc_stats_CPI(root_node,
                                                                                                    concentration_function)
                    missing_whole_data = float(missing_value_df.mean(axis=0))

                    statistics = {
                        f"CM Index": [concentration_measure],
                        f"{concentration_measure}_parameter": [cm_val],
                        "p_parameter": [p_val],
                        f"Training {concentration_measure}": [training_data_gini],
                        concentration_measure: [cm_index],
                        f"{concentration_measure} SPH": [sph_cm_value],
                        f"{concentration_measure} CPI": [cpi_cm_value],
                        "Missing": [missing_whole_data], "Missing SPH": [float(sph_missing)],
                        "Missing CPI": [float(cpi_missing_values)],
                        "#Classes": [n_classes], "#Classes SPH": [sph_n_classes], "#Classes CPI": [cpi_n_classes],
                        "#Samples": [n_samples], "#Samples SPH": [sph_n_samples], "#Samples CPI": [cpi_n_samples]
                    }

                    stats_df = pd.concat([stats_df, pd.DataFrame(data=statistics)])
                    stats_df.to_csv(f"{result_output_directory}/{concentration_measure}_stats_{run_id}.csv", sep=';',
                                    decimal=',')

                    print(f"{concentration_measure} index for training dataset is: {training_data_gini}")
                    print("-------------------Statistics------------------------")
                    print(f"--------Missing Values")
                    print(f"    for whole dataset: {missing_whole_data}")
                    print(f"    for SPH: {float(sph_missing)}")
                    print(f"    for SPH+CPI:: {float(cpi_missing_values)}")

                    print(f"--------{concentration_measure} index")
                    print(f"    for whole dataset: {cm_index}")
                    print(f"    for SPH: {sph_cm_value}")
                    print(f"    for SPH+CPI:: {cpi_cm_value}")

                    print(f"--------Classes")
                    print(f"    for whole dataset: {n_classes}")
                    print(f"    for SPH: {sph_n_classes}")
                    print(f"    for SPH+CPI:: {cpi_n_classes}")

                    print(f"--------Samples")
                    print(f"    for whole dataset: {n_samples}")
                    print(f"    for SPH: {sph_n_samples}")
                    print(f"    for SPH+CPI:: {cpi_n_samples}")

                    print(f"--------Accuracy")
                    print(f"    for RF+B: {acc_rf}")
                    print(f"    after SPH: {sph_acc}")
                    print(f"    after SPH+CPI: {acc_vitali}")

                    result_df.to_csv(f"{result_output_directory}/{concentration_measure}_accuracy_all_runs.csv",
                                       index=False)
