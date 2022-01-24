import logging
import math
import random
from collections import Counter
from typing import List

import anytree
import imblearn
from anytree import RenderTree, PreOrderIter
from scipy.stats import boltzmann, zipfian, dlaplace, poisson, zipf
from skclean.simulate_noise import flip_labels_uniform
from sklearn.datasets import make_classification, make_blobs
from itertools import zip_longest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, balanced_accuracy_score, \
    precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from Hierarchy import Node, HardCodedHierarchy, FlatHierarchy, make_unbalance_hierarchy
import concentrationMetrics as cm


# np.random.seed(1)
# random.seed(1)


def _check_groups_samples_classes(n_groups, n_samples_per_group, n_classes_per_group):
    if not (n_groups or n_samples_per_group or n_classes_per_group):
        logging.info(
            "Neither n_groups nor n_samples_per_group nor n_classes_per_group are given. using default parameters.")
        return True


def assign_class(cls, n_classes):
    while cls > n_classes - 1:
        cls = cls - n_classes
    return cls


class ImbalanceGenerator:
    """
    responsible for generating the data.
    currently, there are two options for generating the data. you may pass one of the hierarchies in hierarchy.py
    or you do not specify a hierarchy and then a 'default' one is generated.
    Yet, for both ways, the groups (or the actual data) is generated based on the specification in the same way.
    """
    imbalance_degrees = ['very_low', 'low', 'normal', 'high', 'very_high']

    # Not used yet. At the moment we simply use the function from scipy as parameter. That way we do not need mappings
    # for each distribution
    BOLTZMAN = "boltzman"
    ZIPFIAN = "zipfian"
    POISSON = "poisson"

    DISTRIBUTIONS = [BOLTZMAN, ZIPFIAN, POISSON]

    distribution_mapping = {
        BOLTZMAN: boltzmann.rvs,
        ZIPFIAN: zipfian.rvs,
        POISSON: poisson.rvs,
    }

    def __init__(self, n_features=100, n_samples_total=1050, n_levels=4, total_n_classes=84,
                 features_remove_percent=0.2, imbalance_degree="normal",
                 root=HardCodedHierarchy().create_hardcoded_hierarchy(),
                 noise=0,
                 low_high_split=(0.3, 0.7),
                 distribution=zipfian.rvs,
                 random_state=1234):
        """
        :param n_features: number of features to use for the overall generated dataset
        :param n_samples_total: number of samples that should be generated in the whole dataset
        :param n_levels: number of levels of the hierarchy. Does not need to be specified if a hierarchy is already given!
        :param total_n_classes: number of classes for the whole dataset
        :param features_remove_percent: number of features to remove/ actually this means to have this number of percent
        as missing features in the whole dataset. Currently, this will be +5/6 percent.
        :param imbalance_degree: The degree of imbalance. Should be either 'normal', 'low' or 'high'. Here, normal means
        to actually use the same (hardcoded) hierarchy that is passed via the root parameter.
        'low' means to have a more imbalanced dataset and 'high' means to have an even more imbalanced dataset.
        :param root: Root node of a hierarchy. This should be a root node that represent an anytree and stands for the hierarchy.
        :param distribution: Distribution to use. In the moment, either boltzman.rvs or zipfian.rvs are tested from the scipy.stats module!
        :param noise: Percentage of noise to generate (in [0,1])
        :param low_high_split: split percentage for distribution of samples and classes to the nodes.
        :param random_state:
        """
        self.imbalance_degree = imbalance_degree
        self.hardcoded = None
        self.root = root
        self.n_features = n_features
        self.n_levels = n_levels
        self.features_remove_percent = features_remove_percent
        self.noise = noise
        self.low_split = low_high_split[0]
        self.high_split = low_high_split[1]
        self.cls_prob_distribution = distribution
        self.n_samples_total = n_samples_total
        self.total_n_classes = total_n_classes
        self.random_state = random_state

        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

    def _eq_div(self, N, i):
        """
        Divide N into i buckets while preserving the remainder to the buckerts as well.
        :return: list of length i
        """
        return [] if i <= 0 else [N // i + 1] * (N % i) + [N // i] * (i - N % i)

    def gini(self, x):
        my_index = cm.Index()
        counter = Counter(x)
        return my_index.gini(counter.values())

    def generate_data_with_product_hierarchy(self):
        """
        Main method of Data generation.
        Here, the data is generated according to various parameters.
        We mainly distinguish if we have an hierarchy given. This should be given with the root parameter that contains
        the root of an anytree. This root node can be used as representative for the whole tree.
        :return: A dataframe that contains the data and the hierarchy.
        The data is encoded via the feature columns F_0, ..., F_n_features.
        The hierarchy is implicitly given through the specific attributes that represent the hierarchy.
        """

        if self.imbalance_degree not in ImbalanceGenerator.imbalance_degrees:
            self.logger.error(f"imbalance_degree should be one of {ImbalanceGenerator.imbalance_degrees} but got"
                              f" {self.imbalance_degree}")
            self.logger.warning(f"Setting imbalance_degree to default 'normal'")
            self.imbalance_degree = "normal"

        # np.random.seed(1)

        if not self.root:
            features = list(range(self.n_features))
            self.root = Node(node_id="0", n_samples=self.n_samples_total, feature_set=features,
                             n_classes=self.total_n_classes,
                             classes=(0, self.total_n_classes))
            # generate default hierarchy
            n_nodes_per_level_per_parent = self._generate_default_hierarchy_spec()
            self._assign_samples_classes_features()
            self._generate_class_occurences()
            self.hardcoded = False
        else:
            self._get_hardcoded_hierarchy_spec()
            self.hardcoded = True
            # remove features, don't need this for "generated" hierarchy as we define the features there!
            group_nodes_to_create = self._remove_features_from_spec()
        # check if all features are in the whole data, i.e., we have n_features
        self._check_features()

        # actual generation of the data
        # adjust class distribution is inside data generation, Todo: separate that
        groups = self._generate_groups_from_hierarchy_spec()

        return self._create_dataframe(groups)

    def _get_hardcoded_hierarchy_spec(self):
        """
        Generates the specification for the hardcoded hierarchy. Is required for the journal paper.
        :return:
        """
        # generate features for hierarchy
        features = list(range(self.n_features))
        self.root.feature_set = features

        # if n_samples not already specified, set to default value
        if not self.root.n_samples:
            self.root.n_samples = self.n_samples_total

        groups = self._get_leaf_nodes()

        for group in groups:
            imbalance_degree = self.imbalance_degree
            n_samples = group.n_samples
            occurences = group.class_occurences
            # special condition with n_samples < 15 to cover cases where n_classes=9 and n_samples=12
            if imbalance_degree == 'normal' and n_samples > 15:
                # do nothing in this case
                pass

            elif imbalance_degree == 'high' or imbalance_degree == 'very_high':

                # get max occurence and the index for it
                max_occurence = max(occurences)
                max_index = occurences.index(max_occurence)

                # this will be our new modified occurence list.
                # We need this because we do not (!) want to sort the list!
                # Otherwise this would change the occurence for a specific class
                new_occurences = occurences.copy()
                median = np.median(occurences)
                average = sum(occurences) / len(occurences)
                # important to take integers, we cannot divide float into n buckets later on
                median_or_average = int(average) if average < median else int(median)

                for i, occ in enumerate(occurences):
                    # check if we have at least two samples and this is not the max_occurence.
                    if occ > median_or_average and occ < max_occurence:
                        # This can easily be changed to remove exactly 5%
                        new_occurences[max_index] += occ - median_or_average
                        new_occurences[i] = median_or_average

                    if imbalance_degree == 'high':
                        break
                occurences = new_occurences

            elif imbalance_degree == 'low' or imbalance_degree == 'very_low':
                original_average = sum(occurences) / len(occurences)
                n_max_classes = 1
                if imbalance_degree == 'very_low':
                    # number of classes that are above average
                    n_max_classes = len([x for x in occurences if x > original_average])

                # for each class above average, we run the following procedure
                for i in range(n_max_classes):
                    # here we want to make the classes more balanced
                    # idea: move from majority one sample to each minority class
                    max_occurence = max(occurences)
                    max_index = occurences.index(max_occurence)
                    new_occurences = occurences.copy()
                    median = np.median(occurences)
                    average = sum(occurences) / len(occurences)
                    # important to take integers, we cannot divide float into n buckets later on
                    median_or_average = int(average) if median == 1 else int(median)

                    # if len(new_occurences) < max_occurence:
                    print(max_occurence - median_or_average)
                    new_occurences[max_index] = median_or_average

                    # equal division of max - average
                    rest = self._eq_div((max_occurence - median_or_average), len(new_occurences) - 1)

                    # insert 0 at max position -> We do not want to add something to max.
                    rest.insert(max_index, 0)

                    for i, r in enumerate(rest):
                        new_occurences[i] += r
                    occurences = new_occurences

            group.class_occurences = occurences
        return groups

    def _generate_default_hierarchy_spec(self):
        node_per_level_per_node = {}
        for l in range(1, self.n_levels):
            counter = 1
            leave_nodes = self._get_leaf_nodes()

            n_parents = len(leave_nodes)
            n_child_nodes = n_parents * (l + 1)
            n_childs_per_parent = int(n_child_nodes / n_parents)
            for node in leave_nodes:
                for i in range(n_childs_per_parent):
                    new_node = Node(node_id=f"{counter}",
                                    parent=node)
                    counter += 1
            node_per_level_per_node[l] = n_child_nodes / (n_parents)
        return node_per_level_per_node

    def _generate_groups_from_hierarchy_spec(self):
        """
        Generates the product groups. That is, here is the actual data generated.
        For each group, according to the number of classes, samples and features the data is generated.
        :return: group_nodes: list of nodes that now have set the data and target attributes.
        Here, we only return the group nodes, but the data and target of the parent nodes is also set!
        """
        group_ids = []
        group_nodes = self._get_leaf_nodes()
        # get set of all features. We need this to keep track of the feature limits of all groups
        total_sample_feature_set = set([feature for group in group_nodes for feature in group.feature_set])

        # save limits of each feature --> first all are 0.0
        feature_limits = {feature: 0 for feature in total_sample_feature_set}

        current_class_num = 0

        remaining_samples = []
        if self.hardcoded:
            n_samples_to_generate = sum([int(group.n_samples * self.n_samples_total / 1050) for group in group_nodes])

            if self.n_samples_total > n_samples_to_generate:
                # share the remaining samples among the groups
                remaining_samples = self._eq_div(self.n_samples_total - n_samples_to_generate, len(group_nodes))

        # bottom up approach
        for i, group in enumerate(group_nodes):
            feature_set = group.feature_set
            n_features = len(feature_set)
            n_samples = group.n_samples
            if self.hardcoded:
                mult_factor = self.n_samples_total / 1050
                n_samples = int(n_samples * mult_factor)

            # add samples that are missing due to rounding errors
            if len(remaining_samples) > i:
                n_samples += remaining_samples[i]

            n_classes = group.n_classes
            classes = group.classes

            if group.class_occurences is not None:
                occurences = group.class_occurences
                occurences = list(occurences)

                # Calculate the weights (in range [0,1]) from the occurrences.
                # The weights are needed for the sklearn function 'make_classification'
                weights = [occ / sum(occurences) for occ in occurences]
            else:
                logging.error("No occurences specified!")

            n_features_to_move = 1
            # take random feature(s) along we move the next group
            feature_to_move = np.random.choice(feature_set, n_features_to_move)
            for feature in feature_to_move:
                feature_limits[feature] += 1

            # set number of informative features
            n_informative = n_features - 1

            # The questions is, if we need this function if we have e.g., less than 15 samples. Maybe for this, we
            # can create the patterns manually?
            X, y = make_classification(n_samples=n_samples,
                                       n_classes=n_classes,
                                       # > 1 could lead to less classes created, especially for low n_samples or
                                       # if the occurence for a class is less than this value
                                       n_clusters_per_class=1,
                                       n_features=n_features,
                                       n_repeated=0,
                                       n_redundant=0,
                                       n_informative=n_informative,
                                       weights=weights,
                                       random_state=self.random_state,
                                       # higher value can cause less classes to be generated
                                       flip_y=0,
                                       # class_sep=0.1,
                                       hypercube=True,
                                       # shift=random.random(),
                                       # scale=random.random()
                                       )

            created_classes = len(np.unique(y))
            created_samples = X.shape[0]

            # todo: If we do not generated n_classes or n_samples (or both), we could generate/relabel them
            if created_classes < n_classes:
                print(
                    f"should create {n_classes} and have created {created_classes} classes for n_samples {n_samples} "
                    f"and weights={weights}")

            if created_samples < n_samples:
                print(f"should create {n_samples} and have created {created_samples}")

            # normalize x into [0,1] interval
            X = (X - X.min(0)) / X.ptp(0)

            for i, f in enumerate(feature_set):
                # move each feature by its feature limits
                X[:, i] = X[:, i] + feature_limits[f]

            # we create class in range (0, n_classes), but it should be in range (x, x+n_classes)
            if classes:
                y = y + min(classes)
            else:
                y = y + current_class_num
                current_class_num += created_classes
                y = [assign_class(y_, self.total_n_classes) for y_ in y]

            # randomly set 5% of the values to nan
            X.ravel()[np.random.choice(X.size, int(0.05 * X.size), replace=False)] = np.NaN

            # we want to assign the data in the hierarchy such that the missing features get already none values
            # this will make it easier for SPH and CPI
            X_with_NaNs = np.full((X.shape[0], len(total_sample_feature_set)), np.NaN)

            # X is created by just [0, ..., n_features] and now we map this back to the actual feature set
            # columns that are not filled will have the default NaN values
            for i, feature in enumerate(feature_set):
                X_with_NaNs[:, feature] = X[:, i]

            if X_with_NaNs.shape[0] != X.shape[0]:
                print(f"shape of X_with_NaNs is {X_with_NaNs.shape} and for X is {X.shape}")
            group.data = X_with_NaNs
            group.target = y
            class_counter = Counter(y)
            group.class_counter = class_counter
            group.gini = self.gini(y)

            if self.noise > 0 and n_samples > 30:
                group.noisy_target = flip_labels_uniform(np.array(y), self.noise)
            else:
                group.noisy_target = y

            # add data and labels to parent nodes as well
            traverse_node = group
            while traverse_node.parent:
                traverse_node = traverse_node.parent

                if traverse_node.data is not None:
                    traverse_node.data = np.concatenate([traverse_node.data, X_with_NaNs])
                    traverse_node.target = np.concatenate([traverse_node.target, y])

                else:
                    traverse_node.data = X_with_NaNs
                    traverse_node.target = y
                traverse_node.gini_index = self.gini(traverse_node.target)

            group_ids.extend([i for _ in range(X.shape[0])])

        return group_nodes

    def _check_features(self):
        group_nodes_to_create = self._get_leaf_nodes()
        current_used_feature_set = set([feature for group in group_nodes_to_create for feature in group.feature_set])

        # features that are currently not used by the groups
        features_not_used = np.setdiff1d(self.root.feature_set, list(current_used_feature_set))
        print(f"features that are currently not used: {features_not_used}")

        if len(features_not_used) > 0:

            for not_used_feature in features_not_used:
                # assign each feature to a group with weighted probability
                # the less features the groups have, the higher is the probability that they get the feature

                # assign probability that each group is chosen (1- (group_features/total_features))
                probability_choose_group = list(map(lambda x: 1 - (len(x.feature_set) / len(self.root.feature_set)),
                                                    group_nodes_to_create))
                # normalize probabilities so that they sum up to 1
                probability_normalized = [prob / sum(probability_choose_group) for prob in probability_choose_group]

                print(
                    f"probability distribution of the groups for feature {not_used_feature} is: {probability_normalized}")
                # choose random index with the given probabilities
                group_index = np.random.choice(len(group_nodes_to_create), 1, p=probability_normalized)
                assert len(group_index) == 1
                # convert list with "one" element to int
                group_index = group_index[0]

                group_node = group_nodes_to_create[group_index]
                group_node.feature_set.append(not_used_feature)
                # add also to parent nodes
                node = group_node
                while node.parent:
                    node = node.parent
                    if node.ancestors:
                        node.feature_set.append(not_used_feature)

                group_nodes_to_create[group_index] = group_node
        return group_nodes_to_create

    def _create_dataframe(self, groups):
        dfs = []
        levels = list(range(self.n_levels - 1))

        for group in groups:
            features_names = [f"F{f}" for f in range(self.n_features)]
            df = pd.DataFrame(group.data, columns=features_names)
            # assign classes and groups
            df["target"] = group.target
            df["noisy target"] = group.noisy_target
            df["group"] = group.node_id

            # assign higher values of the hierarchy to the group (i.e., the levels)
            for l in levels:
                df[f"level-{l}"] = group.hierarchy_level_values[l]
            dfs.append(df)

        return pd.concat(dfs).reset_index()

    def _remove_features_from_spec(self):
        # Determine how many features should be removed at each level
        # We do this such that the same amount is removed at each level
        n_levels = self.root.height
        features_to_remove_per_level = self._eq_div(int(self.features_remove_percent * len(self.root.feature_set)),
                                                    n_levels)

        parent_nodes = [self.root]
        for l in range(n_levels):
            new_parent_nodes = []
            for parent_node in parent_nodes:

                if not parent_node.n_classes:
                    self.logger.warning("Node without n_classes! This should not occur, please check the specified"
                                        " hiearchy again")

                childs = parent_node.get_child_nodes()

                # assert sum of childs are equal to parent node n_samples
                childs_n_samples_sum = sum(map(lambda x: x.n_samples, childs))
                print(parent_node)
                assert parent_node.n_samples == childs_n_samples_sum
                parent_features = parent_node.feature_set

                for child in childs:
                    # remove randomly the number of features as specified for this level
                    random_features = random.sample(parent_features, features_to_remove_per_level[l])
                    # take random features from parent and the rest are the features for children
                    child_feature_set = [f for f in parent_features if f not in random_features]
                    child.feature_set = child_feature_set
                    new_parent_nodes.append(child)
            parent_nodes = new_parent_nodes

        # parent nodes are now the group nodes
        return parent_nodes

    @staticmethod
    def _get_distribution_parameter(imbalance_degree):
        if imbalance_degree == "very_low":
            return 0.1
        elif imbalance_degree == "low":
            return 1
        elif imbalance_degree == "normal":
            return 1.5
        elif imbalance_degree == "high":
            return 3
        elif imbalance_degree == "very_high":
            return 10

    def _get_leaf_nodes(self):
        return anytree.search.findall(self.root, lambda x: x.is_leaf)

    def _assign_samples_classes_features(self):
        """
        Assigns the number of samples, classes and features to each node! Uses the pre-given information, especially the
        low_high_split to assign samples and classes to each node in the Hierarchy specification!
        Hence, we assume a pre-defined hierarchy specification that defines the structure but has not set the actual
        samples and classes
        :return:
        """

        current_nodes = [self.root]

        ######## Determine number of samples for each node ########
        while current_nodes:
            node = current_nodes.pop()
            if not node.children:
                continue

            n_children = len(node.children)
            min_samples_per_node = [2 for i in range(n_children)]
            remaining_samples = node.n_samples - sum(min_samples_per_node)
            samples_count = zipfian.rvs(a=1.5, size=remaining_samples, n=n_children)
            samples_count = list(Counter(samples_count).values())
            samples_count = [x+y for x,y in zip_longest(min_samples_per_node, samples_count, fillvalue=0)]
            samples_per_node = sorted(samples_count)

            classes_count = zipfian.rvs(a=1, size=int(node.n_classes * 1.5), n=n_children)
            n_classes_per_node = list(Counter(classes_count).values())

            if len(n_classes_per_node) < n_children:
                n_classes_per_node.extend([2 for i in range(n_children - len(n_classes_per_node))])

            n_classes_per_node = sorted(n_classes_per_node)

            ### Features per child node ##################################
            n_levels = self.root.height
            features_to_remove_per_level = self._eq_div(int(self.features_remove_percent * len(self.root.feature_set)),
                                                        n_levels)
            parent_features = node.feature_set
            features_to_remove_per_child = [
                random.sample(parent_features, features_to_remove_per_level[child.depth - 1])
                for child in node.children]
            features_per_child = [f for f in parent_features if f not in features_to_remove_per_child]

            # marks start and end range for classes
            classes_start, classes_end = node.classes
            current_class_start = classes_start

            ############### Assign classes, samples, features to child nodes #############
            for i, child in enumerate(node.children):
                n_classes = n_classes_per_node[i]
                # edge cases:
                if n_classes > node.n_classes:
                    # we have more classes than parent node
                    n_classes = node.n_classes
                elif n_classes < 2:
                    # we have 0 or 1 class
                    n_classes = 2

                n_samples = samples_per_node[i]
                child.feature_set = parent_features
                child.n_samples = n_samples
                child.n_classes = n_classes
                child.feature_set = features_per_child
                current_nodes.append(child)

                current_class_end = current_class_start + n_classes
                child.classes = (current_class_start, current_class_end)
                current_class_start = current_class_start + n_classes

            if current_class_end > classes_end:
                diff = current_class_end - classes_end
                for child in node.children:
                    child_cls_start, child_cls_end = child.classes
                    if child_cls_start - diff > classes_start:
                        child.classes = (child_cls_start - diff, child_cls_end - diff)

        ############################################################################

    def _generate_class_occurences(self, split_minority_majority=False):
        """
        We assume we have set the number of samples, classes, features per node in the hierarchy, so we can now define
        how often each class should occur for each node, i.e., the actual class distribution!
        :return:
        """

        # first, we need some statistics (percentiles) of the samples --> use this for class distribution
        low_samples = np.percentile([node.n_samples for node in PreOrderIter(self.root) if not node.children], 25)
        median_samples = np.percentile([node.n_samples for node in PreOrderIter(self.root) if not node.children], 50)
        high_samples = np.percentile([node.n_samples for node in PreOrderIter(self.root) if not node.children], 75)
        zero_percentile = np.percentile([node.n_samples for node in PreOrderIter(self.root) if not node.children], 0)

        print(
            f"5-percentile: {np.percentile([node.n_samples for node in PreOrderIter(self.root) if not node.children], 3)}")
        print(f"25-percentile: {low_samples}")
        print(f"50-percentile: {median_samples}")
        print(f"75-percentile: {high_samples}")

        group_nodes = self._get_leaf_nodes()
        for node in group_nodes:
            n_samples = node.n_samples

            #if n_samples < median_samples:
            class_occurences = [max(1, int(self.n_samples_total / 1000)) for _ in range(node.n_classes)]
            #else:
                # Each class should occur at least twice
             #   class_occurences = [max(2, 2 * int(self.n_samples_total / 1000)) for _ in range(node.n_classes)]

            remaining_samples = n_samples - sum(class_occurences)

            # we split into "minority" and "majority" classes --> We define high split majority classes and the rest
            # as minority classes.
            # However, it will (!) occur that at least one of the minority classes is then actually a majority class
            n_majority_samples = math.ceil(self.high_split * remaining_samples)
            n_minority_samples = math.floor(self.low_split * remaining_samples)
            n_majority_classes = math.floor(node.n_classes * self.low_split)
            n_minority_classes = math.ceil(node.n_classes * self.high_split)

            if n_majority_classes == 0:
                # Edge case if we do not have majority classes. Take at least one
                n_majority_classes += 1
                n_minority_classes -= 1

            # get parameter for distribution, based on defined imbalance degree
            distribution_parameter = self._get_distribution_parameter(self.imbalance_degree)

            if not split_minority_majority:
                #class_occurences = [max(1, int(self.n_samples_total / 1000)) for _ in range(node.n_classes)]
                remaining_samples = n_samples - sum(class_occurences)

                print(class_occurences)
                print(remaining_samples)
                print(node.n_samples)
                drawn_class_occurences = self.cls_prob_distribution(distribution_parameter,
                                                                    node.n_classes,
                                                                    size=remaining_samples,
                                                                    random_state=self.random_state)
                print(drawn_class_occurences)
                drawn_class_occurences = list(Counter(drawn_class_occurences).values())
                for i, d_cls_occ in enumerate(drawn_class_occurences):
                    class_occurences[i] += d_cls_occ
            else:
                # Generate occurences of minority and majority classes based on the defined distribution
                majority_classes_occurrences = self.cls_prob_distribution(distribution_parameter, n_majority_classes,
                                                                          size=n_majority_samples,
                                                                          random_state=self.random_state)
                minority_classes_occurences = self.cls_prob_distribution(distribution_parameter, n_minority_classes,
                                                                         size=n_minority_samples,
                                                                         random_state=self.random_state)

                majority_classes_occurrences = list(Counter(majority_classes_occurrences).values())
                minority_classes_occurences = list(Counter(minority_classes_occurences).values())

                # assign class occurences of minority and majority classes
                for i in range(0, len(majority_classes_occurrences)):
                    class_occurences[i] += majority_classes_occurrences[i]

                for j in range(0, len(minority_classes_occurences)):
                    class_occurences[len(majority_classes_occurrences) + j] += minority_classes_occurences[j]

                if self.imbalance_degree == "very_high":
                    min_occurence = np.min(class_occurences)
                    second_max, max_index = np.argpartition(class_occurences, -2)[-2:]
                    print(class_occurences)
                    if class_occurences[second_max] > min_occurence:
                        class_occurences[max_index] += class_occurences[second_max] - min_occurence
                        class_occurences[second_max] = min_occurence

                elif self.imbalance_degree == "very_low":
                    original_average = int(sum(class_occurences) / len(class_occurences))
                    # number of classes that are above average
                    n_max_classes = len([x for x in class_occurences if x > original_average])

                    # for each class above average, we run the following procedure
                    for i in range(n_max_classes):
                        # here we want to make the classes more balanced
                        # idea: move from majority one sample to each minority class
                        max_occurence = max(class_occurences)
                        max_index = class_occurences.index(max_occurence)
                        median = int(np.median(class_occurences))

                        print(max_occurence - median)
                        class_occurences[max_index] = median
                        for i in range(max_occurence - median):
                            min_occ_index = np.argmin(class_occurences)
                            class_occurences[min_occ_index] += 1

            assert sum(class_occurences) == node.n_samples

            # maybe we have rounding errors --> Add random samples until n_samples == class_occurences
            current_samples = sum(class_occurences)

            # instead of while loop, we could also add all samples that are not assigned to a class to only one class
            while current_samples < n_samples:
                max_index = np.argmax(class_occurences)
                class_occurences[max_index] += n_samples - current_samples
                current_samples = sum(class_occurences)

            assert sum(class_occurences) == node.n_samples

            node.class_occurences = class_occurences

        return group_nodes


def discrete_gauss(a, n, size, random_state):
    """
    Wrapper around a discretized gaussian distribution. No need to call .rvs.
    a and random_state are not used but are there for convenience to have a unique interface for the
    probability distributions.
    :param a:
    :param n:
    :param size:
    :param random_state:
    :return:
    """
    import scipy.special as sps
    f = np.array([sps.comb(n - 1, i, exact=True) for i in range(n)], dtype='O')
    f = np.float64(f)/np.float64(f).sum()

    if not np.allclose(f.sum(), 1.0):
        raise ValueError("The distribution sum is not close to 1.\n" 
                         "f.sum(): %s" % f.sum())
    f = np.random.choice(n, size, p=f)
    return f


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(10)
    result_for_imb = {}

    for imb in ImbalanceGenerator.imbalance_degrees:
        if imb != "normal":
            continue
        generator = ImbalanceGenerator(root=None, # describes the used taxonomy using an anytree instance
                                       n_samples_total=1000,
                                       imbalance_degree=imb,
                                       low_high_split=(0.3, 0.7),
                                       random_state=1234,
                                       distribution=zipfian.rvs
                                       )
        df = generator.generate_data_with_product_hierarchy(
            # distribution=boltzmann.rvs
        )
        print(RenderTree(generator.root))
        exit()
        print(len(df))

        from imblearn.over_sampling import SMOTE
        from imblearn.ensemble import BalancedRandomForestClassifier

        X = df[[f"F{i}" for i in range(100)]].to_numpy()
        y = df["target"]


        def gini(x):
            my_index = cm.Index()
            class_frequencies = np.array(list(Counter(x).values()))
            return my_index.gini(class_frequencies)


        print(sorted(list(Counter(y).values())))
        print(np.unique(y))
        print(f"Gini Index is: {gini(df['target'])}")
        try:
            X_train, X_test, y_train, y_true = train_test_split(X, y, train_size=750 / 1050,
                                                                stratify=y,
                                                                random_state=1234)
        except ValueError:
            X_train, X_test, y_train, y_true = train_test_split(X, y, train_size=750 / 1050,
                                                                random_state=1234)
        X_train = KNNImputer().fit_transform(X_train)

        #X_train, y_train = SMOTE(k_neighbors=5).fit_resample(X_train, y_train)
        rf = RandomForestClassifier(random_state=1234)
        rf.fit(X_train, y_train)
        X_test = KNNImputer().fit_transform(X_test)
        y_pred = rf.predict(X_test)
        # y_true = y_test
        print(f"Accuracy score is: {accuracy_score(y_pred, y_true)}")
        result_for_imb[imb] = (gini(df['target']), accuracy_score(y_pred, y_true), generator.root)
        print(f"micro: {f1_score(y_true, y_pred, average='micro')}")
        print(f"macro: {f1_score(y_true, y_pred, average='macro')}")
        print(f"weighted: {f1_score(y_true, y_pred, average='weighted')}")
        print(f"balanced-accuracy: {balanced_accuracy_score(y_true, y_pred)}")
        # print(f"prec_rec_fscore_support: {precision_recall_fscore_support(y_true, y_pred)}")

        #print(classification_report(y_true, y_pred, digits=3, zero_division=1))

        print(Counter(y))

        # root, df_test= make_unbalance_hierarchy(df_test=df_test, level_cutoff=2, root_node=generator.root,
        #                                        n_nodes_to_cutoff=2)
        # print(RenderTree(root))

    for k, v in result_for_imb.items():
        print("-------------------------")
        print(f"Imbalance Degree: {k}")
        print(RenderTree(v[2]))
        print(f"Gini: {v[0]}")
        print(f"Accuracy RF: {v[1]}")
        print("-------------------------")
