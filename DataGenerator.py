import logging
import math
import random
from collections import Counter
from typing import List

import imblearn
from anytree import RenderTree, PreOrderIter
from skclean.simulate_noise import flip_labels_uniform
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_blobs

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
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

    def __init__(self):
        self.root = None
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

    def generate_data_with_product_hierarchy(self, n_features=100, n_samples_total=1050, n_levels=4, total_n_classes=84,
                                             features_remove_percent=0.2, imbalance_degree="normal",
                                             root=HardCodedHierarchy().create_hardcoded_hierarchy(), noise=0,
                                             low_high_split=(0.3,0.7)
                                             # , seed=random_state
                                             ):

        """
        Main method of Data generation.
        Here, the data is generated according to various parameters.
        We mainly distinguish if we have an hierarchy given. This should be given with the root parameter that contains
        the root of an anytree. This root node can be used as representative for the whole tree.
        If we have a root, we should also have specified
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
        :return: A dataframe that contains the data and the hierarchy.
        The data is encoded via the feature columns F_0, ..., F_n_features.
        The hierarchy is implicitly given through the specific attributes that represent the hierarchy.
        """
        if imbalance_degree not in ImbalanceGenerator.imbalance_degrees:
            self.logger.error(f"imbalance_degree should be one of {ImbalanceGenerator.imbalance_degrees} but got"
                              f" {imbalance_degree}")
            self.logger.warning(f"Setting imbalance_degree to default 'normal'")
            imbalance_degree = "normal"

        # np.random.seed(1)

        if not root:
            features = list(range(n_features))
            root = Node(node_id="0", n_samples=n_samples_total, feature_set=features, n_classes=total_n_classes,
                        classes=(0, total_n_classes - 1))
            self.root = root
            # generate default hierarchy
            self._generate_default_hierarchy_spec(n_levels=n_levels, low_high_split=low_high_split)
        else:
            self._get_hardcoded_hierarchy_spec(root=root, n_features=n_features,
                                               n_samples_total=n_samples_total)

        # remove features
        group_nodes_to_create = self._remove_features_from_spec(features_remove_percent)
        # check if all features are in the whole data, i.e., we have n_features
        group_nodes_to_create = self._check_features(group_nodes_to_create)

        # actual generation of the data
        # adjust class distribution is inside data generation, Todo: separate that
        groups = self._generate_groups_from_hierarchy_spec(group_nodes_to_create, total_n_classes,
                                                           imbalance_degree=imbalance_degree,
                                                           n_samples_total=n_samples_total, noise=0)

        return self._create_dataframe(n_levels, groups, n_features)

    def _get_hardcoded_hierarchy_spec(self, root=HardCodedHierarchy().create_hardcoded_hierarchy(),
                                      n_features=100,
                                      n_samples_total=1050,
                                      # seed=random_state
                                      ):
        # np.random.seed(seed)

        # generate features for hierarchy
        features = list(range(n_features))
        root.feature_set = features

        # if n_samples not already specified, set to default value
        if not root.n_samples:
            root.n_samples = n_samples_total
        self.root = root

        return [node for node in PreOrderIter(root) if not node.children]

    def _generate_default_hierarchy_spec(self, n_levels, low_high_split=(0.3, 0.7)):
        node_per_level_per_node = {}
        features = self.root.feature_set

        n_child_nodes = 0
        for l in range(0, n_levels):
            n_parents = n_child_nodes
            if l == 0:
                n_child_nodes = 1
                n_parents = 1
            else:
                n_child_nodes = n_parents * (l + 1)
            node_per_level_per_node[l] = n_child_nodes / (n_parents)

        low_split = low_high_split[0]
        high_split = low_high_split[1]

        current_nodes = [self.root]

        ######## Generate nodes for each level with samples and number of classes ########
        for l in range(1, n_levels):
            n_nodes_per_node = node_per_level_per_node[l]
            new_nodes = []
            counter = 0

            for j, node in enumerate(current_nodes):
                parent_features = node.feature_set

                n_nodes_per_node = int(n_nodes_per_node)
                if n_nodes_per_node == 2:
                    splitting = [low_split, high_split]
                elif n_nodes_per_node == 3:
                    splitting = [low_split * low_split, low_split * high_split, high_split]
                    """
                    if j % 2 == 1:
                        splitting = [low_split * 1 / 3, low_split * 2 / 3, high_split]
                    else:
                        splitting = [low_split, high_split * 1 / 3, high_split * 2 / 3]
                    """
                elif n_nodes_per_node == 4:
                    splitting = [low_split * low_split, low_split * high_split, high_split * low_split, high_split * high_split]
                elif n_nodes_per_node == 5:
                    splitting = [low_split / 3, low_split / 3, low_split / 3, high_split * 1 / 2, high_split / 2]
                elif n_nodes_per_node == 6:
                    splitting = [low_split / 3, low_split / 3, low_split / 3, high_split / 3, high_split / 3,
                                 high_split / 3]

                # determine samples for each node
                samples_per_node = [int(splitting[i] * node.n_samples) for i in range(n_nodes_per_node)]

                weight_interval = [1 + .5 / i for i in range(1, n_nodes_per_node + 1)]
                print(weight_interval)

                sample_based_weight = random.choices(population=weight_interval, weights=samples_per_node,
                                                     k=n_nodes_per_node)
                print(sample_based_weight)

                # Assign the number of classes for the nodes based on the number of samples,
                # number of classes from parent and the splitting percentage
                n_classes_per_node = [int(splitting[i] * node.n_classes * sample_based_weight[i]) for i in
                                      range(n_nodes_per_node)]

                # due to rounding errors there are samples missing, add them randomly
                while sum(samples_per_node) < node.n_samples:
                    rand_ind = random.randint(0, len(samples_per_node) - 1)
                    samples_per_node[rand_ind] += 1

                for i in range(int(n_nodes_per_node)):

                    n_classes = n_classes_per_node[i]

                    # edge cases:
                    if n_classes > node.n_classes:
                        # we have more classes than parent node
                        n_classes = node.n_classes
                    elif n_classes < 2:
                        # we have 0 or 1 class
                        n_classes = 2

                    n_samples = samples_per_node[i]

                    # if n_samples > np.median()
                    new_node = Node(node_id=f"{counter}", n_samples=n_samples,
                                    # we assign parent features first and remove them later on
                                    feature_set=parent_features,
                                    parent=node,
                                    n_classes=n_classes)
                    counter += 1
                    new_nodes.append(new_node)
            current_nodes = new_nodes
        ############################################################################

        ############################################################################
        ######## Generate class occurences for each node ########

        # first, we need some statistics (percentiles) of the samples --> use this for class distribution
        low_samples = np.percentile([node.n_samples for node in PreOrderIter(self.root) if not node.children], 15)
        median_samples = np.percentile([node.n_samples for node in PreOrderIter(self.root) if not node.children], 50)
        high_samples = np.percentile([node.n_samples for node in PreOrderIter(self.root) if not node.children], 75)
        zero_percentile = np.percentile([node.n_samples for node in PreOrderIter(self.root) if not node.children], 0)

        print(
            f"5-percentile: {np.percentile([node.n_samples for node in PreOrderIter(self.root) if not node.children], 3)}")
        print(f"25-percentile: {low_samples}")
        print(f"50-percentile: {median_samples}")
        print(f"75-percentile: {high_samples}")

        group_nodes = [node for node in PreOrderIter(self.root) if not node.children]
        for node in group_nodes:
            n_samples = node.n_samples
            # Each class should occur at least twice
            class_occurences = [2 for _ in range(node.n_classes)]
            remaining_samples = n_samples - sum(class_occurences)
            print(n_samples)
            # Dependent on the number of samples in the group, we define how many majority classes
            # and how often they should occur
            if remaining_samples < median_samples:

                if node.n_samples < low_samples:
                    # take only one majority class
                    majority_classes_occurrences = [1]
                else:
                    # We have not much samples, but still over 25 percentile, take 2 majority classes
                    majority_classes_occurrences = [math.ceil(0.7 * remaining_samples),
                                                    math.floor(0.3 * remaining_samples)]

            elif n_samples < high_samples:
                majority_classes_occurrences = [math.ceil(0.7 * remaining_samples), math.floor(0.2 * remaining_samples),
                                                math.floor(0.1 * remaining_samples)]

            else:
                lower_class_occurencs = int(low_samples)
                if lower_class_occurencs == 2:
                    # if this is 2, we have an empty set in the following!
                    lower_class_occurencs = 3
                class_occurences = np.random.choice(list(range(2, lower_class_occurencs)),
                                                    int(len(class_occurences)/2))

                class_occurences = np.concatenate((class_occurences, [2 for _ in range(node.n_classes - len(class_occurences))]), axis=0)
                print(class_occurences)
                # class_occurences = [1 for _ in class_occurences]
                remaining_samples = node.n_samples - sum(class_occurences)
                majority_classes_occurrences = [math.ceil(0.7 * remaining_samples), math.floor(0.15 * remaining_samples),
                                                math.floor(0.075 * remaining_samples),
                                                math.floor(0.075 * remaining_samples)]

                print(class_occurences)
                print(node.n_classes)
            print(class_occurences)
            print(majority_classes_occurrences)
            print(n_samples)
            # Now take random classes and assign them as the majority classes defined above
            rand_inds = np.random.choice(node.n_classes, len(majority_classes_occurrences), replace=False)
            for i, rand_ind in enumerate(rand_inds):
                class_occurences[rand_ind] += majority_classes_occurrences[i]

            # maybe we have rounding errors --> Add random samples until n_samples == class_occurences
            current_samples = sum(class_occurences)
            # instead of while loop, we could also add all samples that are not assigned to a class to only one class
            while current_samples < n_samples:
                class_probability = [c / current_samples for c in class_occurences]
                rand_ind = np.random.choice(len(class_occurences), p=class_probability)
                class_occurences[rand_ind] += n_samples - current_samples
                current_samples = sum(class_occurences)

            node.class_occurences = class_occurences
        return group_nodes

    def _generate_groups_from_hierarchy_spec(self, group_nodes: List[Node], total_n_classes,
                                             imbalance_degree="normal", noise=0, n_samples_total=1050):
        """
        Generates the product groups. That is, here is the actual data generated.
        For each group, according to the number of classes, samples and features the data is generated.
        Note, that at the moment we also make us of the class occurences! If these are not specified in the hierarchy,
        we have to make a different strategy. I will do that in the near future.

        :param group_nodes: a list of of the nodes from the hierarchy specification.
        Each node should have at least specified the number of samples and classes.
        The features do not need necessarily to be specified at the moment.
        :param total_n_classes: Number of total classes to generate in the whole dataset
        :param imbalance_degree: Degree of imbalance. Should be one of 'low', 'normal', 'high'.
        :return: group_nodes: list of nodes that now have set the data and target attributes.
        Here, we only return the group nodes, but the data and target of the parent nodes is also set!
        """
        resulting_groups = []
        target_classes = []
        group_ids = []

        # get set of all features. We need this to keep track of the feature limits of all groups
        total_sample_feature_set = set([feature for group in group_nodes for feature in group.feature_set])

        # save limits of each feature --> first all are 0.0
        feature_limits = {feature: 0 for feature in total_sample_feature_set}

        current_class_num = 0

        n_samples_to_generate = sum([int(group.n_samples * n_samples_total / 1050) for group in group_nodes])

        remaining_samples = []
        if n_samples_total > n_samples_to_generate:
            # share the remaining samples among the groups
            remaining_samples = self._eq_div(n_samples_total - n_samples_to_generate, len(group_nodes))

        # bottom up approach
        for i, group in enumerate(group_nodes):
            feature_set = group.feature_set
            n_features = len(feature_set)
            n_samples = group.n_samples
            mult_factor = n_samples_total / 1050
            n_samples = int(n_samples * mult_factor)

            # add samples that are missing due to rounding errors
            if len(remaining_samples) > i:
                n_samples += remaining_samples[i]

            n_classes = group.n_classes
            classes = group.classes

            # take a random feature along we move the next group
            feature_to_move = np.random.choice(feature_set)
            feature_limits[feature_to_move] += 1

            if group.class_occurences is not None:
                occurences = group.class_occurences
                occurences = list(occurences)

                # special condition with n_samples < 15 to cover cases where n_classes=9 and n_samples=12
                if imbalance_degree == 'normal' or n_samples < 15:
                    # do nothing in this case
                    pass

                elif imbalance_degree == 'high' or imbalance_degree == 'very_high':

                    # get max occurence and the index for it
                    max_occurence = max(occurences)
                    max_index = occurences.index(max_occurence)

                    # keep track if 5% are removed
                    samples_removed = 0
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
                            samples_removed += occ - median_or_average

                        # do we have removed 5% of samples?
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

                # Calculate the weights (in range [0,1]) from the occurrences.
                # The weights are needed for the sklearn function 'make_classification'
                weights = [occ / sum(occurences) for occ in occurences]
            else:
                logging.error("No occurences specified!")

            # set number of informative features
            n_informative = n_features - 1

            # The questions is, if we need this function if we have e.g., less than 15 samples. Maybe for this, we
            # can create the patterns manually?
            X, y = make_classification(n_samples=n_samples, n_classes=n_classes,
                                       n_clusters_per_class=2,
                                       n_features=n_features, n_repeated=0, n_redundant=0,
                                       n_informative=n_informative,
                                       weights=weights,
                                       random_state=0,
                                       # higher value can cause less classes to be generated
                                       # flip_y=0.01,
                                       # class_sep=0.8,
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

            # we create class in range (0, n_classes), but it can be in range (x, x+n_classes)
            if classes:
                y = y + min(classes)
            else:
                y = y + current_class_num
            current_class_num += created_classes
            y = [assign_class(y_, total_n_classes) for y_ in y]

            # randomly set 5% of the values to nan
            X.ravel()[np.random.choice(X.size, int(0.05 * X.size), replace=False)] = np.NaN

            # we want to assign the data in the hierarchy such that the missing features get already none values
            # this will make it easier for SPH and CPI
            X_with_NaNs = np.full((X.shape[0], len(total_sample_feature_set)), np.NaN)

            # X is created by just [0, ..., n_features] and now we map this back to the actual feature set
            # columsn that are not filled will have the default NaN values
            for i, feature in enumerate(feature_set):
                X_with_NaNs[:, feature] = X[:, i]

            if X_with_NaNs.shape[0] != X.shape[0]:
                print(f"shape of X_with_NaNs is {X_with_NaNs.shape} and for X is {X.shape}")
            group.data = X_with_NaNs
            group.target = y
            class_counter = Counter(y)
            group.class_counter = class_counter
            group.gini = self.gini(y)

            if noise > 0 and n_samples > 30:
                group.noisy_target = flip_labels_uniform(np.array(y), noise)
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

            resulting_groups.append(X)
            target_classes.extend(y)
            group_ids.extend([i for _ in range(X.shape[0])])

        return group_nodes

    def _check_features(self, group_nodes_to_create):
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

    def _create_dataframe(self, n_levels, groups, n_features):
        dfs = []
        levels = list(range(n_levels - 1))

        for group in groups:
            features_names = [f"F{f}" for f in range(n_features)]
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

    def _remove_features_from_spec(self, features_remove_percent):
        # Determine how many features should be removed at each level
        # We do this such that the same amount is removed at each level
        n_levels = self.root.height
        features_to_remove_per_level = self._eq_div(int(features_remove_percent * len(self.root.feature_set)), n_levels)

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


if __name__ == '__main__':
    generator = ImbalanceGenerator()
    df = generator.generate_data_with_product_hierarchy(root=None, imbalance_degree="very_high", low_high_split=(0.3, 0.7))
    print(RenderTree(generator.root))
    print(len(df))
    def gini(x):
        my_index = cm.Index()
        class_frequencies = np.array(list(Counter(x).values()))
        return my_index.gini(class_frequencies)
    print(gini(df["target"]))
    df_train, df_test = train_test_split(df, train_size=0.7, stratify=df["target"])
    #root, df_test= make_unbalance_hierarchy(df_test=df_test, level_cutoff=2, root_node=generator.root,
    #                                        n_nodes_to_cutoff=2)
    #print(RenderTree(root))
