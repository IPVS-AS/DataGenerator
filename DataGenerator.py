import logging
import random
from collections import Counter
from typing import List

from anytree import RenderTree
from sklearn.datasets import make_classification, make_blobs

import numpy as np
import pandas as pd

from Hierarchy import Node, HardCodedHierarchy
import concentrationMetrics as cm

np.random.seed(1)
random.seed(1)


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
                                             root=HardCodedHierarchy().create_hardcoded_hierarchy()):

        """
        Main method of Data generation.
        Here, the data is generated according to various parameters.
        We mainly distinguish if we have an hierarchy given. This should be given with the root parameter that contains
        the root of an anytree. This root node can be used as representative for the whole tree.
        If we have a root, we should also have specified
        :param n_features: number of features to use for the overall generated dataset
        :param n_samples_total: number of samples that should be generated in the whole dataset
        :param n_levels: number of levels of the hierachy. Does not need to be specified if a hierarchy is already given!
        :param total_n_classes: number of classes for the whole dataset
        :param features_remove_percent: number of features to remove/ actually this means to have this number of percent
        as missing features in the whole dataset. Currently, this will be +5/6 percent.
        :param imbalance_degree: The degree of imbalabnce. Should be either 'normal', 'low' or 'high'. Here, normal means
        to actually use the same (hardcoded) hierarchy that is passed via the root parameter.
        'low' means to have a more imbalanced dataset and 'high' means to have an even more imbalanced dataset.
        :param root: Root node of a hierarchy. This should be a root node that represent an anytree and stands for the hierarchy.
        :return: A dataframe that contains the data and the hierarchy.
        The data is encoded via the feature columns F_0, ..., F_n_features.
        The hierarchy is implcitly given through the specific attributes that represent the hierarchy.
        """
        if root:
            # we have a hierarchy given, so we use this hierarchy
            return self._generate_product_hierarchy_from_specification(root=root, n_features=n_features,
                                                                       n_samples_total=n_samples_total,
                                                                       features_remove_percent=features_remove_percent,
                                                                       n_classes=total_n_classes,
                                                                       imbalance_degree=imbalance_degree)
        else:
            return self._generate_default_product_hierarchy(n_features=n_features, n_samples_total=n_samples_total,
                                                            total_n_classes=total_n_classes,
                                                            imbalance_degree=imbalance_degree,
                                                            features_remove_percent=features_remove_percent,
                                                            n_levels=n_levels)

    def _generate_product_hierarchy_from_specification(self, root=HardCodedHierarchy().create_hardcoded_hierarchy(),
                                                       n_features=100,
                                                       n_samples_total=1050, n_classes=84, features_remove_percent=0.2,
                                                       imbalance_degree="normal"):
        if imbalance_degree not in ImbalanceGenerator.imbalance_degrees:
            self.logger.error(f"imbalance_degree should be one of {ImbalanceGenerator.imbalance_degrees} but got"
                              f" {imbalance_degree}")
            self.logger.warning(f"Setting imbalance_degree to default 'normal'")
            imbalance_degree = "normal"

        # generate features for hierarchy
        features = list(range(n_features))
        root.feature_set = features

        # if n_samples not already specified, set to default value
        if not root.n_samples:
            root.n_samples = n_samples_total
        parent_nodes = [root]

        # Determine how many features should be removed at each level
        # We do this somehow that the same amount is removed at the same level
        n_levels = root.height
        features_to_remove_per_level = self._eq_div(int(features_remove_percent * n_features), n_levels)

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

        # the last parent nodes are the group nodes on the last level, i.e., which we want to create first
        group_nodes_to_create = parent_nodes

        current_used_feature_set = set(
            [feature for group in group_nodes_to_create for feature in group.feature_set])

        # features that are currently not used by the groups
        features_not_used = np.setdiff1d(root.feature_set, list(current_used_feature_set))
        print(f"features that are currently not used: {features_not_used}")

        if len(features_not_used) > 0:

            for not_used_feature in features_not_used:
                # assign each feature to a group with weighted probability
                # the less features the groups have, the higher is the probability that they get the feature

                # assign probability that each group is chosen (1- (group_features/total_features))
                probability_choose_group = list(map(lambda x: 1 - (len(x.feature_set) / len(root.feature_set)),
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

        # Now, we generate the actual data for the groups!
        groups = self._generate_product_groups_from_hierarchy(group_nodes_to_create, n_classes,
                                                              imbalance_degree=imbalance_degree)

        dfs = []
        levels = list(range(n_levels - 1))
        self.root = root
        # self.root.data = np.vstack(groups)

        # self.root.target = np.vstack

        for group in groups:
            features_names = [f"F{f}" for f in range(n_features)]
            df = pd.DataFrame(group.data, columns=features_names)
            # assign classes and groups
            df["target"] = group.target
            df["group"] = group.node_id

            # assign higher values of the hierarchy to the group (i.e., the levels)
            for l in levels:
                df[f"level-{l}"] = group.hierarchy_level_values[l]
            dfs.append(df)

        return pd.concat(dfs).reset_index()

    def _generate_default_product_hierarchy(self, n_features=100, n_samples_total=1050, n_levels=4, total_n_classes=84,
                                            features_remove_percent=0.2, imbalance_degree="normal"):
        """
        Generate specification on its own with "default" settings. Here, no hierarchy needs to be passed at all.
        This is mainly for future work to have a more generic data generator that does not require to have a specific
        hierarchy.
        At the moment there is also some code redundancy with the :func:_generate_product_hierarchy_from_specification.
        However, this will not be maintained at the moment as the focus is on the generation with the
        hierarchy specification.

        :param n_features:
        :param n_samples_total:
        :param n_levels:
        :param n_classes:
        :param features_remove_percent:
        :return:
        """

        # Basic idea is to first specify features, classes and samples for each node on each level
        # Second, the nodes are created based on the specification

        # to make it simple at first, we specify how many subgroups we want per node
        n_sub_groups_per_level = {l: 3 for l in range(n_levels - 1)}

        total_n_groups = 1
        # calculate how many groups we will have
        # I think this only works when each node in a level has the same number of child nodes
        for l in range(n_levels - 1):
            if l == 0:
                last_total_n_groups = 1
            else:
                last_total_n_groups = n_sub_groups_per_level[l - 1]
            total_n_groups += last_total_n_groups * n_sub_groups_per_level[l]

        # all features to use, i.e., for root node
        features = list(range(n_features))

        current_node_id = 0
        # create root node
        root = Node(node_id=current_node_id, n_samples=n_samples_total, feature_set=features, n_classes=total_n_classes)

        current_node_id += 1
        parent_nodes = [root]

        # Determine how many features should be removed at each level
        # We do this somehow that the same amount is removed at the same level
        features_to_remove_per_level = self._eq_div(int(features_remove_percent * n_features), n_levels)

        for l in range(n_levels - 1):
            child_nodes = []
            # iterate over each parent node and perform a "split" into subgroups
            for parent_node in parent_nodes:
                parent_features = parent_node.feature_set
                n_sub_groups = n_sub_groups_per_level[l]

                # range where to pick classes
                # if we have two subgroups the lower bound is to choose half of possible classes
                # such that each class will definitely occur.
                lower_bound_classes = int(parent_node.n_classes / n_sub_groups) + 1
                # highly influences vitalis approach
                upper_bound_classes = int(parent_node.n_classes * 0.6) + 1

                # choose n_classes for each children node
                n_classes_per_group = [random.choice(range(lower_bound_classes, upper_bound_classes + 1))
                                       for _ in range(n_sub_groups)]

                if sum(n_classes_per_group) < parent_node.n_classes:
                    # in this case not all classes are used, so we add them randomly to a group
                    remaining_classes = (parent_node.n_classes - sum(n_classes_per_group))
                    rand_index = random.choice(range(len(n_classes_per_group)))
                    n_classes_per_group[rand_index] = n_classes_per_group[rand_index] + remaining_classes

                # todo: class overlap has to be considered below

                # more classes should mean more samples
                n_samples_per_group = [int(parent_node.n_samples * (n_classes / sum(n_classes_per_group)))
                                       for n_classes in n_classes_per_group]
                diff = parent_node.n_samples - sum(n_samples_per_group)
                i = 0
                while diff > 0:
                    n_samples_per_group[i % len(n_samples_per_group)] = n_samples_per_group[
                                                                            i % len(n_samples_per_group)] + 1
                    diff = diff - 1

                for n_samples, n_classes in zip(n_samples_per_group, n_classes_per_group):
                    # remove randomly the number of features as specified for this level
                    random_features = random.sample(parent_features, features_to_remove_per_level[l])
                    # take random features from parent and the rest are the features for children
                    child_feature_set = [f for f in parent_features if f not in random_features]

                    child = Node(node_id=current_node_id, n_samples=n_samples, parent=parent_node,
                                 feature_set=child_feature_set, n_classes=n_classes)
                    current_node_id += 1

                    # append to the child nodes --> this will become the new parent nodes in next iteration
                    child_nodes.append(child)

            # update parent nodes to child nodes
            parent_nodes = child_nodes

        # the last parent nodes are the group nodes on the last level, i.e., which we want to create first
        group_nodes_to_create = parent_nodes

        current_used_feature_set = set([feature for group in group_nodes_to_create for feature in group.feature_set])

        # features that are currently not used by the groups
        features_not_used = np.setdiff1d(root.feature_set, list(current_used_feature_set))
        print(f"features that are currently not used: {features_not_used}")

        if len(features_not_used) > 0:

            for not_used_feature in features_not_used:
                # assign each feature to a group with weighted probability
                # the less features the groups have, the higher is the probability that they get the feature

                # assign probability that each group is chosen (1- (group_features/total_features))
                probability_choose_group = list(map(lambda x: 1 - (len(x.feature_set) / len(root.feature_set)),
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

        groups = self._generate_product_groups_default(group_nodes_to_create, total_n_classes)

        dfs = []
        levels = list(range(n_levels - 1))
        self.root = root
        # self.root.data = np.vstack(groups)

        # self.root.target = np.vstack

        for group in groups:
            features_names = [f"F{f}" for f in range(n_features)]
            df = pd.DataFrame(group.data, columns=features_names)
            # assign classes and groups
            df["target"] = group.target
            df["group"] = group.node_id

            # assign higher values of the hierarchy to the group (i.e., the levels)
            for l in levels:
                df[f"level-{l}"] = group.hierarchy_level_values[l]
            dfs.append(df)

        return pd.concat(dfs).reset_index()

    def _generate_product_groups_from_hierarchy(self, group_nodes: List[Node], total_n_classes,
                                                imbalance_degree="normal"):
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
        # bottom up approach
        for i, group in enumerate(group_nodes):
            feature_set = group.feature_set
            n_features = len(feature_set)
            n_samples = group.n_samples
            n_classes = group.n_classes
            classes = group.classes

            # take a random feature along we move the next group
            feature_to_move = np.random.choice(feature_set)
            feature_limits[feature_to_move] += 1

            if group.class_occurences:
                occurences = group.class_occurences

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
                    median = int(np.median(occurences))

                    for i, occ in enumerate(occurences):
                        # check if we have at least two samples and this is not the max_occurence.
                        if occ > median and occ < max_occurence:
                            # This can easily be changed to remove exactly 5%
                            new_occurences[max_index] += occ - median
                            new_occurences[i] = median
                            samples_removed += occ - median

                        # do we have removed 5% of samples?
                        if samples_removed >= 0.05 * n_samples and imbalance_degree=='high':
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
                        # here we want to make the classes more imbalanced
                        # idea: move from majority one sample to each minority class
                        max_occurence = max(occurences)
                        max_index = occurences.index(max_occurence)
                        new_occurences = occurences.copy()
                        median = np.median(occurences)
                        average = sum(occurences) / len(occurences)
                        # important to take integers, we cannot divide float into n buckets later on
                        median_or_average = int(average) if median == 1 else int(median)

                        if len(new_occurences) < max_occurence:
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
                # if we do not have specified occurences, we use a simple way at the moment, just make the classes somehow equally.
                # Yet, this will yield a much to imbalanced dataset (gini ~ 0.3)
                if n_classes == 2:
                    weights = [0.95, 0.05]
                elif n_classes == 3:
                    weights = [0.6, 0.3, 0.1]
                else:
                    # we define 70% as the three majority classes
                    majority_weights = [0.4, 0.2, 0.1]
                    sum_min_weights = 1 - sum(majority_weights)
                    n_min_classes = n_classes - len(majority_weights)
                    # the rest 30% are equally divided to the other classes
                    minority_weights = [sum_min_weights / n_min_classes for _ in range(n_min_classes)]
                    weights = majority_weights + minority_weights

            # set number of informative features
            n_informative = n_features - 1

            # The questions is, if we need this function if we have e.g., less than 15 samples. Maybe for this, we
            # can create the patterns manually?
            X, y = make_classification(n_samples=n_samples, n_classes=n_classes,
                                       n_clusters_per_class=1,
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
            if created_classes < n_classes:
                print(
                    f"should create {n_classes} and have created {created_classes} classes for n_samples {n_samples} "
                    f"and weights={weights}")

            if created_samples < n_samples:
                print(f"should create {n_samples} and have created {created_samples}")

            # normalize x into [0,1] interval
            X = (X - X.min(0)) / X.ptp(0)

            # also add a random number that makes the intersection between groups "softer"
            rand_num = random.random()
            for i, f in enumerate(feature_set):
                # move each feature by its feature limits
                X[:, i] = X[:, i] + feature_limits[f]  # - rand_num

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
                traverse_node.gini = self.gini(traverse_node.target)

            resulting_groups.append(X)
            target_classes.extend(y)
            group_ids.extend([i for _ in range(X.shape[0])])

        return group_nodes

    def _generate_product_groups_default(self, group_nodes: List[Node], total_n_classes):
        resulting_groups = []
        target_classes = []
        group_ids = []

        # get set of all features. We need this to keep track of the feature limits of all groups
        total_sample_feature_set = set([feature for group in group_nodes for feature in group.feature_set])

        # save limits of each feature --> first all are 0.0
        feature_limits = {feature: 0 for feature in total_sample_feature_set}

        current_class_num = 0

        # bottom up approach
        for i, group in enumerate(group_nodes):
            feature_set = group.feature_set
            n_features = len(feature_set)
            n_samples = group.n_samples
            n_classes = group.n_classes
            classes = group.classes

            # take a random feature along we move the next group
            feature_to_move = np.random.choice(feature_set)
            feature_limits[feature_to_move] += 1

            # if n_samples_per_class_avg <= 3:
            #    use_blobs = True
            if n_classes == 2:
                weights = [0.95, 0.05]
            elif n_classes == 3:
                weights = [0.6, 0.3, 0.1]
            else:
                # we define 70% as the three majority classes
                majority_weights = [0.4, 0.2, 0.1]
                sum_min_weights = 1 - sum(majority_weights)
                n_min_classes = n_classes - len(majority_weights)
                # the rest 30% are equally divided to the other classes
                minority_weights = [sum_min_weights / n_min_classes for _ in range(n_min_classes)]
                weights = majority_weights + minority_weights

            n_informative = n_features - 1

            X, y = make_classification(n_samples=n_samples, n_classes=n_classes,
                                       n_clusters_per_class=1,
                                       n_features=n_features, n_repeated=0, n_redundant=0,
                                       n_informative=n_informative,
                                       weights=weights,
                                       # higher value can cause less classes to be generated
                                       flip_y=0.01,
                                       random_state=0,
                                       # class_sep=0.8,
                                       hypercube=True
                                       # shift=None,
                                       # scale=None
                                       )
            created_classes = len(np.unique(y))
            created_samples = X.shape[0]
            if created_classes < n_classes:
                print(f"should create {n_classes} and have created {created_classes} for n_samples {n_samples}")

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

            resulting_groups.append(X)
            target_classes.extend(y)
            group_ids.extend([i for _ in range(X.shape[0])])

        return group_nodes


if __name__ == '__main__':
    def gini(x):
        my_index = cm.Index()
        class_frequencies = np.array(list(Counter(x).values()))
        return my_index.gini(class_frequencies)



    print('----------------------------------------------------------------------------------------')
    print('------------------------------Very Low Imbalance Degree --------------------------------')

    generator = ImbalanceGenerator()
    df = generator.generate_data_with_product_hierarchy(imbalance_degree="very_low")
    root = generator.root
    y_true = df['target'].to_numpy()
    gini_value = gini(y_true)
    print(RenderTree(root))
    print(f'Gini index for whole data is: {gini_value}')
    print('----------------------------------------------------------------------------------------')

    print('----------------------------------------------------------------------------------------')
    print('------------------------------Low Imbalance Degree -------------------------------------')

    generator = ImbalanceGenerator()
    df = generator.generate_data_with_product_hierarchy(imbalance_degree="low")
    root = generator.root
    y_true = df['target'].to_numpy()
    gini_value = gini(y_true)
    print(RenderTree(root))
    print(f'Gini index for whole data is: {gini_value}')
    print('----------------------------------------------------------------------------------------')

    print('----------------------------------------------------------------------------------------')
    print('------------------------------Normal Imbalance Degree -------------------------------------')
    generator = ImbalanceGenerator()
    df = generator.generate_data_with_product_hierarchy(imbalance_degree="normal")
    y_true = df['target'].to_numpy()
    gini_value = gini(y_true)

    root = generator.root
    print(RenderTree(root))
    print(f'Gini index for whole data is: {gini_value}')
    print('----------------------------------------------------------------------------------------')

    print('----------------------------------------------------------------------------------------')
    print('------------------------------High Imbalance Degree -------------------------------------')
    generator = ImbalanceGenerator()
    df = generator.generate_data_with_product_hierarchy(imbalance_degree="high")
    y_true = df['target'].to_numpy()
    gini_value = gini(y_true)

    root = generator.root
    print(RenderTree(root))
    print(f'Gini index for whole data is: {gini_value}')
    print('----------------------------------------------------------------------------------------')

    print('----------------------------------------------------------------------------------------')
    print('------------------------Very High Imbalance Degree -------------------------------------')
    generator = ImbalanceGenerator()
    df = generator.generate_data_with_product_hierarchy(imbalance_degree="very_high")
    y_true = df['target'].to_numpy()
    gini_value = gini(y_true)

    root = generator.root
    print(RenderTree(root))
    print(f'Gini index for whole data is: {gini_value}')
    print('----------------------------------------------------------------------------------------')
