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
from sklearn import svm
from sklearn.datasets import make_classification, make_blobs
from itertools import zip_longest

import numpy as np
import pandas as pd
from Hierarchy import Node, HardCodedHierarchy, EngineTaxonomy
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
    imbalance_degrees = ['very_balanced', 'balanced', 'medium', 'imbalanced', 'very_imbalanced']

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

    def __init__(self, n_features=100, n=1050,
                 n_levels=4,
                 c=84,
                 features_remove_percent=0,
                 gs=1,
                 n_group_features=1,
                 cls_imbalance="medium",
                 group_imbalance="medium",
                 root=HardCodedHierarchy().create_hardcoded_hierarchy(),
                 noise=0,
                 distribution=zipfian.rvs,
                 class_overlap=1.5,
                 hardcoded=False,
                 random_state=1234):
        """
        :param n_features: number of features to use for the overall generated dataset
        :param n: number of samples that should be generated in the whole dataset
        :param n_levels: number of levels of the hierarchy. Does not need to be specified if a hierarchy is already given!
        :param c: number of classes for the whole dataset
        :param features_remove_percent: number of features to remove/ actually this means to have this number of percent
        as missing features in the whole dataset. Currently, this will be +5/6 percent.
        :param cls_imbalance: The degree of imbalance. Should be either 'medium', 'low' or 'high'. Here, medium means
        to actually use the same (hardcoded) hierarchy that is passed via the root parameter.
        'low' means to have a more imbalanced dataset and 'high' means to have an even more imbalanced dataset.
        :param root: Root node of a hierarchy. This should be a root node that represent an anytree and stands for the hierarchy.
        :param distribution: Distribution to use. In the moment, either boltzman.rvs or zipfian.rvs are tested from the scipy.stats module!
        :param noise: Percentage of noise to generate (in [0,1])
        :param random_state:
        """
        self.cls_imbalance = cls_imbalance
        self.group_imbalance = group_imbalance
        self.hardcoded = hardcoded
        self.root = root
        self.n_features = n_features
        self.n_levels = n_levels
        self.prob_distribution = distribution
        self.group_separation = gs
        self.n_group_features = n_group_features

        self.n_samples_total = n
        self.total_n_classes = c
        self.random_state = random_state
        self.class_overlap = class_overlap

        self.noise = noise
        self.features_remove_percent = features_remove_percent

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

        if isinstance(self.cls_imbalance, str) and self.cls_imbalance not in ImbalanceGenerator.imbalance_degrees:
            self.logger.error(f"cls_imbalance_degree should be one of {ImbalanceGenerator.imbalance_degrees} but got"
                              f" {self.cls_imbalance}")
            self.logger.warning(f"Setting cls_imbalance_degree to default 'medium'")
            self.cls_imbalance = "medium"
        if isinstance(self.group_imbalance, str) and self.group_imbalance not in ImbalanceGenerator.imbalance_degrees:
            self.logger.error(f"group_imbalance should be one of {ImbalanceGenerator.imbalance_degrees} but got"
                              f" {self.group_imbalance}")
            self.logger.warning(f"Setting group_imbalance to default 'medium'")
            self.group_imbalance = "medium"

        # np.random.seed(1)

        features = list(range(self.n_features))
        if self.hardcoded:
            self._get_hardcoded_hierarchy_spec()

        else:
            if not self.root:
                self.root = Node(node_id="0", n_samples=self.n_samples_total, feature_set=features,
                                 n_classes=self.total_n_classes,
                                 classes=(0, self.total_n_classes))
                # generate default hierarchy
                self._generate_default_hierarchy_spec()
            else:
                self.root.n_samples = self.n_samples_total
                self.root.n_classes = self.total_n_classes
                self.root.feature_set = features
                self.root.classes = (0, self.total_n_classes)
            self._assign_samples_classes_features()
            self._generate_class_occurences()

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
            imbalance_degree = self.cls_imbalance
            n_samples = group.n_samples
            occurences = group.class_occurences
            # special condition with n_samples < 15 to cover cases where n_classes=9 and n_samples=12
            if imbalance_degree == 'medium' and n_samples > 15:
                # do nothing in this case
                pass

            elif imbalance_degree == 'imbalanced' or imbalance_degree == 'very_imbalanced':

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

                    if imbalance_degree == 'imbalanced':
                        break
                occurences = new_occurences

            elif imbalance_degree == 'balanced' or imbalance_degree == 'very_balanced':
                original_average = sum(occurences) / len(occurences)
                n_max_classes = 1
                if imbalance_degree == 'very_balanced':
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

            # take random feature(s) along we move the next group
            feature_to_move = np.random.choice(feature_set, self.n_group_features)
            for feature in feature_to_move:
                feature_limits[feature] += self.group_separation

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
                                       # class_sep=10,
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
            # X.ravel()[np.random.choice(X.size, int(0.05 * X.size), replace=False)] = np.NaN

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
        group_nodes_to_create = list(self._get_leaf_nodes())
        current_used_feature_set = set([feature for group in group_nodes_to_create for feature in group.feature_set])

        # features that are currently not used by the groups
        features_not_used = np.setdiff1d(self.root.feature_set, list(current_used_feature_set))
        print(f"features that are currently not used: {features_not_used}")

        if len(features_not_used) > 0:

            for not_used_feature in features_not_used:
                # assign each feature to a group with weighted probability
                # the less features the groups have, the higher is the probability that they get the feature

                # choose random index with the given probabilities
                group_index = np.random.choice(len(group_nodes_to_create), 1)
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
            n_children = len(node.children)
            if n_children == 0:
                continue

            min_samples_per_node = [1 for i in range(n_children)]
            remaining_samples = node.n_samples - sum(min_samples_per_node)
            group_distribution = self._get_group_distribution_parameter(self.group_imbalance)

            samples_count = self.prob_distribution(a=group_distribution, size=remaining_samples, n=n_children)
            samples_count = list(Counter(samples_count).values())
            samples_count = [x + y for x, y in zip_longest(min_samples_per_node, samples_count, fillvalue=0)]
            samples_per_node = sorted(samples_count)

            classes_count = self.prob_distribution(a=group_distribution, size=int(node.n_classes * self.class_overlap),
                                                   n=n_children)
            n_classes_per_node = list(Counter(classes_count).values())

            if len(n_classes_per_node) < n_children:
                n_classes_per_node.extend([2 for i in range(n_children - len(n_classes_per_node))])

            n_classes_per_node = sorted(n_classes_per_node)

            ### Features per child node ##################################
            n_levels = self.root.height

            features_to_remove_per_level = self._eq_div(int(self.features_remove_percent * len(self.root.feature_set)),
                                                        n_levels)
            parent_features = node.feature_set
            if self.features_remove_percent > 0:
                features_to_remove_per_child = [
                    random.sample(parent_features, features_to_remove_per_level[child.depth - 1])
                    for child in node.children]
                features_per_child = [f for f in parent_features if f not in features_to_remove_per_child]
            else:
                features_per_child = parent_features

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

                current_class_end = current_class_start + n_classes

                current_diff = (current_class_end - classes_end)
                while current_class_end > classes_end and current_class_start - current_diff >= 0:
                    current_diff = (current_class_end - classes_end)
                    current_class_start -= (current_diff)
                    current_class_end -= (current_diff)
                child.classes = (current_class_start, current_class_end)
                current_class_start = current_class_start + n_classes

                current_nodes.append(child)

        ############################################################################

    def _generate_class_occurences(self):
        """
        We assume we have set the number of samples, classes, features per node in the hierarchy, so we can now define
        how often each class should occur for each node, i.e., the actual class distribution!
        :return:
        """

        group_nodes = self._get_leaf_nodes()
        for node in group_nodes:
            n_samples = node.n_samples
            class_occurences = [max(2, int(self.n_samples_total / 1000)) for _ in range(node.n_classes)]

            # get parameter for distribution, based on defined imbalance degree
            cls_distribution = self._get_distribution_parameter(self.cls_imbalance)

            # class_occurences = [max(1, int(self.n_samples_total / 1000)) for _ in range(node.n_classes)]
            remaining_samples = n_samples - sum(class_occurences)

            if remaining_samples > 0:
                drawn_class_occurences = self.prob_distribution(cls_distribution,
                                                                node.n_classes,
                                                                size=remaining_samples,
                                                                random_state=self.random_state)
                drawn_class_occurences = list(Counter(drawn_class_occurences).values())
                for i, d_cls_occ in enumerate(drawn_class_occurences):
                    class_occurences[i] += d_cls_occ

            #  Maybe we sampled too much, so we have to adjust the class occurences (randomly)
            if sum(class_occurences) > node.n_samples:
                remove_occs = np.random.choice(len(class_occurences), sum(class_occurences) - node.n_samples)
                for i in remove_occs:
                    class_occurences[i] = class_occurences[i] - 1
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

    @staticmethod
    def _get_distribution_parameter(imbalance_degree):
        if imbalance_degree == "very_balanced":
            return 0
        elif imbalance_degree == "balanced":
            return 1
        elif imbalance_degree == "medium":
            return 2
        elif imbalance_degree == "imbalanced":
            return 3
        elif imbalance_degree == "very_imbalanced":
            return 5

        return imbalance_degree

    @staticmethod
    def _get_group_distribution_parameter(imbalance_degree):
        if imbalance_degree == "very_balanced":
            return 0
        elif imbalance_degree == "balanced":
            return 0.5
        elif imbalance_degree == "medium":
            return 1
        elif imbalance_degree == "imbalanced":
            return 1.5
        elif imbalance_degree == "very_imbalanced":
            return 2

        return imbalance_degree


if __name__ == '__main__':
    c = 30
    f = 50
    #gi = "very_imbalanced"
    ci = 1
    gs = 0
    n = 1000
    co = 1.5

    for gi in range(10):
        print(f"Using config: c={c}, co={co}, gi={gi}, ci={ci}, gh={gs}")
        generator = ImbalanceGenerator(n_features=50,
                                       n=n,
                                       # n_levels=4,
                                       c=c,
                                       features_remove_percent=0,
                                       hardcoded=False,
                                       group_imbalance=gi,
                                       class_overlap=co,
                                       cls_imbalance=ci,
                                       root=EngineTaxonomy().create_taxonomy(),
                                       gs=gs,
                                       n_group_features=10)
        df = generator.generate_data_with_product_hierarchy()
        print(ci)
        print(generator.gini(df["group"]))
