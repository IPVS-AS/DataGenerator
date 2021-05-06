# 1.) Generate dataset with n_samples, n_features, n_class, mean_imbalance

# vary the dataset size for a 1:100 imbalanced dataset
import logging
import random
from collections import Counter
from typing import List

from anytree import NodeMixin, RenderTree
from anytree.exporter import DotExporter
from boruta import BorutaPy
from sklearn.datasets import make_classification, make_blobs
from sklearn.datasets import make_multilabel_classification
from matplotlib import pyplot
from numpy import where
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score

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
    class_separation_dic = {"very_low": 0.1, "low": 0.5, "medium": 1, "high": 2, "very high": 5}
    imbalance_degrees = ["low", "medium", "high", "equal"]

    def __init__(self, n_samples=1000, n_features=2, clusters_per_class=1, noise=0.0,
                 class_sep="very_low", imbalance_degree="medium", n_classes=2, weights=None, levels=4, n_groups=10):
        self.root = None
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.class_sep = class_sep
        self.n_features = n_features
        self.clusters_per_class = clusters_per_class

        self.noise = noise
        self.n_samples = n_samples
        self.imbalance_degree = imbalance_degree
        self.weights = weights

        if self.class_sep not in ImbalanceGenerator.class_separation_dic:
            self.logger.warning(
                f"class separation not knwon, choose one in {ImbalanceGenerator.class_separation_dic.keys()}")
            self.logger.warning("Setting class separation to medium")
            self.class_sep = "medium"

        if self.imbalance_degree not in ImbalanceGenerator.imbalance_degrees:
            self.logger.warning(f"Imbalance degree not knwon, choose one in {ImbalanceGenerator.imbalance_degrees}")
            self.logger.warning("Setting imbalance degree to medium")
            self.imbalance_degree = "medium"

        self.class_sep_percent = ImbalanceGenerator.class_separation_dic[self.class_sep]
        self.n_classes = n_classes

        self.levels = levels
        self.n_groups = n_groups

    def make_imb_classification(self, weights=None):
        if weights == None:
            weights = self._generate_weigths()
        # weights = self._get_class_weights()
        X, y = make_classification(n_samples=self.n_samples, n_features=self.n_features, n_redundant=0,
                                   n_classes=self.n_classes,
                                   n_clusters_per_class=self.clusters_per_class, flip_y=self.noise,
                                   class_sep=self.class_sep_percent, weights=weights,
                                   random_state=0)
        return X, y

    def _generate_weigths(self):
        # assign weight for each class to its index
        weights = [i for i in range(self.n_classes)]
        return weights

    def _eq_div(self, N, i):
        """
        Divide N into i buckets while preserving the remainder to the buckerts as well.
        :return: list of length i
        """
        return [] if i <= 0 else [N // i + 1] * (N % i) + [N // i] * (i - N % i)

    def generate_product_hierarchy_from_specification(self, root=HardCodedHierarchy().create_hardcoded_hierarchy(),
                                                      n_features=100,
                                                      n_samples_total=1050, n_classes=84, features_remove_percent=0.2,
                                                      imbalance_degree="normal"):
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

        groups = self.generate_product_groups_from_hierarchy(group_nodes_to_create, n_classes,
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

    def generate_data_with_product_hierarchy(self, n_features=100, n_samples_total=1050, n_levels=4, total_n_classes=84,
                                             features_remove_percent=0.2, imbalance_degree="normal",
                                             root=HardCodedHierarchy().create_hardcoded_hierarchy()):
        if root:
            # we have a hierarchy given, so we use this hierarchy
            return self.generate_product_hierarchy_from_specification(root=root, n_features=n_features,
                                                                      n_samples_total=n_samples_total,
                                                                      features_remove_percent=features_remove_percent,
                                                                      n_classes=total_n_classes,
                                                                      imbalance_degree=imbalance_degree)
        else:
            return self.generate_default_product_hierarchy(n_features=n_features, n_samples_total=n_samples_total,
                                                           total_n_classes=total_n_classes,
                                                           imbalance_degree=imbalance_degree,
                                                           features_remove_percent=features_remove_percent,
                                                           n_levels=n_levels)

    def generate_default_product_hierarchy(self, n_features=100, n_samples_total=1050, n_levels=4, total_n_classes=84,
                                           features_remove_percent=0.2, imbalance_degree="normal"):
        """
        Generate specification on its own with "default" settings-
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
        # todo: have to check this in the future
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

        groups = self.generate_product_groups_default(group_nodes_to_create, total_n_classes)

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

    def generate_product_groups_default(self, group_nodes: List[Node], total_n_classes, use_blobs=True):
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

            # hard-coded weights for classes
            # todo: range von samples pro Klasse ist [2, 13].
            #  Wenn 2 ist wird mit den aktuellen weights nicht alle Klassen generiert!
            # todo: Spezialfälle beachten!

            n_samples_per_class_avg = n_samples / n_classes
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
            # create dataset for each group
            n_samples_per_class = [int(w * n_samples) if int(w * n_samples) > 0 else 1 for w in weights]

            # add remaining, i.e., if not all samples are used due to rounding errors
            if sum(n_samples_per_class) < n_samples:
                remaining = n_samples - sum(n_samples_per_class)
                for i in range(remaining):
                    n_samples_per_class[remaining % len(n_samples_per_class)] += 1

            # todo: Functionality to assign values for the parameters: n_inf, n_red, n_clusters_per_class
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

            # third possibility would be to use make_gaussian_qauntiles
            # moons and circles are also possible

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

    def gini(self, x):
        my_index = cm.Index()
        counter = Counter(x)
        return my_index.gini(counter.values())

    def generate_product_groups_from_hierarchy(self, group_nodes: List[Node], total_n_classes, use_blobs=True,
                                               imbalance_degree="normal"):

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

                elif imbalance_degree == 'high':
                    # get max occurence and the index for it
                    max_occurence = max(occurences)
                    max_index = occurences.index(max_occurence)

                    # keep track if 5% are removed
                    samples_removed = 0
                    # this will be our new modified occurence list.
                    # We need this because we do not (!) want to sort the list!
                    # Otherwise this would change the occurence for a specific class
                    new_occurences = occurences.copy()

                    for i, occ in enumerate(occurences):
                        # check if we have at least two samples and this is not the max_occurence.
                        if occ > 2 and occ < max_occurence:
                            # This can easily be changed to remove exactly 5%
                            new_occurences[max_index] += occ - 2
                            new_occurences[i] = 2
                            samples_removed += occ - 2

                        # do we have removed 5% of samples?
                        if samples_removed >= 0.05 * n_samples:
                            break
                    occurences = new_occurences

                elif imbalance_degree == 'low':

                    # here we want to make the classes more imbalanced
                    # idea: move from majority one sample to each minority class
                    max_occurence = max(occurences)
                    max_index = occurences.index(max_occurence)
                    new_occurences = occurences.copy()

                    if len(new_occurences) < max_occurence:
                        new_occurences = [i + 1 if i != max_occurence else max_occurence for i in new_occurences]
                        new_occurences[max_index] = max_occurence - len(new_occurences) + 1
                    else:
                        new_occurences[0:max_index] = [i + 1 for i in new_occurences[0:max_index]]
                        new_occurences[max_index + 1:max_occurence - 1] = [i + 1 for i in new_occurences[
                                                                                          max_index + 1:max_occurence - 1]]
                        new_occurences[max_index] = 2
                    occurences = new_occurences

                # Calculate the weights (in range [0,1]) from the occurrences.
                # The weights are needed for the sklearn function 'make_classification'
                weights = [occ / sum(occurences) for occ in occurences]

            # create dataset for each group
            n_samples_per_class = group.class_occurences

            # add remaining, i.e., if not all samples are used due to rounding errors
            if sum(n_samples_per_class) < n_samples:
                remaining = n_samples - sum(n_samples_per_class)
                for i in range(remaining):
                    n_samples_per_class[remaining % len(n_samples_per_class)] += 1

            cluster_stds = [2 for _ in range(n_classes)]

            n_informative = n_features - 1

            if use_blobs and False:
                # we do not use blobs atm, maybe in the future?
                X, y = make_blobs(n_samples=n_samples_per_class, n_features=n_features,
                                  cluster_std=cluster_stds
                                  )
            else:
                # The questions is, if we need this function if we have e.g., less than 15 samples. Maybe for this, we
                # can create the patterns manually??
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

            # third possibility would be to use make_gaussian_qauntiles

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

    def _get_class_weights(self):
        """Todo: Easy way to assign weigths by an average input of class imbalance?"""
        classes = range(self.n_classes)
        if self.imbalance_degree == "equal":
            # all classes with same weight
            return [1 for n in classes]
        if self.imbalance_degree == "low":
            # we make each majority class occur twice as often as minority class
            weights = [1 for n in classes]
            half_class_n = int(self.n_classes / 2)
            weights = weights[0:half_class_n] + weights[half_class_n + 1:] * 2


if __name__ == '__main__':
    def gini(x):
        my_index = cm.Index()
        class_frequencies = np.array(list(Counter(x).values()))
        return my_index.gini(class_frequencies)

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

    """
    n_samples_for_class = []
    for n in range(50):
        n_samples_for_class.append(10)
    for n in range(10):
        n_samples_for_class.append(20)
    for n in range(10):
        n_samples_for_class.append(30)
    for n in range(5):
        n_samples_for_class.append(40)
    for n in range(3):
        n_samples_for_class.append(50)
    for n in range(4):
        n_samples_for_class.append(60)
    for n in range(1):
        n_samples_for_class.append(70)
    for n in range(1):
        n_samples_for_class.append(100)

    class_counter = Counter(n_samples_for_class)
    classes = class_counter.keys()
    numbers = class_counter.values()
    plt.bar(height=numbers, x=classes)
    plt.title("Class Distribution")
    plt.xlabel("#samples")
    plt.ylabel("#classes")
    plt.show()
    """
"""
    fig = pyplot.figure()
    # fig, axes = fig.add_subplot(n_groups, 1, projection="3D")
    for i in range(n_groups):
        if i == n_groups - 1:
            ax = fig.add_subplot(n_groups, 1, 1, projection="3d")
            group_df = df[df["group"] <= i]
            # ax = axes[i]

            if i == 0:
                ax.set_title(f'Classes per group')
            # scatter plot of examples by class label
            X = df[["F0", "F1", "F2"]].to_numpy()
            y = group_df["target"].to_numpy()
            group = group_df["group"].to_numpy()
            counter = Counter(y)
            for label, _ in counter.items():
                row_ix = where(y == label)[0]
                ax.scatter(X[row_ix, 0], X[row_ix, 1], X[row_ix, 2], label=str(label))
            if i == n_groups - 1:
                ax.legend()
    # fig.savefig("C:\\Users\\tschecds\\Workspace\\ClassPartitioning\\DataGenerator\\Examples\\class_evolved.png")
    pyplot.show()

    fig = pyplot.figure()
    for i in range(n_groups):
        if i == n_groups - 1:

            ax = fig.add_subplot(n_groups, 1, 1, projection="3d")
            group_df = df[df["group"] <= i]
            # ax = axes[i]
            if i == 0:
                ax.set_title(f'Groups')
            X = df[["F0", "F1", "F2"]].to_numpy()
            y = group_df["target"].to_numpy()
            group = group_df["group"].to_numpy()
            counter = Counter(group)

            for label, _ in counter.items():
                row_ix = where(group == label)[0]
                ax.scatter(X[row_ix, 0], X[row_ix, 1], X[row_ix, 2], label=str(label))
            if i == n_groups - 1:
                ax.legend()
    # fig.savefig("C:\\Users\\tschecds\\Workspace\\ClassPartitioning\\DataGenerator\\Examples\\group_evolved.png")
    pyplot.show()
"""
"""
    fig = pyplot.figure(2)
    ax = fig.add_subplot(2, 1, 1)
    # pyplot.subplot(4, 2, 2*i+1)
    ax.set_title(f'actual classes')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # scatter plot of examples by class label
    counter = Counter(y)

    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        ax.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    ax.legend()

    ax = fig.add_subplot(2, 1, 2)
    # pyplot.subplot(4, 2, 2*i+1)
    ax.set_title(f'Groups')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # scatter plot of examples by class label
    counter = Counter(group)

    for label, _ in counter.items():
        row_ix = where(group == label)[0]
        ax.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    ax.legend()
    pyplot.show()
"""
"""
    pyplot.show()
    fig.savefig(
        "C:\\Users\\tschecds\\Workspace\\ClassPartitioning\\DataGenerator\\Examples\\class_groups_subconcepts.png")
            """
