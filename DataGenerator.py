import logging
import random
from collections import Counter
from itertools import zip_longest

import anytree
import concentrationMetrics as cm
import numpy as np
import pandas as pd
from scipy.stats import boltzmann, zipfian, poisson
from sklearn.datasets import make_classification

from Taxonomy import Node, EngineTaxonomy


def _check_groups_instances_classes(n_groups, n_instances_per_group, n_classes_per_group):
    if not (n_groups or n_instances_per_group or n_classes_per_group):
        logging.info(
            "Neither n_groups nor n_instances_per_group nor n_classes_per_group are given. using default parameters.")
        return True


def assign_class(cls, n_classes):
    while cls > n_classes - 1:
        cls = cls - n_classes
    return cls


class Generator:
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
                 cf=1,
                 sC="medium",
                 sG="medium",
                 root=EngineTaxonomy().create_taxonomy(),
                 distribution=zipfian.rvs,
                 class_overlap=1.5,
                 hardcoded=False,
                 random_state=1234):
        """
        :param n_features: number of features to use for the overall generated dataset
        :param n: number of instances that should be generated in the whole dataset
        :param n_levels: number of levels of the hierarchy. Does not need to be specified if a hierarchy is already given!
        :param c: number of classes for the whole dataset
        :param features_remove_percent: number of features to remove/ actually this means to have this number of percent
        as missing features in the whole dataset. Currently, this will be +5/6 percent.
        :param sC: The degree of imbalance. Should be either 'medium', 'low' or 'high'. Here, medium means
        to actually use the same (hardcoded) hierarchy that is passed via the root parameter.
        'low' means to have a more imbalanced dataset and 'high' means to have an even more imbalanced dataset.
        :param root: Root node of a hierarchy. This should be a root node that represent an anytree and stands for the hierarchy.
        :param distribution: Distribution to use. In the moment, either boltzman.rvs or zipfian.rvs are tested from the scipy.stats module!
        :param random_state:
        """
        self.cls_imbalance = sC
        self.group_imbalance = sG
        self.hardcoded = hardcoded
        self.root = root
        self.n_features = n_features
        if root:
            self.n_levels = root.height + 1
        else:
            self.n_levels = n_levels
        self.prob_distribution = distribution
        self.group_separation = gs
        self.n_group_features = cf

        self.n_instances_total = n
        self.total_n_classes = c
        self.random_state = random_state
        self.class_overlap = class_overlap

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

    def generate_data_from_taxonomy(self):
        """
        Main method of Data generation.
        Here, the data is generated according to various parameters.
        We mainly distinguish if we have an hierarchy given. This should be given with the root parameter that contains
        the root of an anytree. This root node can be used as representative for the whole tree.
        :return: A dataframe that contains the data and the hierarchy.
        The data is encoded via the feature columns F_0, ..., F_n_features.
        The hierarchy is implicitly given through the specific attributes that represent the hierarchy.
        """

        if isinstance(self.cls_imbalance, str) and self.cls_imbalance not in Generator.imbalance_degrees:
            self.logger.error(f"cls_imbalance_degree should be one of {Generator.imbalance_degrees} but got"
                              f" {self.cls_imbalance}")
            self.logger.warning(f"Setting cls_imbalance_degree to default 'medium'")
            self.cls_imbalance = "medium"
        if isinstance(self.group_imbalance, str) and self.group_imbalance not in Generator.imbalance_degrees:
            self.logger.error(f"group_imbalance should be one of {Generator.imbalance_degrees} but got"
                              f" {self.group_imbalance}")
            self.logger.warning(f"Setting group_imbalance to default 'medium'")
            self.group_imbalance = "medium"

        features = list(range(self.n_features))

        if not self.root:
            self.root = Node(node_id="0", n_instances=self.n_instances_total, feature_set=features,
                             n_classes=self.total_n_classes,
                             classes=(0, self.total_n_classes))
            # generate default hierarchy (use simple automatic approach to specify a taxonomy)
            self._generate_default_taxonomy_spec()
        else:
            # Init attribute values for root node
            self.root.n_instances = self.n_instances_total
            self.root.n_classes = self.total_n_classes
            self.root.feature_set = features
            self.root.classes = (0, self.total_n_classes)

            # Algorithm 1 from our Paper
            self._assign_instances_classes_features()
            self._generate_class_occurences()

            # remove features, don't need this for "generated" hierarchy as we define the features there!
            group_nodes_to_create = self._remove_features_from_spec()

        # check if all features are in the whole data, i.e., we have n_features
        self._check_features()

        # actual generation of the data
        # adjust class distribution is inside data generation
        # Algorithm 2 from our Paper
        groups = self._generate_groups_from_taxonomy_spec()

        return self._create_dataframe(groups)

    def _generate_default_taxonomy_spec(self):
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

    def _generate_groups_from_taxonomy_spec(self):
        """
        Generates the product groups. That is, here is the actual data generated.
        For each group, according to the number of classes, instances and features the data is generated.
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
        remaining_instances = []

        # bottom up approach
        for i, group in enumerate(group_nodes):
            feature_set = group.feature_set
            n_features = len(feature_set)
            n_instances = group.n_instances

            # add instances that are missing due to rounding errors
            if len(remaining_instances) > i:
                n_instances += remaining_instances[i]

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

            # The questions is, if we need this function if we have e.g., less than 15 instances. Maybe for this, we
            # can create the patterns manually?
            X, y = make_classification(n_samples=n_instances,
                                       n_classes=n_classes,
                                       # > 1 could lead to less classes created, especially for low n_instances or
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
                                       hypercube=True,
                                       )

            # normalize x into [0,1] interval
            X = (X - X.min(0)) / X.ptp(0)

            for i, f in enumerate(feature_set):
                # move each feature by its feature limits
                X[:, i] = X[:, i] + feature_limits[f]

            # we create class in range (0, n_classes), but it should be in range (x, x+n_classes)
            if classes:
                y = y + min(classes)
            else:
                created_classes = len(np.unique(y))
                y = y + current_class_num
                current_class_num += created_classes
                y = [assign_class(y_, self.total_n_classes) for y_ in y]

            # we want to assign the data in the hierarchy such that the missing features get already none values
            X_with_NaNs = np.full((X.shape[0], len(total_sample_feature_set)), np.NaN)

            # X is created by just [0, ..., n_features] and now we map this back to the actual feature set
            # columns that are not filled will have the default NaN values
            for i, feature in enumerate(feature_set):
                X_with_NaNs[:, feature] = X[:, i]

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

            group_ids.extend([i for _ in range(X.shape[0])])

        return group_nodes

    def _check_features(self):
        group_nodes_to_create = list(self._get_leaf_nodes())
        current_used_feature_set = set([feature for group in group_nodes_to_create for feature in group.feature_set])

        # features that are currently not used by the groups
        features_not_used = np.setdiff1d(self.root.feature_set, list(current_used_feature_set))

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
            df["group"] = group.node_id

            # assign higher values of the hierarchy to the group (i.e., the levels)
            for l in levels:
                df[f"level-{l}"] = group.hierarchy_level_values[l]
            dfs.append(df)

        return pd.concat(dfs).reset_index().drop("index", axis=1)

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

                # assert sum of childs are equal to parent node n_instances
                childs_n_instances_sum = sum(map(lambda x: x.n_instances, childs))
                assert parent_node.n_instances == childs_n_instances_sum
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

    def _assign_instances_classes_features(self):
        """
        Assigns the number of instances, classes and features to each node! Uses the pre-given information, especially the
        low_high_split to assign instances and classes to each node in the Hierarchy specification!
        Hence, we assume a pre-defined hierarchy specification that defines the structure but has not set the actual
        instances and classes
        :return:
        """

        current_nodes = [self.root]

        ######## Determine number of instances for each node ########
        while current_nodes:
            node = current_nodes.pop()
            n_children = len(node.children)
            if n_children == 0:
                continue

            min_instances_per_node = [1 for i in range(n_children)]
            remaining_instances = node.n_instances - sum(min_instances_per_node)
            group_distribution = self._get_group_distribution_parameter(self.group_imbalance)

            instances_count = self.prob_distribution(a=group_distribution, size=remaining_instances, n=n_children)
            instances_count = list(Counter(instances_count).values())
            instances_count = [x + y for x, y in zip_longest(min_instances_per_node, instances_count, fillvalue=0)]
            instances_per_node = sorted(instances_count)

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

            ############### Assign classes, instances, features to child nodes #############
            for i, child in enumerate(node.children):
                n_classes = n_classes_per_node[i]
                # edge cases:
                if n_classes > node.n_classes:
                    # we have more classes than parent node
                    n_classes = node.n_classes
                elif n_classes < 2:
                    # we have 0 or 1 class
                    n_classes = 2

                n_instances = instances_per_node[i]
                child.feature_set = parent_features
                child.n_instances = n_instances
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
        We assume we have set the number of instances, classes, features per node in the hierarchy, so we can now define
        how often each class should occur for each node, i.e., the actual class distribution!
        :return: group_nodes, i.e., list of leave nodes where he have set the class_occurences
        """

        group_nodes = self._get_leaf_nodes()
        for node in group_nodes:
            n_instances = node.n_instances
            class_occurences = [max(2, int(self.n_instances_total / 1000)) for _ in range(node.n_classes)]

            # get parameter for distribution, based on defined imbalance degree
            # We either use pre-defined values or the direct value
            cls_distribution = self._get_distribution_parameter(self.cls_imbalance)

            # class_occurences = [max(1, int(self.n_instances_total / 1000)) for _ in range(node.n_classes)]
            remaining_instances = n_instances - sum(class_occurences)

            if remaining_instances > 0:
                drawn_class_occurences = self.prob_distribution(cls_distribution,
                                                                node.n_classes,
                                                                size=remaining_instances,
                                                                random_state=self.random_state)
                drawn_class_occurences = list(Counter(drawn_class_occurences).values())
                for i, d_cls_occ in enumerate(drawn_class_occurences):
                    class_occurences[i] += d_cls_occ

            #  Maybe we sampled too much, so we have to adjust the class occurences (randomly)
            if sum(class_occurences) > node.n_instances:
                remove_occs = np.random.choice(len(class_occurences), sum(class_occurences) - node.n_instances)
                for i in remove_occs:
                    class_occurences[i] = class_occurences[i] - 1
            assert sum(class_occurences) == node.n_instances

            # maybe we have rounding errors --> Add random instances until n_instances == class_occurences
            current_instances = sum(class_occurences)

            # instead of while loop, we could also add all instances that are not assigned to a class to only one class
            while current_instances < n_instances:
                max_index = np.argmax(class_occurences)
                class_occurences[max_index] += n_instances - current_instances
                current_instances = sum(class_occurences)

            assert sum(class_occurences) == node.n_instances
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
    taxonomy_root = Node("Root Node")
    child_1 = Node("child_1", parent=taxonomy_root)
    child_2 = Node("child_2", parent=taxonomy_root)
    child_1_1 = Node("child_1_1", parent=child_1)
    child_1_2 = Node("child_1_2", parent=child_1)
    child_2_1 = Node("child_2_1", parent=child_2)
    child_2_2 = Node("child_2_2", parent=child_2)

    # Basic parameters
    n_instances = 1000
    n_features = 20
    n_classes = 30

    generator = Generator(root=taxonomy_root)
    df = generator.generate_data_from_taxonomy()
    print(df)
