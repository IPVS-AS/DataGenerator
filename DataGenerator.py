# 1.) Generate dataset with n_samples, n_features, n_class, mean_imbalance

# vary the dataset size for a 1:100 imbalanced dataset
import logging
import random
from collections import Counter
from sklearn.datasets import make_classification, make_blobs
from sklearn.datasets import make_multilabel_classification
from matplotlib import pyplot
from numpy import where
import numpy as np
import pandas as pd


def _check_groups_samples_classes(n_groups,n_samples_per_group, n_classes_per_group):
    if not (n_groups or n_samples_per_group or n_classes_per_group):
        logging.info("Neither n_groups nor n_samples_per_group nor n_classes_er_group are given. using default parameters.")
        return True


class ImbalanceGenerator:
    class_separation_dic = {"very_low": 0.1, "low": 0.5, "medium": 1, "high": 2, "very high": 5}
    imbalance_degrees = ["low", "medium", "high", "equal"]

    def __init__(self, n_samples=1000, n_features=2, clusters_per_class=1, noise=0.0,
                 class_sep="very_low", imbalance_degree="medium", n_classes=2, weights=None, levels=3, n_groups=10):
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
                                   class_sep=self.class_sep_percent, weights=weights)
        return X, y

    def _generate_weigths(self):
        # assign weight for each class to its index
        weights = [i for i in range(self.n_classes)]
        return weights

    def generate_product_groups(self, n_groups=None, n_samples_per_groups=None, n_classes_per_group=None,
                                n_features=None, class_weights_per_group=None, std_per_class=None):
        ######### Start of parameters and setting to default values ###############################
        if not n_groups:
            if n_samples_per_groups:
                n_groups = len(n_samples_per_groups)
            elif n_classes_per_group:
                n_groups = len(n_classes_per_group)
            else:
                n_groups = 3
        n_classes_per_group = [3 for _ in range(n_groups)]
        n_samples_per_groups = [1000 for _ in range(n_groups)]

        if not class_weights_per_group:
            #use this to have somehow imbalanced data
            class_weights_per_group = [[(j + 1) ** 2 / n_classes for j in range(n_classes)] for n_classes in n_classes_per_group]
            # normalize weights
            # We get the sampels per group by multiplying each weight with n_samples for each group
            class_weights_per_group = [ [x / sum(weights) for x in weights] for weights in class_weights_per_group]


        # todo: check lengths are all equal
        #_check_groups_samples_classes(n_groups, n_samples_per_groups, n_classes_per_group)

        if not n_samples_per_groups:
            n_samples_per_groups = [1000 for _ in range(n_groups)]

        if not n_classes_per_group:
            n_classes_per_group = [3 for _ in range(n_groups)]

        if not std_per_class:
            std_per_class = [[1.5 for _ in range(n_classes)] for n_classes in n_classes_per_group]

        if not n_features:
            n_features = 2
        ########################### End of parameters section ##############################################

        # We make the names to S1, S2, ... for the different Sensors
        features_names = [f"S{i + 1}" for i in range(n_features)]

        resulting_groups = []
        target_classes = []
        group_ids = []

        # save limits of each feature --> first all are 0.0
        feature_limits = {i: 0 for i in range(n_features)}

        # bottom up approach
        for i in range(n_groups):
            n_samples = n_samples_per_groups[i]

            weights = class_weights_per_group[i]

            # create dataset for each group
            n_samples_per_class = [int(w * n_samples) for w in weights]

            print(n_samples_per_class)
            # todo: need this if we want multiple clusters per class in a group
            # n_samples_half = [int(n / 2) for n in n_samples_per_class]
            cluster_stds = std_per_class[i]
            X, y = make_blobs(n_samples=n_samples_per_class, n_features=n_features,
                              cluster_std=cluster_stds)

            # want to add subconcepts --> can be done for example for lower number of classes
            # X_2, y_2 = make_blobs(n_samples=n_samples_half, n_features=n_features, cluster_std=[1.5 for _ in range(n_classes)])
            # X = np.concatenate((X, X_2), axis=0)
            # y = np.concatenate((y, y_2), axis=0)

            # normalize x into [0,1] interval
            X = (X - X.min(0)) / X.ptp(0)

            for f in range(n_features):
                X[:, f] = X[:, f] + feature_limits[f]

            # take a random feature along we move the next group
            feature_to_move = np.random.choice(range(n_features))
            feature_limits[feature_to_move] += 1
            print(f"moving feature {feature_to_move}")

            if i > 0:
                print(f"sum: {sum(n_classes_per_group[0:i])}")
                # make clear we have "new" classes
                y = y + sum(n_classes_per_group[0:i])

            resulting_groups.append(X)
            target_classes.extend(y)
            group_ids.extend([i for _ in range(X.shape[0])])

        # merge the product groups into one sample set
        df = pd.DataFrame(np.concatenate(resulting_groups), columns=features_names)
        df["target"] = target_classes
        df["group"] = group_ids
        return df

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
        # if self.imbalance_degree == "medium":
        # For medium, the majority classes


def get_group_from_row(x):
    if x[0] <= 0.5 and x[1] <= 0.5:
        result = 0
    elif x[0] <= 0.5 and x[1] > 0.5:
        result = 1
    elif x[0] > 0.5 and x[1] <= 0.5:
        result = 2
    else:
        result = 3

    if x[0] <= 1:
        return 0
    else:
        return 1
    return result


if __name__ == '__main__':
    n_groups = 3

    df = ImbalanceGenerator().generate_product_groups(n_groups)
    print(df.head())
    print(df.loc[1000])

    X = df[["S1", "S2"]].to_numpy()
    y = df["target"].to_numpy()
    group = df["group"].to_numpy()


    fig, axes = pyplot.subplots(n_groups, 1)
    for i in range(n_groups):
        group_df = df[df["group"] <= i]
        ax = axes[i]
        if i==0:
            ax.set_title(f'Classes per group')
        # scatter plot of examples by class label
        X = group_df[["S1", "S2"]].to_numpy()
        y = group_df["target"].to_numpy()
        group = group_df["group"].to_numpy()
        counter = Counter(y)

        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            ax.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        if i == n_groups - 1:
            ax.legend()
    #fig.savefig("C:\\Users\\tschecds\\Workspace\\ClassPartitioning\\DataGenerator\\Examples\\class_evolved.png")
    pyplot.show()

    fig, axes = pyplot.subplots(n_groups, 1)
    for i in range(n_groups):
        group_df = df[df["group"] <= i]
        ax = axes[i]
        if i==0:
            ax.set_title(f'Groups')
        X = group_df[["S1", "S2"]].to_numpy()
        y = group_df["target"].to_numpy()
        group = group_df["group"].to_numpy()
        counter = Counter(group)

        for label, _ in counter.items():
            row_ix = where(group == label)[0]
            ax.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        if i == n_groups-1:
            ax.legend()
    #fig.savefig("C:\\Users\\tschecds\\Workspace\\ClassPartitioning\\DataGenerator\\Examples\\group_evolved.png")
    pyplot.show()
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
