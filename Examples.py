import random
import numpy as np

from anytree import RenderTree

from Hierarchy import HardCodedHierarchy
from SPH_CPI import RandomForestClassMethod, KMeansClassification, \
    BirchClassification, SPHandCPI
from DataGenerator import ImbalanceGenerator
from Utility import train_test_splitting, get_train_test_X_y



#### Very Important! #####
# Data Generation is based on randomness, so we should set random seeds such that the results are reproducible!
np.random.seed(0)
random.seed(0)

# Parameters for Data generation
n_features = 100
n_samples = 1000
n_classes = 84

# First of all, we instantiate an imbalance generator. We leave the parameters as "default" parameters.
generator = ImbalanceGenerator(n_features=n_features, n=n_samples, cls_imbalance="medium",
                               root=HardCodedHierarchy().create_hardcoded_hierarchy(),
                               # root=None, (Could also be none for an automatic hierarchy generation)
                               c=n_classes, features_remove_percent=0)

# Then we generate the data. The result is a dataframe. The actual features of the dataset are contained in the columns
# F0, F1, ..., F{n_features - 1}
# The classes are contained in the column "target" and we also have attributes for the different levels of the hierarchy
df = generator.generate_data_with_product_hierarchy()

# We can also access the generated hierarchy via the root attribute of the generator instance
hierarchy_root = generator.root

# Then we can also print the hierarchy
print(RenderTree(hierarchy_root))

# Now we do the train/test split. We want 75% as training samples
df_train, df_test = train_test_splitting(df, n_train_samples=int(0.75*n_samples))

# Transform to X_train, X_test and y_train, y_test
X_train, X_test, y_train, y_test = get_train_test_X_y(df_train=df_train, df_test=df_test, n_features=n_features)

from Utility import update_data_and_training_data
root = update_data_and_training_data(hierarchy_root, df_train=df_train)
# Now we train our model by running clustering and then train a Random Forest on each cluster.
n_clusters = 20
classif_method = SPHandCPI(root)

# There are at the moment 3 different clustering methods that can be used in the same way.
# However, yet this is only possible because they have the same parameter "k".
# >> rf_classif_method = BirchClassification(n_clusters=n_clusters)
# >> rf_classif_method = GMMClassification(n_clusters=n_clusters)
classif_method.fit(X_train, y_train)

# Predict test samples. We use df_test and not y_test, because for SPH we also need the info of the hierarchy.
# However, we want a unique interface. This functions returns dictionary of top-e accuracy (A@e).
classif_method.predict_test_samples(df_test=df_test)
top_e_accuracy = classif_method.get_accuracy_per_e_df()
print("-------------------------------------")
print("Top-K Accuracy:")
print(top_e_accuracy)
print("-------------------------------------")

# We can also access further statistics.
# Holds information about samples, classes features, missing features etc. per cluster
stats_df = classif_method.get_stats_df()
print("-------------------------------------")
print("Statistics:")
print(stats_df)
print("-------------------------------------")

# We can even get the prediction for each separate sample.
# Here, the column "correct_position" defines if the class is predicted correctly at first, second, ... k-th try.
# A "0" means that the class is not predicted within the default range (10).
predictions_df = classif_method.get_predictions_df()
print("-------------------------------------")
print("Predictions:")
print(predictions_df)
print("-------------------------------------")

rf = RandomForestClassMethod()
rf.fit(X_train, y_train)
print("--------------------------------------------")
print("Baseline:")
rf.predict_test_samples(df_test)
print(rf.get_accuracy_per_e_df())
print("--------------------------------------------")

# For further usage of running multiple methods, you can also have a look at the Experiment.py file.