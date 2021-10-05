import random
from collections import Counter

import numpy as np
import concentrationMetrics as cm

from anytree import RenderTree

from DataGenerator.DataGenerator import ImbalanceGenerator
from DataGenerator.Hierarchy import FlatHierarchy
from DataGenerator.Utility import train_test_splitting

if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import matplotlib.pyplot as plt


    def gini(x):
        my_index = cm.Index()
        class_frequencies = np.array(list(Counter(x).values()))
        return my_index.gini(class_frequencies)


    generator = ImbalanceGenerator()
    df = generator.generate_data_with_product_hierarchy(root=None, imbalance_degree="normal", n_samples_total=1000)
    X = df[[f"F{i}" for i in range(100)]].to_numpy()
    y_true = df['target'].to_numpy()
    gini_value = gini(y_true)
    # Render the hierarchy
    print(RenderTree(generator.root))
    print(gini_value)
    print(df.isna().sum())

    print('----------------------------------------------------------------------------------------')
    print('------------------------------Example with noisy Data --------------------------------')
    from dcm import dcm

    generator = ImbalanceGenerator()
    # of course this also works with different noise levels and different imbalance degrees
    df = generator.generate_data_with_product_hierarchy(imbalance_degree="normal", noise=0.0, n_samples_total=10000,
                                                        root=FlatHierarchy().create_hierarchy())
    root = generator.root
    X = df[[f"F{i}" for i in range(100)]].to_numpy()
    y_true = df['target'].to_numpy()
    y_nois = df['noisy target']
    noisyCount = np.count_nonzero((y_true == y_nois) == False)
    print(noisyCount)
    gini_value = gini(y_true)
    # Render the hierarchy
    print(RenderTree(root))
    # noisy labels
    # actual labels
    print(df['target'])
    exit()

    print('----------------------------------------------------------------------------------------')
    print(len(df))

    for group in df["group"].unique():
        group_df = df[df["group"] == group]
        print(f"group {group} has {len(group_df)} samples")
    exit()
    print('----------------------------------------------------------------------------------------')
    print('------------------------------Very Low Imbalance Degree --------------------------------')




    np.random.seed(10 * 5)
    random.seed(10 * 10)
    imb_to_one_sample_class = {}
    imb_to_one_sample_count = {imb: 0 for imb in ImbalanceGenerator.imbalance_degrees}

    for imb in ImbalanceGenerator.imbalance_degrees:
        generator = ImbalanceGenerator()
        df = generator.generate_data_with_product_hierarchy(imbalance_degree=imb)
        root = generator.root
        y_true = df['target'].to_numpy()
        gini_value = gini(y_true)
        print(RenderTree(root))
        print(f'Gini index for whole data is: {gini_value}')
        df, _ = train_test_splitting(df, n_train_samples=750)

        average_one_samples = 0
        for group in df['group'].unique():
            group_targets = df[df['group'] == group]['target'].value_counts()
            group_targets_dic = dict(group_targets)
            print(group_targets_dic)
            group_targets_dic = {k: v for k, v in group_targets_dic.items() if v == 1}

            average_one_samples += len(group_targets_dic) / sum(group_targets)
            imb_to_one_sample_count[imb] += len(group_targets_dic) / len(group_targets)

        average_one_samples = average_one_samples / len(df['group'].unique())
        imb_to_one_sample_class[imb] = average_one_samples

    print(imb_to_one_sample_count)
    print(imb_to_one_sample_class)

    exit()
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
    counter = Counter(df["target"].to_numpy())
    counter_one = {t: counter[t] for t in counter.keys() if counter[t] == 1}
    print(f"number of classes that have only one sample: {counter_one}")
    print('----------------------------------------------------------------------------------------')