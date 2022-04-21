#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from DataGenerator import ImbalanceGenerator
from sklearn.model_selection import train_test_split
from Hierarchy import EngineTaxonomy
from anytree import RenderTree
import numpy as np
import random
from pathlib import Path
import pandas as pd
import sklearn.metrics as skm
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pymfe.mfe import MFE

from fairlearn.metrics import count
from sklearn.linear_model import LogisticRegression
from dcm import dcm

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

f = 100
n = 1000

gi = 'medium'
ci = 'medium'
c = 10
gs = 0.25

metric_mapping = {'c1': "Class entropy", "c2": "Imbalance Degree", "f1v.mean": "Fishers DR", "f2.mean": "Class Overlap", "vdu": "Dunn Index",
                 "n1": "Border Points", "n2.mean": "Inter/Intra Class Dist", "sil": "SIL", "n3.mean": "NN Error", "ch": "CHI"}

result_df = pd.DataFrame(columns=list(metric_mapping.values())+["target", "gs", "#c"])
for c in [10, 30, 50, 70, 100]:#, 100]:
    for gs in [0, 0.25, 0.5, 1.0]:
        for gi in ['balanced', 'medium', 'imbalanced']:
            for ci in ['balanced', 'medium', 'imbalanced']:

                print('---------------------------------')
                print(f'---- #classes: {c}, gs={gs}--------')
                generator = ImbalanceGenerator(n_features=f,
                                           n_samples_total=n,
                                           total_n_classes=c,
                                           features_remove_percent=0,
                                           hardcoded=False,
                                           group_imbalance=gi,
                                           cls_imbalance=ci,
                                           class_overlap=1.5,
                                           root=EngineTaxonomy().create_taxonomy(),
                                           group_separation=gs,
                                           n_group_features=10)
                df = generator.generate_data_with_product_hierarchy()
                X, y = df[[f"F{i}" for i in range(f)]], df["target"]
                groups = df["group"]

                for target_name, target in zip(["G", "C"],[groups, y]):
                    ## Complex Statistics from PyMFE ###
                    mfe = MFE(groups=["all"], features=["vdu", "f1v", "f2", "n1", "n2", "c1", "c2", "n3", "sil", "ch"], summary=["mean"])
                    mfe.fit(X.to_numpy(), target.to_numpy())
                    ft = mfe.extract()
                    for metric, value in zip(ft[0], ft[1]):
                        print(f"{metric_mapping[metric]} ({target_name}): {value}")
                    stats_df[f"Gini ({taget_name})"] = generator.gini(target)

                    stats_df = pd.DataFrame({metric_mapping[metric]: [value] for metric, value in zip(ft[0], ft[1])} )
                    # Basic stats (#instances etc.)

                    stats_df["avg #n groups"] = df.groupby(['group']).size().mean()
                    stats_df["avg #n classes+groups"] = df.groupby(['group', 'target']).size().mean()
                    stats_df["min #n groups"] = df.groupby(['group']).size().min()
                    stats_df["max #n groups"] = df.groupby(['group']).size().max()

                    stats_df["target"] = target_name
                    stats_df["#c"] = c
                    stats_df["#n"] = df.shape[0]
                    stats_df["gs"] = gs
                    stats_df["n"] = n
                    stats_df["gi"] = gi
                    stats_df["ci"] = ci

                    result_df = pd.concat([result_df, stats_df])

                    result_df.to_csv('complexity_metrics.csv', sep=';', decimal=',')


