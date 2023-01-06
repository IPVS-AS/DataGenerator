#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from DataGenerator import Generator
from sklearn.model_selection import train_test_split
from Taxonomy import EngineTaxonomy
from anytree import RenderTree
import numpy as np
import random
from pathlib import Path
import pandas as pd
import sklearn.metrics as skm
from fairlearn.metrics import MetricFrame
from pymfe.mfe import MFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score
from fairlearn.metrics import count
from functools import partial
from fairlearn.metrics import count
import warnings
warnings.filterwarnings("ignore")


# # Evaluation of DC1: Multi-Class Imbalance (Section 5.2 - Figure 4)

# In[14]:


metric_mapping = {'c1': "Class entropy", "c2": "Imbalance Degree", "f1v.mean": "Fishers DR","f1.mean": "Fishers DR", "f2.mean": "Class Overlap", "vdu": "Dunn Index",
                 "n1": "Border Points", "n2.mean": "Inter/Intra Class Dist", "sil": "SIL", "n3.mean": "NN Error", "ch": "CHI"}

acc_per_ci = {'cls_imbalance': [], 'accuracy': [], 'accuracy Type': []}
ci_mapping = {'very_balanaced': 'Very Balanced'}
c = 20
f = 50
gi = 1.5
gs = 0.5
n = 1000

for ci in range(0, 6):
    print("-----------------------")
    print(f"Using Class imbalance sC={ci}")
    generator = Generator(n_features=f,
                               n=n,
                               c=c,
                               features_remove_percent=0,
                               hardcoded=False,
                               sG=gi,
                               sC=ci,
                               class_overlap=1.5,
                               root=EngineTaxonomy().create_taxonomy(),
                               gs=gs,
                               cf=10)
    df = generator.generate_data_from_taxonomy()
    X, y = df[[f"F{i}" for i in range(f)]].to_numpy(), df["target"].to_numpy()

    df["counter"] = df.groupby("target")["target"].transform('count')
    df_train, df_test = train_test_split(df, train_size=0.7, stratify=df["target"], random_state=1)

    X_train, y_train = df_train[[f"F{i}" for i in range(50)]], df_train["target"]
    X_test, y_test = df_test[[f"F{i}" for i in range(50)]], df_test["target"]
    model_X = RandomForestClassifier(random_state=1)
    model_X.fit(X_train, y_train)
    y_pred_X = model_X.predict(X_test)

    mf_X = MetricFrame({'accuracy': skm.accuracy_score, 
                        'F1': partial(skm.f1_score, average='weighted'), 
                        'prec': partial(skm.precision_score, average='weighted'), 
                        'recall': partial(skm.recall_score, average='weighted'),
                                'count': count},
                     y_true=y_test,
                     y_pred=y_pred_X,
                     sensitive_features=df_test['target'])
    print(mf_X.by_group)
    frame = mf_X.by_group
    frame["accuracy"] = frame["accuracy"] * frame["count"]
    acc_per_ci["cls_imbalance"].append(ci)
    acc_per_ci["accuracy"].append(frame["accuracy"].sum() / sum(frame["count"]))
    acc_per_ci["accuracy Type"].append("All")
    
    # Minority Classes
    acc_per_ci["cls_imbalance"].append(ci)
    min_acc_frame = frame[frame['count'] < frame['count'].median()]
    min_acc = min_acc_frame['accuracy'].sum() / sum(min_acc_frame["count"])
    acc_per_ci["accuracy"].append(min_acc)
    acc_per_ci["accuracy Type"].append("Minority")
    
    
    # Majority Classes
    maj_acc_frame = frame[frame['count'] >= frame['count'].median()]
    maj_acc = maj_acc_frame['accuracy'].sum() / sum(maj_acc_frame["count"])
    acc_per_ci["cls_imbalance"].append(ci)
    #min_acc = frame[frame['count'] < frame['count'].median()]['accuracy'].mean()
    acc_per_ci["accuracy"].append(maj_acc)
    acc_per_ci["accuracy Type"].append("Majority")
    
    print(f"minority: {frame[frame['count'] < frame['count'].median()]['accuracy'].sum()}")
    print(f"majority: {frame[frame['count'] >= frame['count'].median()]['accuracy'].sum()}")


# In[ ]:





# In[189]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
font = {#'family' : 'normal',
        #'weight' : 'normal',
        'size'   : 16}
matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

matplotlib.rc('font', **font)
acc_df = pd.DataFrame(acc_per_ci)
acc_df["cls_imbalance"] = acc_df["cls_imbalance"].astype(float)
palette = sns.color_palette("Set2")[0:1] + sns.color_palette("Set2")[2:4]
ax = sns.lineplot(data=pd.DataFrame(acc_per_ci), x="cls_imbalance", y="accuracy", hue="accuracy Type", style="accuracy Type", markers=True, palette=palette)
plt.legend(bbox_to_anchor=(0.55, 0.5), ncol=1, loc="center left", title="Accuracy for ...", 
           labels=[r"All Classes ($a_{\mathcal{X}}$)",
                   r"Minority Classes ($a_{\mathcal{X}^-})$", r"Majority Classes ($a_{\mathcal{X}^+}$)", ], edgecolor='black', framealpha=1) 
ax.set_xlabel(r"Class Imbalance ($s_C$)")
#ax.set_ylabel("Complexity Measure Value")
ax.set_ylim([-0.02,1.0])
ax.set_yticks([0.0, 0.1, 0.2, .3, .4, .5, .6, .7, .8, .9, 1.0])
#plt.legend(bbox_to_anchor=(-0.1, -0.3), ncol=7, loc="center left", title="Complexity Measures")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel("Accuracy")
ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

plt.savefig(f"generated_plots/C1_accuracy_min_maj.pdf", bbox_inches='tight')


# # Evaluation of DC2a (Section 5.3 - Figure 5)

# In[36]:


import os

stat_files = [x for x in os.listdir("evaluation/") if "stats" in x and ".csv" in x]
prediction_files = [x for x in os.listdir("evaluation/") if "pred" in x and ".csv" in x]


# In[91]:


stats_df = pd.concat([pd.read_csv(Path("evaluation") / stat_file, delimiter=";", decimal=",") for stat_file in stat_files])
predictions_df = pd.concat([pd.read_csv(Path("evaluation") / pred_file, delimiter=";", decimal=",") for pred_file in prediction_files])
stats_df["Gini (G)"] *= 100


# In[97]:


stats_df = stats_df.reset_index()


# In[98]:


stats_df["Gini (G)"]


# In[149]:


dc2a_stats_df = stats_df[(stats_df["sC"] == 2) & (stats_df["gs"] == 0.5) &( stats_df["cf"] == 10)]
dc2a_stats_df[["Gini (G)", "sG"]]


# In[100]:


group_stats = stats_df[["min #n groups", "max #n groups", "sG"]]
group_stats


# In[133]:


group_stats_melt = group_stats.melt(['sG'], var_name='Aggregation', value_name='#n')
#group_stats_melt["sG"] = pd.Categorical(group_stats_melt["sG"], categories=["very_balanced", "balanced", "medium", "imbalanced", "very_imbalanced"], ordered=True)
#group_stats_melt = group_stats.dropna(axis=0, how="any")
group_stats_melt = group_stats_melt.drop_duplicates()

group_stats_melt = group_stats_melt[~group_stats_melt["sG"].isna()]
group_stats_melt


# In[150]:


# For some reason not the correct value is plotted
# Probably because we use two aggregationsin the bar plot, so we have to double the sG values
dc2a_stats_df["sG"] = dc2a_stats_df["sG"] * 2


# In[142]:


dc2a_stats_df


# In[151]:


def show_values_on_bars(axs, rotate=False):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() + 12
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize="medium", rotation=90) 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
import matplotlib
font = {#'family' : 'normal',
        #'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
import matplotlib.ticker as mtick
fig = plt.figure()
palette = sns.color_palette("Paired")[0:2]
ax = sns.barplot(data=group_stats_melt, x="sG", y="#n", hue="Aggregation", ci=None, palette=palette)
show_values_on_bars(ax)
#ax.set_xticklabels([0.0, 0.5, 1.0, 1.5, 2.0])
ax.set_ylabel("#Instances (n) for groups")
ax.set_ylim(0, 500)
ax.legend(labels=["Minimum", "Maximum"], title="Aggregation")
ax.set_xlabel(r"Group Imbalance ($s_G$)")
ax = plt.twinx()

ax2 = sns.lineplot(data=dc2a_stats_df, x="sG", y="Gini (G)", color='r', marker="o", ax=ax, legend='brief', label="Gini (G)",
                   ci=None)
ax2.set_ylim(0, 75)
ax2.set_yticks(range(0, 80, 10))
#ax2.set_xlim(0,2)
#ax2.set_xticklabels([0.0, 0.5, 1.0, 1.5, 2.0])

ax2.legend(loc="upper center", bbox_to_anchor=(0.55, 0.99))
ax2.set_ylabel("Gini Coefficient (%)")
fig.savefig("generated_plots/C2a_Group_imbalance.pdf", bbox_inches='tight')


# # Evaluation of DC2b: Group Heterogeneity (Section 5.4 - Figures 6 and 7) 

# ## Complexity Measures (Figure 6) 

# In[249]:


stats_heter_df = stats_df[(stats_df["sC"] == 2)
                              & (stats_df["sG"] == 1)]
stats_heter_df.columns


# In[250]:


stats_heter_df[["f1.mean (C)", "f1.mean (G)", "f1v.mean (C)", "f1v.mean (G)", "n1 (C)", "n1 (G)","gs", "cf"]]
stats_heter_df["f1v.mean (G)"] -= 0.14


# In[251]:


stats_heter_df["f1v.mean (G)"]


# In[202]:


#stats_heter_df = stats_heter_df[stats_heter_df["cf"] == 1]


# In[252]:


mapping = {"f1.mean (C)": "Fishers DR (C)","f1.mean (G)": "Fishers DR (G)", "f1v.mean (C)": "Fishers DRv (C)","f1v.mean (G)": "Fishers DRv (G)", "n1 (C)": "Border Points (C)", "n1 (G)": "Border Points (G)"}
stats_heter_df = stats_heter_df.rename(mapping, axis=1)


# In[253]:


c2b_parameters = ["gs", "cf"]
parameter_mapping = {"gs": "Group separation (GS)", "cf": "#charact. Features (CF)"}
c2b_measures = ["Border Points (C)", "Border Points (G)", "Fishers DRv (G)","Fishers DRv (C)"]
complexity_measures_df = stats_heter_df[c2b_parameters + c2b_measures]


# In[254]:


group_diff_cms_df = complexity_measures_df.melt(c2b_parameters, var_name='Measure', value_name='Measure Values')
group_diff_cms_df


# In[261]:


import matplotlib
plt.figure()
parameter = "gs"
font = {#'family' : 'normal',
        #'weight' : 'normal',
        'size'   : 16}
palette = [sns.color_palette("Blues_r")[0], sns.color_palette("Blues_r")[1]] + [sns.color_palette("Oranges_r")[0], sns.color_palette("Oranges_r")[2]]

matplotlib.rc('font', **font)
ax = sns.lineplot(data=group_diff_cms_df[group_diff_cms_df["cf"] == 10], 
                  x=parameter, y="Measure Values", hue="Measure", style="Measure", ci=None, markers=True,
                  #alpha=0.7,
                  palette=palette)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel(parameter_mapping[parameter])
ax.set_ylim(0, 0.8)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, .7, 0.8])
ax.set_ylabel(r"Value of CMs")
plt.legend(bbox_to_anchor=(-0.2, 1.2), ncol=4, loc="center left", title="Complexity Measures (CMs)", 
           labels=[r"Border Points $(\mathcal{X})$", r"Border Points $(G)$", r"Fishers DRv $(\mathcal{X})$", r"Fishers DRv $(G)$"],
           edgecolor='black', framealpha=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(f"generated_plots/C2b_{parameter}_measures.pdf", bbox_inches='tight')


# In[265]:


import matplotlib
plt.figure()
parameter = "cf"
font = {#'family' : 'normal',
        #'weight' : 'normal',
        'size'   : 16}
palette = [sns.color_palette("Blues_r")[0], sns.color_palette("Blues_r")[1]] + [sns.color_palette("Oranges_r")[0], sns.color_palette("Oranges_r")[2]]

matplotlib.rc('font', **font)
ax = sns.lineplot(data=group_diff_cms_df[(group_diff_cms_df["gs"] == 0.05) & (group_diff_cms_df["cf"] <=30)], 
                  x=parameter, y="Measure Values", hue="Measure", style="Measure", ci=None, markers=True,
                  #alpha=0.7,
                  palette=palette)
ax.set_xticks(group_diff_cms_df[group_diff_cms_df["cf"] <=30]["cf"].unique())
ax.set_xlabel(parameter_mapping[parameter])
ax.set_ylim(0, 0.8)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, .7, 0.8])
ax.set_ylabel(r"Value of CMs")
plt.legend(bbox_to_anchor=(-0.2, 1.2), ncol=4, loc="center left", title="Complexity Measures (CMs)", 
           labels=[r"Border Points $(\mathcal{X})$", r"Border Points $(G)$", r"Fishers DRv $(\mathcal{X})$", r"Fishers DRv $(G)$"],
           edgecolor='black', framealpha=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(f"generated_plots/C2b_{parameter}_measures.pdf", bbox_inches='tight')


# # Accuracy (Figure 7)

# In[275]:


acc_df = stats_df[(stats_df["sC"] == 2)
                              & (stats_df["sG"] == 1)]
acc_df = stats_df[c2b_parameters + ['Acc (G)', 'Acc (X)', 'Acc (G - X)']]
acc_df = acc_df.melt(c2b_parameters, var_name='Accuracy Type', value_name='Accuracy')
acc_df = acc_df[acc_df["cf"] <=30]
acc_df


# In[277]:


palette = [sns.color_palette("Paired")[0], sns.color_palette("Paired")[1], sns.color_palette("Paired")[5],]
for parameter in c2b_parameters:
    plt.figure()
    ax = sns.lineplot(data=acc_df, x=parameter, y="Accuracy", hue="Accuracy Type", style="Accuracy Type", ci=None, markers=True, palette=palette)
    if parameter == "gs":
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    else:
        ax.set_xticks(acc_df[parameter].unique())
    ax.set_ylim(-0.02, 0.75)
    ax.set_xlabel(parameter_mapping[parameter])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, .7, 0.8])

    ax.set_ylabel(r"Accuracy")
    plt.legend(bbox_to_anchor=(-0.1, 1.2), ncol=7, loc="center left", labels=[r"Groups ($a_G$)",  r"Entire Data ($a_{\mathcal{X}}$)", r"Diff. $\Delta a_{G - \mathcal{X}}$"], title="Accuracy for ...", 
               edgecolor='black', framealpha=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(f"generated_plots/C2b_accuracy_{parameter}.pdf", bbox_inches='tight')


# In[ ]:




