## Synthetic Data Generator with addititonal Domain Knowledge

In this repo, there are actually three parts implemented: (1) The data generator (see `DataGenerator.py`), (2) the definition of 
a hierarchy (see `Hierarchy.py`) and (3) an approach for partitioning data and running a machine learning algorithm on each 
partition separately (see `Hirsch_et_al_Approach.py`).

### The data generator

The data generator is responsible for generating the data according to a real-world use case.
Here, we focus on generating data that is specified by a _hierarchy_. An example of such a hierarchy can be found via the
HarcdodedHierarchy in ``Hierarchy.py``.
The data generator uses this pre-defined specification to actually generate the data.


###  The Hierarchy Definition

The definition of a pre-defined hardcoded Hierarchy is done in the ``Hierarchy.py``.
This is the basis that should be used and according to this, it should be possible to also change certain parameters 
in each node and also to add/remove nodes or even levels.


### Partitioning Data and Machine Learning 

The file `Hirsch_et_al_Approach.py` contains the approach from Hirsch et al. 
The details of this approach are not important, however,
there are certain parameters that are required for this approach and that should be able to specify.
The parameters can be found in the file, they are there specified via a command line interface.
There is also an example how the measurements are performed at the moment depending on the specified parameters.


### Examples of data and results of current measurements

In this project, there are also some files that show the (I) result of the data generation, (II) the resutls
of the machine learning algorithms, (III) some statistics that I need to take track off, and (IV) some basic accuracy visualization .
In the following, the different kind of files are shortly covered.

#### (I) Dataset files of the generated data
The data is stored in the folder `data_split`. 
Here, we distinguish between train and test data, yet they have the same format.
The data is named with the following pattern: `train_hardcoded_{data_id}.csv`.
The {data_id} refers to the id of the dataset. 
Typically, we execute multiple runs of data generation and hence have different ids for the reuslts.
The `class_dist_{data_id}.png` shows a basic class distribution of the data. This shows the distribution of the whole dataset, not for train/test. 
This is of course something that could also be done.

#### (II) Measurement Results of running the machine learning algorithms
The results of the machine learning experiments are stored in `result_split`.
Here, we have three different kinds of files:
A csv file for all  runs/dataset that contains all results for each run/dataset.
The file is named as `gini_accuracy_all_runs.csv`.
Moreover, there are multiple `Baseline_{dataset_id}.csv` files that contain results of runnin another machine learning algorithm (the baseline).
However, they are generated for each run separately.

#### (III) Statistics
There are also `gini_stats_{dataset_id}` files that store some statistics for each dataset separately. These 
are important statistics that I need to take track of and which help me to decide which dataset/run is more suited.

##### (IV) Basic visualization
There are also files, that show separately for each dataset/parameter combination the accuracy results.
So for example `accuracy_gini25.0_q70.0_dataset2.png` shows the accuracy for dataset with the dataset_id 2 and with the parameter values
of gini_threshold=25 and q=70 (actually you will find sometimes p instead of q in the code. My fault!)
These are examples how a visualization of a result could be possible.

### Usage Example

In each of the three mentioned python files, there should be a main method, where some examples are conducted. 
In general, the usage of the data generation is the following:

````python
# imports. You may not need the 'DataGenerator.*'
from DataGenerator.DataGenerator import ImbalanceGenerator
from DataGenerator.Hierarchy import HardCodedHierarchy

# any tree is important for the structure of the hierarchy
# We need the RenderTree here to print the tree on the console
from anytree import RenderTree

# create an ImbalanceGenerator object 
generator = ImbalanceGenerator()

# we use the hardcoded hierarchy
root = HardCodedHierarchy().create_hardcoded_hierarchy()

# generate the dataset
df = generator.generate_data_with_product_hierarchy(root=root)

# show the resulting dataframe
print(df.head())

# We can also retrieve the hierarchy that also contains the generated data now
hiearchy_root = generator.root

# We can also basically print the hierarchy on the console
print(RenderTree(hiearchy_root))
````



### Installation

Make sure to install python >= 3.6 (newest python 3 version should be fine).
On Windows, you may want to try Anaconda, which also helps in managing your dependencies and virutal environments.
To download the libaries that are required for the project, you can simply install them with ``pip install -r requirements.txt``
There should be a similar command that installs the requirements on conda, e.g., ``conda install --file requirements.txt``
(taken from [here](https://stackoverflow.com/questions/51042589/conda-version-pip-install-r-requirements-txt-target-lib/51043636)).
Yet, I did not tried that so I cannot guarantee that it will work.

If you want to test that your setup was successfull, you can simply run one of the python files.
I recommend the ``DataGenerator.py``.
The ``Hirsch_et_al_Approach.py`` will take quiet some time ;)

If you run the ``DataGenerator.py`` you should be able to see an example of the hierarchy on the console like it is 
printed from anytree.

For working with this project, you can also think about making it to a library. For this you simply need to add an setup file.
Subsequently, you could create your own project and only use the data generator as a 'library'.
However, you can also build on this codebase further.
