# Synthetic Data Generation for Imbalanced Multi-class Problems with Heterogeneous Groups

This repository contains the code for the submitted paper "Synthetic Data Generation for Imbalanced Multi-class Problems with Heterogeneous Groups".

The repository contains (i) an installation instruction, (ii) the code for the data generator, 
(iii) the code for the used taxonomy, (iv) the code for the experiments of our evaluation, 
and (v) an example notebook on how to use the generator.

## Installation and Execution

This section describes how to run the full example of this project, which consists mainly of evaluation and plot generation. 

### Prerequisites 

To run the script successfully, Python 3.9 must be installed and the environment variable `python` must point to this version.
This can be achieved by installing Python 3.9 system-wide or using a Python environment manager such as [Anaconda](https://anaconda.org/). 

If Anaconda is installed, the following steps must be performed: 

1. Create a virtual environment with the command `conda create -n data_generator_p39 python=3.9`
2. Activate the environment with the command `conda activate data_generator_p39`

### Execution

This script does not install or overwrite any packages already installed in the Python environment, but creates a package that contains all required Python packages and libraries (e.g. NumPy, Pandas etc.) in the required version.

To run the script, the `RunExample.py` file must be executed. This script needs a parameter `mode`. This parameter determines whether the complete evaluation is executed or only the generated plot. The reason for this is that the complete evaluation can take several hours or even days (depending on the host computer). Please note that it is not necessary to run the evaluation before plot generation is possible. Due to the long runtime, the evaluation results are included in this repository.

To run the full evaluation (evaluation and plots) execute the script as follows: 

```bash
python RunExample.py -m eval
```

If the plot generation is sufficient run the script with: 

```bash
python RunExample.py -m plot
```

The evaluation files are generated in the `evaluation` folder and the plot files in the `generated_plots` folder.

To recompile the LaTeX document, unpack the file `DataGenerator.zip` into the root directory of the project. Then run the plot generation (`python RunExample.py -m plot`). The plots will be automatically copied to the LaTeX part of the project. Since the LaTeX project was mainly created with [overleaf](https://www.overleaf.com/), it works best if you upload it to overleaf and recompile it there. Also, the "Main Document" option in the project settings must be changed to `btw/btw.tex`. 

## Data Generation

The code for the data generator, i.e., in particular the two algorithms from our paper are implemented in the script 
``DataGenerator.py``.
The generator can be used with just two lines of code: Instantiating the generator object and running the method to generate the data.

## Taxonomy

The script `Taxonomy.py` includes the taxonomy that we created using the anytree package.
This taxonomy that is a simplified except regarding vehicle engines.
The taxonomy is hardcoded in the script.
So, you can easily adjust the taxonomy with your own custom taxonomy.
An example can also be found in the Examples folder.

## Experiments

To run the experiments of our evaluation, you can just run the script ``Evaluation.py``.
That means it performs the evaluations that we used to evaluate the multi-class imbalance (DC1),
the imbalance of the groups (DC2a), and the heterogeneity of the class patterns (DC2b).
However, note that this script runs more experiments than we reported in our paper.
The reason is that it executes all parameter combinations and also calculates all complexity measures.
However, by adjusting the first two lines of the main method, the parameters and complexity measures can be easily adjusted.

We further provide the results of our experiments in the directory "evaluation/".
We contain two kinds of results for each generated dataset.
The ``stats`` and the ``predictions`` csv files.

The ``stats`` files contain information about statistics of the datasets as well as the values of the complexity measures.
It also contains the values for the complexity measures calculated on the whole data (e.g., "f1v (C)" for the value of Fishers DRv on the whole data)
and when calculated on average for each group (e.g., "f1v (G)" for the average value of Fishers DRv over the groups).
To access the stats or predictions for one parameter setting of the generator (i.e. for one generated dataset) you can retrieve them by their filename.
As naming convention, you append the parameters with their values to the filename.
So, for example to get the stats for sC=1, sG=0, gs=0.5, cf=10 you have to look into ``stats_sC1_sF0_gs0.5_cf10.csv``.
We use the same filename convention for the predictions.

The ``predictions`` files contain the average accuracy, F1 score, precision, and recall on average for each group.
They also have a column "Model". 
If the "Model" has the value "G" that means we have trained one classifier for each group separately.
Otherwise, we have used one classifier for the whole data.

## Examples

The directory `Examples/` contains an example notebook "Examples.ipynb" on how to apply the data generator and how to vary custom parameters.
It also shows an example on how to use a custom taxonomy for the data generation.
