# Code for the smart shoe paper
This repository contains:
1) a program processing raw data, running the random forest training-evaluation treatment and storing the results of the smart shoe project.
2) a notebook for generating figure panels included in the manuscript.



## Content of the repository
* **automatedAnalysis.py** - This program contains a library of functions reading the raw data, extracting features, performing the machine learning treatment (training, evaluation), and storing the results. It automatically manages the feature reduction stepping process and can evaluate multiple sensor configurations, up to the maximum number of threads available.
* **FiguresRerun.ipynb** - This Jupyter notebook contains functions and pieces of code for generating the figure panels showed in the manuscript. The output of **automatedAnalysis.py** are used as inputs. The generation of fig.9B panels used a different set of inputs, also provided in the data repository.

## Prerequisites
Python 3 (>= 3.6)

Jupyter notebook

Packages: Numpy, Pandas, Scipy, Sklearn, pickle, import heapq, time, seaborn, multiprocessing, psutil, sys, os, glob

## Getting Start
* Clone the repository.
* Clone the raw foot pressure data ("FootPressureRawData") from the data repository.
* Prepare two directories, one for storing the results of the "window length" analysis, one for storing the results of the "sensor configuration and feature number reduction" analysis.
* Run the code in the terminal from the directory of **automatedAnalysis.py** 

```bash
python3 automatedAnalysis.py /DIR/foot_pressure_raw_data/ /DIR/result_of_different_configs/ /DIR/result_of_different_window_size/ 
```
* If needed, Run **FiguresRerun.ipynb** in jupyter notebook to generate figure panels.
* Single forest results are not stored by **automatedAnalysis.py** in main analysis. For the sake of single examples shown in Figure9B, single forest performance detailed statistics results have been generated separetely. These results are available in the data repository ("one_config").


