# Code for the smart shoe paper

This repository contains:
1) a program processing raw data, running the random forest training-evaluation treatment and storing the results of the smart shoe project.
2) a notebook for generating figure panels included in the manuscript.


## Content of the repository
* **automatedAnalysis.py** - This program contains a library of functions reading the raw data, extracting features, performing the machine learning treatment (training, evaluation), and storing the results. It automatically manages the feature reduction stepping process and can evaluate multiple sensor configurations, up to the maximum number of threads available. It also automatically manages evaluating performance of multiple window size from 1 second to 60 seconds with full sensor configuration full available features.
* **FigurePanels.ipynb** - This Jupyter notebook contains functions and pieces of code for generating the figure panels showed in the manuscript. The output of **automatedAnalysis.py** are used as inputs. The generation of fig.9B panels used a different set of inputs, also provided in the data repository.

## Prerequisites
Python 3 (>= 3.6)

Jupyter notebook

Packages: Numpy, Pandas, Scipy, Sklearn, pickle, import heapq, time, seaborn, multiprocessing, psutil, sys, os, glob

## Getting Start
* Clone the repository in /DIR
* Clone the raw foot pressure data ("FootPressureRawData") from the data repository in /DIR
* Prepare two folders:
1) /DIR/window_length_results/ for storing the results of the "window length" analysis
2) /DIR/config_feat_results/ for storing the results of the "sensor configuration and feature number reduction" analysis.
* Run the code in the terminal from the directory of **automatedAnalysis.py** with following command to get results of the "window length" analysis.(Example for Linux users.)
```bash
python3 automatedAnalysis.py /DIR/FootPressureRawData/ /DIR/window_length_results/ 
```
* Run the code in the terminal from the directory of **automatedAnalysis.py** with following command to get results of the "number and location of sensors” and “number of features” analyses.
Analysis run for a 20-second window length (Linux users):
```bash
python3 automatedAnalysis.py /DIR/FootPressureRawData/ /DIR/config_feat_results/ 2000
```
* You can run **FigurePanels.ipynb** in jupyter notebook to generate figure panels. Single forest results are not stored by **automatedAnalysis.py** in main analysis. For Figure 13, detailed single forest performance statistics have been generated separetely. These results are available in the data repository ("one_config").
* Note that automatedAnalysis.py can be run for different window lengths. Please replace “2000” in the above example by any length expressed in millisecond (ex: 20 seconds &#8594; 2000, 1 second &#8594; 100, etc.).
* Similarly, the automatedAnalysis.py  can be run for 3 different sets of sensor configurations:
    * 25: the 25 pre-selected sensor configurations presented in the text and the figures of the core of the manuscript.
    * 67: the sensor configurations presented in the supplementary results. Note: for the smallest window lengths, the computation time might take several days depending on the specifications of your machine.
    * 127: all possible sensor configurations. Note #1: for the smallest window lengths, the computation time might take several days depending on the specifications of your machine. Note #2:  “itertools” package needed.
    
    See the two following examples for Linux users.

    Analysis run on the 67 sensor configurations included in the supplementary material section, using a 20-second window length:

```bash
python3 automatedAnalysis.py /DIR/FootPressureRawData/ /DIR/config_feat_results/ 2000 67
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Analysis run on the 25 sensor configurations included in the supplementary material section, using a 15-second window length: 

```bash
python3 automatedAnalysis.py /DIR/FootPressureRawData/ /DIR/config_feat_results/ 1500 25
```

<a href="https://peerj.com/articles/10170/">Article<a> | <a href="https://zenodo.org/records/4050390#.YUq7A_kzYuU">Data and Code<a>
