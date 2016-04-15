# Auto-encoder for anomaly detection

This code does the following:

1. Reads the csv HiggsML dataset and a second distorted dataset or dataset containing anomalies
2. Converts the datasets into numpy arrays
3. Trains an auto-encoder on a subset of the undistorted dataset (labelled background)
4. Tests the auto-encoder on a different subset of the undistorted data (labelled background)
5. Tests the auto-encoder on undistorted data labelled as signal
6. Tests the auto-encoder on the distorted/anomalous data
7. Plots the reconstruction error for each case above

`nn.py`is the main program. `JamesTools.py`contains utility functions for calculating the reconstruction error, for reading in the csv input files, and building the numpy arrays

## Required components
To run you need to install
* SciKitLearn: http://scikit-learn.org/stable/ 
* Neural-network for SciKitLearn (SKNN): http://scikit-neuralnetwork.readthedocs.org/en/latest/index.html
* Numpy, Scipy, Matplotlib (these will probably be installed by the above packages)

The input data csv file can be obtained from http://opendata.cern.ch/record/328

## Running
The code can be run by typing `python nn.py`. The main input file and distorted input file names should be set at the top of the script (default from the opendata page is `atlas-higgs-challenge-2014-v2.csv`).

The first time you run the variable `runTraining`must be set to `True`. Subsequent runs that do not involve modification of the training conditions can be run with this variable set to `False`, which will be considerably faster.

## Variations
Aside of modifying the anomalies/distortions, the following modifications may be instructive:
* switching variables on and off by removing/adding them to the `allowedFeatures`list
* changing the number of training/test events (see below)
* changing the number of layers, the number of neurons per layer, and the learning rate and number of training cycles for the neural network

## Controlling the sample sizes
The total amount of available data in the training and anomalous samples is controlled in the `extractVariables`method calls. How much data is then passed to the neural network is then governed by `buildArrays` calls. The first numerical argument is the number of events to skip, and the second the number of process. One must ensure that the sum of these two numbers does not exceed the number of events read in via `extractVariables`.  



