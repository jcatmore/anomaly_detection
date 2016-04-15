import sys
import numpy as np
from sklearn import svm
from sklearn import preprocessing
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import metrics
from JamesTools import extractVariables,buildArrays

###############################################
# MAIN PROGRAM

normalInputFile = open("atlas-higgs-challenge-2014-v2.csv","r")
distortedInputFile = open("distorted5pc.csv","r")
allowedFeatures = [
                   "PRI_tau_pt", #0
                   "PRI_tau_eta", #1
                   "PRI_tau_phi", #2
                   "PRI_lep_pt", #3
                   "PRI_lep_eta", #4
                   "PRI_lep_phi", #5
                   "PRI_met", #6
                   "PRI_met_phi", #7
                   "PRI_met_sumet", #8
                   "PRI_jet_num", #9
                   "PRI_jet_leading_pt", #10
                   "PRI_jet_leading_eta", #11
                   "PRI_jet_leading_phi", #12
                   "PRI_jet_subleading_pt", #13
                   "PRI_jet_subleading_eta", #14
                   "PRI_jet_subleading_phi", #15
                   "PRI_jet_all_pt", #16
                   "DER_mass_transverse_met_lep", #17
                   "DER_mass_vis", #18
                   "DER_pt_h", #19
                   "DER_deltar_tau_lep", #20
                   "DER_pt_ratio_lep_tau", #21
                   "DER_met_phi_centrality", #22
                   "DER_mass_MMC", #23
                   "DER_deltaeta_jet_jet", #24
                   "DER_mass_jet_jet", #25
                   "DER_prodeta_jet_jet", #26
                   "DER_pt_tot,DER_sum_pt", #27
                   "DER_lep_eta_centrality" #28
                   ]

# Get the data
normalData = extractVariables(normalInputFile,200000)
distortedData = extractVariables(distortedInputFile,20000)

# Assemble data arrays
cutNormal = normalData["Label"]=="b"
cutDistorted = distortedData["Label"]=="b"
cutSignal = normalData["Label"]=="s"
X_train = buildArrays(allowedFeatures,cutNormal,normalData,0,180000,"TRAINING SAMPLE")
X_test = buildArrays(allowedFeatures,cutNormal,normalData,180000,20000,"TESTING SAMPLE - same distribution as training")
X_signal = buildArrays(allowedFeatures,cutSignal,normalData,0,20000,"TESTING SAMPLE - different distribution")
X_anomaly = buildArrays(allowedFeatures,cutDistorted,distortedData,0,20000,"ANOMALOUS SAMPLE - distorted distribution")

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
X_signal = min_max_scaler.transform(X_signal)
X_anomaly = min_max_scaler.transform(X_anomaly)

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
X_signal = min_max_scaler.transform(X_signal)
X_anomaly = min_max_scaler.transform(X_anomaly)

# One-class fitting with the SVM
clf = svm.OneClassSVM(nu=0.001, kernel='rbf')
clf.fit(X_train)
# Testing
predicted_same = clf.predict(X_train)
predicted_diff = clf.predict(X_test)
predicted_signal = clf.predict(X_signal)
predicted_anomaly = clf.predict(X_anomaly)
# Results
print "========================================"
print "Input events identified as anomalous, %:", 100.0*predicted_same[predicted_same==-1].size/predicted_same.size
print "Fresh events identified as anomalous, %:", 100.0*predicted_diff[predicted_diff==-1].size/predicted_diff.size
print "Signal events identified as anomalous, %:", 100.0*predicted_signal[predicted_signal==-1].size/predicted_signal.size
print "Anomalous events identified as anomalous, %:", 100.0*predicted_anomaly[predicted_anomaly==-1].size/predicted_anomaly.size







