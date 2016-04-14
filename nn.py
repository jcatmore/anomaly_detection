import sys
import pickle

import numpy as np
from sknn.mlp import Regressor, Layer
from sklearn import preprocessing

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D

from JamesTools import reconstructionError,reconstructionErrorByFeature,extractVariables,buildArrays,makeMetrics

###############################################
# MAIN PROGRAM

runTraining = True

normalInputFile = open("atlas-higgs-challenge-2014-v2.csv","r")
distortedInputFile = open("distorted.csv","r")
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
weightsFeature = [ "Weight" ]

# Get the data
normalData = extractVariables(normalInputFile,40000)
distortedData = extractVariables(distortedInputFile,20000)

# Assemble data arrays
cutNormal = normalData["Label"]=="b"
cutDistorted = distortedData["Label"]=="b"
cutSignal = normalData["Label"]=="s"
X_train = buildArrays(allowedFeatures,cutNormal,normalData,0,20000,"TRAINING SAMPLE")
W_train = buildArrays(weightsFeature,cutNormal,normalData,0,20000,"TRAINING SAMPLE WEIGHTS").reshape(X_train.shape[0])
X_test = buildArrays(allowedFeatures,cutNormal,normalData,20000,20000,"TESTING SAMPLE - same distribution as training")
X_signal = buildArrays(allowedFeatures,cutSignal,normalData,0,20000,"TESTING SAMPLE - different distribution")
X_anomaly = buildArrays(allowedFeatures,cutDistorted,distortedData,0,20000,"ANOMALOUS SAMPLE - distorted distribution")

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
X_signal = min_max_scaler.transform(X_signal)
X_anomaly = min_max_scaler.transform(X_anomaly)

# Set target equal to input - replicator NN
Y_train = X_train

# NEURAL NETWORK TRAINING AND TESTING
# Set up neural network
if runTraining:
    print "Starting neural network training"
    nn = Regressor(
               layers=[
                       Layer("Rectifier", units=100),
                       Layer("Rectifier", units=100),
                       Layer("Rectifier", units=100),
                       Layer("Linear")],
               learning_rate=0.01,
               n_iter=100)
    # Training
    nn.fit(X_train,Y_train,W_train)
    pickle.dump(nn, open('nn.pkl', 'wb'))
if not runTraining:
    nn = pickle.load(open('nn.pkl', 'rb'))


# Testing
predicted_same = nn.predict(X_train)
predicted_diff = nn.predict(X_test)
predicted_signal = nn.predict(X_signal)
predicted_anomaly = nn.predict(X_anomaly)

# Reconstruction error
rec_errors_same = reconstructionError(X_train,predicted_same)
rec_errors_diff = reconstructionError(X_test,predicted_diff)
rec_errors_sig = reconstructionError(X_signal,predicted_signal)
rec_errors_anomaly = reconstructionError(X_anomaly,predicted_anomaly)
print "Training error: ",sum(rec_errors_same)/rec_errors_same.shape[0]
print "Test error: ",sum(rec_errors_diff)/rec_errors_diff.shape[0]
print "Signal error: ",sum(rec_errors_sig)/rec_errors_sig.shape[0]
print "Anomaly input error: ",sum(rec_errors_anomaly)/rec_errors_anomaly.shape[0]

# Plotting - reconstruction errors
fig, axs = plt.subplots(3, 1)
ax1, ax2, ax3 = axs.ravel()
for ax in ax1, ax2, ax3:
    ax.set_ylabel("Events")
    ax.set_xlabel("log10(Reconstruction error)")

ax1.hist(rec_errors_same, 250, facecolor='blue', alpha=0.4, histtype='stepfilled')
ax2.hist(rec_errors_diff, 250, facecolor='green', alpha=0.4, histtype='stepfilled')
ax3.hist(rec_errors_diff, 250, facecolor='green', alpha=0.4, histtype='stepfilled')
ax3.hist(rec_errors_sig, 250, facecolor='red', alpha=0.4, histtype='stepfilled')
ax3.hist(rec_errors_anomaly, 250, facecolor='black', alpha=0.4, histtype='stepfilled')

# Plotting
true_positive,false_positive,precisions,recalls = makeMetrics(51,-5.0,0.1,rec_errors_anomaly,rec_errors_diff)
figB, axsB = plt.subplots(1,2)
axB1,axB2 = axsB.ravel()
# ROC
axB1.plot(false_positive, true_positive, label='ROC curve')
axB1.plot([0, 1], [0, 1], 'k--')
axB1.set_xlim([0.0, 1.0])
axB1.set_ylim([0.0, 1.05])
axB1.set_xlabel('False Anomaly Rate')
axB1.set_ylabel('True Anomaly Rate')
# Precision, recall
axB2.plot(recalls, precisions, label='Precision-recall curve')
axB2.plot([0, 1.0], [0.5, 0.5], 'k--')
axB2.set_xlim([0.0, 1.0])
axB2.set_ylim([0.0, 1.05])
axB2.set_xlabel('Recall')
axB2.set_ylabel('Precision')

plt.show()









