#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:45:41 2022

@author: Work
"""
sub = 'sub-008'
subNum = '008'
ses = 'ses-01'
task = 'localizer'

n_trunc_beginning=14 #Number of volumes to trim from beginning of run
n_trunc_end=10 #Number of volumes to trim from end of run
ROIs=['bilateral_hippo','bilateral_oc-temp',]

import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import nibabel as nib
import nilearn
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from nilearn.masking import intersect_masks
from nilearn import image
from nilearn import plotting
from nilearn.plotting import plot_roi
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import scipy.io
#from mpi4py import MPI
import os
import pickle 
import time
from scipy.sparse import random
from scipy.stats import zscore
from scipy.spatial.distance import euclidean
from pathlib import Path
from shutil import copyfile
import seaborn as sns
import importlib

# Import machine learning libraries
import sklearn
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, PredefinedSplit
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score

from platform import python_version

# Set printing precision
np.set_printoptions(precision=2, suppress=True)

# load some helper functions
sys.path.insert(0, '/jukebox/norman/karina/adderzip_fMRI/adderzip/code/mainanalysis')
import adderzip_utils
from adderzip_utils import load_adderzip_stim_labels_localizer,load_adderzip_epi_data, shift_timing, label2TR, mask_data

importlib.reload(adderzip_utils)

# load some constants
from adderzip_utils import adderzip_dir, adderzip_bids_dir, adderzip_label_dict, n_runs, run_names, adderzip_TR, adderzip_hrf_lag, TRs_run

task_index = run_names.index(task)

n_runs_localizer = n_runs[task_index]

TRs_run_localizer=TRs_run[task_index]-n_trunc_beginning-n_trunc_end

trials_run_localizer = 90

anat_dir=adderzip_bids_dir + 'derivatives/deface/'
out_dir= adderzip_bids_dir + 'derivatives/firstlevel/%s/' % sub
mask_fold = adderzip_bids_dir + 'derivatives/firstlevel/%s/masks/' % sub
#mask_fold_hipp = adderzip_bids_dir + 'derivatives/firstlevel/%s/masks/' % sub
data_dir='/jukebox/norman/karina/adderzip_fMRI/adderzip/data/mainanalysis/output'

#ses0_dir=adderzip_bids_dir + 'derivatives/fmriprep/%s/ses-00/func/' % sub
ses1_dir=adderzip_bids_dir + 'derivatives/fmriprep/%s/ses-01/func/' % sub

stim_label = [];
stim_label_allruns = [];

stim_label_allruns = load_adderzip_stim_labels_localizer(subNum)
        

# load stimulus labels from regressor file for each run and concatenate
# NOTE: Regressor files are already trimmed (beginning only), but not shifted, in Matlab using gen_loc_regressor_0101.m

stim_label = [];
stim_label_allruns = [];

stim_label_allruns = load_adderzip_stim_labels_localizer(subNum)

#image category is 0, 1, 2. change to 1,2,3
if stim_label_allruns.shape[1]==8:
    imCat = stim_label_allruns[1:,3] 
    ones = np.ones(int(np.shape(imCat)[0]))
    ones = np.float64(ones)
    imCat = np.float64(imCat)
    newCol_imCat = imCat + ones
    zeros = np.zeros((int(np.shape(imCat)[0]+1),1))
    stim_label_allruns = np.hstack((stim_label_allruns,zeros))
    stim_label_allruns[1:,8] = newCol_imCat

#targetOnset has an error. Fix by adding 7.5sec delay
if stim_label_allruns.shape[1]==9:
    stim_label_allruns = np.hstack((stim_label_allruns,zeros))
    
    targetOnset = np.float64(stim_label_allruns[1:,6])
    ones = np.ones(stim_label_allruns.shape[0]-1)
    delay = np.float64(ones * 7.5)
    newTargetOnset = targetOnset + delay
    
    #print('newCol_targetOnset',newCol_targetOnset)
    stim_label_allruns[1:,9] = newTargetOnset

    
#change targetOnset time to TR
#truncate initial 
if stim_label_allruns.shape[1]==10:
    stim_label_allruns = np.hstack((stim_label_allruns,zeros))
    newCol_TR = np.float64(stim_label_allruns[1:,9])
    newCol_TR = newCol_TR / np.float64(adderzip_TR)
    
    newCol_TR = newCol_TR - np.float64(ones * n_trunc_beginning)
    
    stim_label_allruns[1:,10] = newCol_TR

    
#targetOnset starts count at each run. Make it continuous
if stim_label_allruns.shape[1]==11:
    stim_label_allruns = np.hstack((stim_label_allruns,zeros))
    for eachRun in range(n_runs_localizer):
        start = trials_run_localizer*eachRun + 1
        end = start + trials_run_localizer
        TRs_thisRun = np.float64(stim_label_allruns[start:end,10])
        cont_diff = np.ones(trials_run_localizer)
        cont_diff = np.float64(cont_diff * TRs_run_localizer)
        new_TRs_thisRun = TRs_thisRun + cont_diff*eachRun

        if eachRun == 0:
            newCol_TRs = new_TRs_thisRun
        else:
            newCol_TRs = np.hstack((newCol_TRs,new_TRs_thisRun))
        
    stim_label_allruns[1:,11] = newCol_TRs
    
    
# Shift the data labels to account for hemodynamic lag
shift_size = int(adderzip_hrf_lag / adderzip_TR)  # Convert the shift into TRs

#add shift to TRs
TRs = stim_label_allruns[1:,11] 
ones = np.ones(int(np.shape(TRs)[0]))
shiftBy = ones * shift_size
shiftBy = np.float64(shiftBy)
TRs = np.float64(TRs)
shiftedTRs = TRs + shiftBy
shiftBy = int(shift_size+1) #last TR from last run also got truncated from bold data
shiftedTRs[-shiftBy:] = 0
stim_label_allruns_shifted = np.copy(stim_label_allruns)
stim_label_allruns_shifted[1:,11] = shiftedTRs

# load defaced T1 image (merged T1 from fmriprep)
t1_file = anat_dir + sub +'_' +ses + '_desc-preproc_T1w_defaced.nii.gz'
t1_img = image.load_img(t1_file) 

# Make a function to load the mask data
def load_adderzip_mask(ROI_name, sub):
    """Load the mask for the svd data 
    Parameters
    ----------
    ROI_name: string
    sub: string 
    
    Return
    ----------
    the requested mask
    """    
    # load the mask
    #if ROI_name == 'bilateral_hippo':
        #mask_fold=mask_fold_other
    #else:
        #mask_fold=mask_fold_hipp
    maskfile = (mask_fold + sub + "_%s.nii.gz" % (ROI_name))
    mask = nib.load(maskfile)
    return mask


# load voxel x TR data for each ROI
mask_list = ROIs
masked_data = []
masked_data_all = [0] * len(mask_list)

for mask_counter in range(len(mask_list)):
        # load the mask for the corresponding ROI
        this_mask = mask_list[mask_counter]
        
        #if this_mask == 'bilateral_hippo':
        #    mask_fold=mask_fold_other
        #else:
            #mask_fold=mask_fold_hipp
        
        mask = load_adderzip_mask(mask_list[mask_counter], sub)
        
        # Load in data from matlab
        in_file = (adderzip_bids_dir + 'derivatives/firstlevel/%s/masked_epi_data_v1/threshold-75/%s_task-%s_run-ALL_space-T1w_trim%dand%dTRs_mask-%s.mat' % (sub, sub, task, n_trunc_beginning, n_trunc_end, this_mask))
        masked_data = scipy.io.loadmat(in_file);
        masked_data = np.array(masked_data['data']);
        masked_data_all[mask_counter] = masked_data

# check dimensionality of the data and plot value of voxel_id across timeseries; make sure data are z-scored 
num_voxels = [0] * len(mask_list)

voxel_id = 100
for mask_counter in range(len(mask_list)):
    this_mask = mask_list[mask_counter]
    num_voxels[mask_counter] = masked_data_all[mask_counter].shape[0] #save number of voxels in each mask
    

def reshape_data(stim_label_allruns_shifted, masked_data_roi):
    label_index = np.float64(stim_label_allruns_shifted[1:,11])
    label_index = label_index.astype(int)
    #print(label_index)
    
    # Pull out the indexes
    indexed_data = np.transpose(masked_data_roi[:,label_index])
    nonzero_labels = stim_label_allruns_shifted[1:,8]
    return indexed_data, nonzero_labels

# Pull out the data from this ROI for these time points

for mask_counter in range(len(mask_list)):
        this_mask = mask_list[mask_counter]
        #print(this_mask)
        
        masked_data_roi = masked_data_all[mask_counter]
        
        bold_data, labels = reshape_data(stim_label_allruns_shifted, masked_data_roi)


find_ones = np.where(labels[:] == '1.0')[0] #to double check that numbers line up
find_twos = np.where(labels[:] == '2.0')[0] #to double check that numbers line up
find_threes = np.where(labels[:] == '3.0')[0] #to double check that numbers line up

def classifier(bold_train, labels_train, bold_test, labels_test, model,linReg):
    # normalize the data 
    scaler = StandardScaler()
    bold_train = scaler.fit_transform(bold_train)
    bold_test = scaler.transform(bold_test)


    # Fit the model
    model.fit(bold_train, labels_train)

    # Compute your evaluation on the test set
    score = model.score(bold_test, labels_test)
    
    if linReg == 1:
        prediction = model.predict(bold_test)
        prediction_prob = model.predict_proba(bold_test)
    else:
        prediction = 'NA'
        prediction_prob = 'NA'
    
    return score, prediction, prediction_prob

model_linearSVC = LinearSVC(C=1)
model_linReg_1 = LogisticRegression(C=1.0, random_state=34, solver='liblinear')

def split(bold_data,labels,trials_run_localizer):
    
    run1_bold = bold_data[0:trials_run_localizer,:]
    run1_labels = labels[0:trials_run_localizer]

    run2_bold = bold_data[trials_run_localizer:(trials_run_localizer*2),:]
    run2_labels = labels[trials_run_localizer:(trials_run_localizer*2)]

    run3_bold = bold_data[(trials_run_localizer*2):(trials_run_localizer*3),:]
    run3_labels = labels[(trials_run_localizer*2):(trials_run_localizer*3)]

    fold1_bold_train = np.vstack((run2_bold,run3_bold))
    fold1_labels_train = np.hstack((run2_labels,run3_labels))

    fold1_bold_test = run1_bold
    fold1_labels_test = run1_labels

    fold2_bold_train = np.vstack((run1_bold,run3_bold))
    fold2_labels_train = np.hstack((run1_labels,run3_labels))

    fold2_bold_test = run2_bold
    fold2_labels_test = run2_labels

    fold3_bold_train = np.vstack((run1_bold,run2_bold))
    fold3_labels_train = np.hstack((run1_labels,run2_labels))

    fold3_bold_test = run3_bold
    fold3_labels_test = run3_labels
    
    return fold1_bold_train,fold2_bold_train,fold3_bold_train,fold1_labels_train,fold2_labels_train,fold3_labels_train,fold1_bold_test,fold2_bold_test,fold3_bold_test,fold1_labels_test,fold2_labels_test,fold3_labels_test


# binary classification of category vs. not category
n_classifierTypes = 3
totalTrials = trials_run_localizer * n_runs_localizer * n_classifierTypes
partiData = np.zeros((totalTrials,6))

#subject
partiData[:,0] = subNum

#classifier type
ones_c = np.ones(trials_run_localizer* n_runs_localizer)
twos_c = np.ones(trials_run_localizer* n_runs_localizer)*2
threes_c = np.ones(trials_run_localizer* n_runs_localizer)*3
classiCol = np.hstack((ones_c,twos_c,threes_c))
partiData[:,1] = classiCol

#run
ones = np.ones(trials_run_localizer)
twos = np.ones(trials_run_localizer)*2
threes = np.ones(trials_run_localizer)*3
runCol = np.hstack((ones,twos,threes))
runCol = np.tile(runCol, n_classifierTypes)
partiData[:, 2] = runCol

#TR
TRs_oneRun = np.arange(1,(trials_run_localizer+1))
TRsCol = np.tile(TRs_oneRun,n_runs_localizer)
TRsCol = np.tile(TRsCol,n_classifierTypes)
partiData[:,3] = TRsCol

# print('partiData')
# print(partiData.shape)
# print(partiData)

#face vs. non-face
labels_face = np.copy(labels)
labels_face[find_ones] = 1
labels_face[find_twos] = 0
labels_face[find_threes] = 0

#scene vs. non-scene
labels_scene = np.copy(labels)
labels_scene[find_ones] = 0
labels_scene[find_twos] = 1
labels_scene[find_threes] = 0

#object vs. non-object
labels_object = np.copy(labels)
labels_object[find_ones] = 0
labels_object[find_twos] = 0
labels_object[find_threes] = 1

for eachClassi in range(n_classifierTypes):
    
    if eachClassi == 0:
        labels = labels_face
        classiType = 'face'
    elif eachClassi == 1:
        labels = labels_scene
        classiType = 'scene'
    elif eachClassi == 2:
        labels = labels_object
        classiType = 'object'
    
    fold1_bold_train,fold2_bold_train,fold3_bold_train,fold1_labels_train,fold2_labels_train,fold3_labels_train,fold1_bold_test,fold2_bold_test,fold3_bold_test,fold1_labels_test,fold2_labels_test,fold3_labels_test = split(bold_data,labels,trials_run_localizer)

    fold1_score,fold1_prediction,fold1_prediction_prob = classifier(fold1_bold_train,fold1_labels_train,fold1_bold_test,fold1_labels_test,model_linReg_1,1)
    fold2_score,fold2_prediction,fold2_prediction_prob = classifier(fold2_bold_train,fold2_labels_train,fold2_bold_test,fold2_labels_test,model_linReg_1,1)
    fold3_score,fold3_prediction,fold3_prediction_prob = classifier(fold3_bold_train,fold3_labels_train,fold3_bold_test,fold3_labels_test,model_linReg_1,1)
    
    start = eachClassi * trials_run_localizer * n_runs_localizer
    end = eachClassi * trials_run_localizer * n_runs_localizer + trials_run_localizer
    
    partiData[start:end,4] = fold1_labels_test      
    partiData[(start+trials_run_localizer):(end+trials_run_localizer),4] = fold2_labels_test
    partiData[(start+trials_run_localizer*2):(end+trials_run_localizer*2),4] = fold3_labels_test

    partiData[start:end,5] = fold1_prediction_prob[:,1]
    partiData[(start+trials_run_localizer):(end+trials_run_localizer),5] = fold2_prediction_prob[:,1]
    partiData[(start+trials_run_localizer*2):(end+trials_run_localizer*2),5] = fold3_prediction_prob[:,1]

np.savetxt(data_dir+'/classifier_localizer_oc-temp/classifier_results_'+subNum+'.csv', partiData, fmt='%f', delimiter=",", header = 'subject,classifier,run,TR,trial_type,prob')



























