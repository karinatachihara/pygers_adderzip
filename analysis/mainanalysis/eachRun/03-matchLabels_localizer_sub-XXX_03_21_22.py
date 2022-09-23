#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:23:28 2022

@author: Work
"""
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
import csv

# Import machine learning libraries
import sklearn
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, PredefinedSplit
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score

from platform import python_version
print('The python version is {}.'.format(python_version()))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The numpy version is {}.'.format(np.__version__))
print('The nilearn version is {}.'.format(nilearn.__version__))
print('The seaborn version is {}.'.format(sns.__version__))

subList = np.array([5,6,7,8,9,10,11,12,13,15,17,18,19,20,22,23,24])
runList = np.array([10,11,12])

for eachParti in range (len(subList)):
    for eachRun in range(len(runList)):
        
        partiNum = subList[eachParti]
        if partiNum <10:
            subNum = '00'+str(partiNum)
        else:
            subNum = '0'+str(partiNum)
        sub = 'sub-'+subNum
        ses = 'ses-01'
        task='localizer'
        run = runList[eachRun]
        
        ROIs=['bilateral_hippo','bilateral_oc-temp']
        
        #find out how much to trunc from beginning and end
        file_name = '/jukebox/norman/karina/adderzip_fMRI/adderzip/data/supanalysis/TR_count/TR_count_%s.csv' % (subNum)
        
        countData = open(file_name)
        countData = csv.reader(countData)
        countData = list(countData)
        countData = countData[1::]
        countData = np.array(countData)
        countData = np.float64(countData)
        
        n_trunc_beginning= int(countData[run-1,1])#Number of volumes to trim from beginning of this run
        n_trunc_end= int(countData[run-1,3]) #Number of volumes to trim from end of run
        
        # Set printing precision
        np.set_printoptions(precision=2, suppress=True)
        
        # load some helper functions
        sys.path.insert(0, '/jukebox/norman/karina/adderzip_fMRI/adderzip/code/analysis/mainanalysis')
        import adderzip_utils
        from adderzip_utils import load_adderzip_stim_labels_localizer,load_adderzip_epi_data, shift_timing, label2TR, mask_data
        
        importlib.reload(adderzip_utils)
        
        # load some constants
        from adderzip_utils import adderzip_dir, adderzip_bids_dir, adderzip_label_dict, n_runs, run_names, adderzip_TR, adderzip_hrf_lag, TRs_run
        
        print('TASKS:', task)
        print('LIST OF ALL TASKS:', run_names)
        task_index = run_names.index(task)
        print('task index:', task_index)
        print('')
        
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
        #ses2_dir=adderzip_bids_dir + 'derivatives/fmriprep/%s/ses-02/func/' % sub
        
        print('bids dir = %s' % (adderzip_bids_dir))
        print('')
        print('subject dir = %s' % (ses1_dir))
        print('')
        print('output dir = %s' % (out_dir))
        print('')
        print('ROIs = %s' % (ROIs))
        print('Labels = %s' % (adderzip_label_dict))
        print('number of runs = %d' % (n_runs_localizer))
        print('TR = %s seconds' % (adderzip_TR))
        print('TRs per run before trimming = %s' % (TRs_run[task_index]))
        print('%d volumes trimmed from beginning of each run' % (n_trunc_beginning))
        print('%d volumes trimmed from end of each run depending on n_trunc_end file'% (n_trunc_end))
        print('TRs per run after trimming = %s' % (TRs_run_localizer))
        
        # load stimulus labels from regressor file for each run and concatenate
        # NOTE: Regressor files are already trimmed (beginning only), but not shifted, in Matlab using gen_loc_regressor_0101.m
        
        stim_label = load_adderzip_stim_labels_localizer(subNum,run)
                
        print('stim_label has shape: ', np.shape(stim_label))
        print('')
        print('stim_label looks like this')
        print(stim_label[:5,:])
        print('...')
        print(stim_label[-5:,:])
        
        
        # Plot the labels
        f, ax = plt.subplots(1,1, figsize = (12,5))
        ax.plot(stim_label[1:,3], c='orange')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('Trial')
        
        #reshape and edit stim labels format
        print('original shape', np.shape(stim_label))
        
        #image category is 0, 1, 2. change to 1,2,3
        if stim_label.shape[1]==8:
            imCat = stim_label[1:,3] 
            ones = np.ones(int(np.shape(imCat)[0]))
            ones = np.float64(ones)
            imCat = np.float64(imCat)
            newCol_imCat = imCat + ones
            zeros = np.zeros((int(np.shape(imCat)[0]+1),1))
            stim_label = np.hstack((stim_label,zeros))
            stim_label[1:,8] = newCol_imCat
        
            print('new shape after adding new column with changesd imCat labels', np.shape(stim_label))
        
        # Plot the labels
        f, ax = plt.subplots(1,1, figsize = (12,5))
        ax.plot(stim_label[1:,8], c='orange')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('Trial')
        
        # Plot the labels for 1st run
        f, ax = plt.subplots(1,1, figsize = (12,5))
        ax.plot(stim_label[1:90,6],stim_label[1:90,8], c='orange')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('TargetOnset Time')
        
        #targetOnset has an error. Fix by adding 7.5sec delay
        if stim_label.shape[1]==9:
            stim_label = np.hstack((stim_label,zeros))
            
            targetOnset = np.float64(stim_label[1:,6])
            ones = np.ones(stim_label.shape[0]-1)
            delay = np.float64(ones * 6)
            newTargetOnset = targetOnset + delay
            
            #print('newCol_targetOnset',newCol_targetOnset)
            stim_label[1:,9] = newTargetOnset
        
            print('new shape after adding new column with 7.5 sec delay', np.shape(stim_label))
        
        #change targetOnset time to TR
        #truncate initial 
        if stim_label.shape[1]==10:
            stim_label = np.hstack((stim_label,zeros))
            newCol_TR = np.float64(stim_label[1:,9])
            newCol_TR = newCol_TR / np.float64(adderzip_TR)
            
            newCol_TR = newCol_TR - np.float64(ones * n_trunc_beginning)
            
            stim_label[1:,10] = newCol_TR
        
            print('new shape after adding new column with trimmed TRs', np.shape(stim_label))
        
        #targetOnset starts count at each run. Make it continuous
        if stim_label.shape[1]==11:
            stim_label = np.hstack((stim_label,zeros))
            
            #eachRun = run-10
            #TRs_thisRun = np.float64(stim_label[1:,10])
            #newCol_TRs = TRs_thisRun + TRs_run_localizer*eachRun
            newCol_TRs = np.float64(stim_label[1:,10])
                
            stim_label[1:,11] = newCol_TRs
            
            print('new shape after adding new column with continuous TRs', np.shape(stim_label))
        
        # Plot the labels 
        f, ax = plt.subplots(1,1, figsize = (12,5))
        ax.plot(stim_label[1:,11],stim_label[1:,8], c='orange')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('TR')
        
        print('')
        print('stim_label looks like this')
        print(stim_label[:5,:])
        print('...')
        print(stim_label[-5:,:])
        
        
        
        # Shift the data labels to account for hemodynamic lag
        shift_size = int(adderzip_hrf_lag / adderzip_TR)  # Convert the shift into TRs
        print('shift by %s TRs' % (shift_size))
        
        #add shift to TRs
        TRs = stim_label[1:,11] 
        ones = np.ones(int(np.shape(TRs)[0]))
        shiftBy = ones * shift_size
        shiftBy = np.float64(shiftBy)
        TRs = np.float64(TRs)
        shiftedTRs = TRs + shiftBy
        #print('TRs',TRs)
        #print('shiftedTRs',shiftedTRs)
        shiftBy = int(shift_size) 
        shiftedTRs[-shiftBy:] = 0
        stim_label_shifted = np.copy(stim_label)
        stim_label_shifted[1:,11] = shiftedTRs
        
        
        # Plot the original and shifted labels
        f, ax = plt.subplots(1,1, figsize = (20,5))
        ax.plot(stim_label[1:50,10],stim_label[1:50,8], label='original', c='orange')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('TR')
        ax.legend()
        
        f, ax = plt.subplots(1,1, figsize = (20,5))
        ax.plot(stim_label_shifted[1:50,11],stim_label_shifted[1:50,8], label='shifted', c='blue')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('TR')
        ax.legend()
        
        print('stim_label_shifted shape', np.shape(stim_label_shifted))
        np.savetxt('/jukebox/norman/karina/adderzip_fMRI/adderzip/data/behavioral/info_processed_v1/locInfo/locInfo_'+subNum+'.csv',stim_label_shifted[1:,:],fmt='%10s',delimiter=",")
        
        print('')
        print('stim_label_shifted looks like this')
        print(stim_label_shifted[:5,:])
        print(stim_label_shifted[-5:,:])
        
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
            print("Loaded mask: %s" % (ROI_name))
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
            
            # plot mask overlayed on subject's T1
            plot_roi(mask, bg_img=t1_img, title=this_mask)
            
            # Load in data from matlab
            in_file = (adderzip_bids_dir + 'derivatives/firstlevel/%s/masked_epi_data_v1/threshold-75/%s_task-%s_run-0%i_space-T1w_trim%dandEndTRs_mask-%s.mat' % (sub, sub, task, run, n_trunc_beginning, this_mask))
            masked_data = scipy.io.loadmat(in_file);
            masked_data = np.array(masked_data['data']);
            masked_data_all[mask_counter] = masked_data
                
        # check dimensionality of the data and plot value of voxel_id across timeseries; make sure data are z-scored 
        num_voxels = [0] * len(mask_list)
        
        voxel_id = 100
        for mask_counter in range(len(mask_list)):
            this_mask = mask_list[mask_counter]
            print('voxel by TR matrix - shape: ', this_mask, masked_data_all[mask_counter].shape)
            num_voxels[mask_counter] = masked_data_all[mask_counter].shape[0] #save number of voxels in each mask
            
            f, ax = plt.subplots(1,1, figsize=(14,5))
            ax.plot(masked_data_all[mask_counter][voxel_id,:])
        
            ax.set_title('Voxel time series, mask = %s, voxel id = %d' % (this_mask, voxel_id))
            ax.set_xlabel('TR')
            ax.set_ylabel('Voxel Intensity')
            
        # Extract bold data for non-zero labels
        print('label list - shape: ', stim_label_shifted[1:,:].shape)
        
        def reshape_data(stim_label_shifted, masked_data_roi):
            label_index = np.float64(stim_label_shifted[1:,11])
            label_index = label_index.astype(int)
            print(label_index)
            
            # Pull out the indexes
            indexed_data = np.transpose(masked_data_roi[:,label_index])
            nonzero_labels = stim_label_shifted[1:,8]
            return indexed_data, nonzero_labels
        
        # Pull out the data from this ROI for these time points
        
        bold_data_reshaped = [0] * len(mask_list)
        
        for mask_counter in range(len(mask_list)):
            this_mask = mask_list[mask_counter]
            #print(this_mask)
        
            masked_data_roi = masked_data_all[mask_counter]
        
            bold_data, labels = reshape_data(stim_label_shifted, masked_data_roi)
        
            bold_data_reshaped[mask_counter] = bold_data
        
            # What is the dimensionality of the data? We need the first dim to be the same
            print('The %s has the dimensionality of: %d time points by %d voxels' % (this_mask, bold_data.shape[0], bold_data.shape[1]))
        
            print(labels)
        
            print('bold data for ',this_mask,'has shape',bold_data.shape)
            print('')
            print('bold data for ',this_mask,'looks like this')
            print(bold_data[:5,:])
            print('...')
            print(bold_data[-5:,:])
        
            np.savetxt(data_dir+'/bold_data_localizer/'+this_mask+'/bold_run-0'+str(run)+'_'+subNum+'.csv', bold_data, fmt='%f', delimiter=",")
            
            labels = np.float64(labels)
            np.savetxt(data_dir+'/label_list_localizer/labels_run-0'+str(run)+'_'+subNum+'.csv', labels, fmt='%f', delimiter=",")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        



























