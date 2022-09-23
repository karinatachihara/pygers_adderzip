#!/usr/bin/env python
# coding: utf-8

# # Classifier cross-validation
# 
# ### Goals of this script
# 1. import labels (already trimmed but not shifted)
# 2. shift labels to account for hemodynamic lag
# 3. load voxel x TR matrix for ROI(s) of interest
# 4. reshape data (remove all fixation timepoints)
# 5. run classifiers and cross-validate on other localizer run (train on run1, test on run2 and vice-versa)
#     - model 1: solver='liblinear', class_weight=None
#     - model 2: solver='liblinear', class_weight='balanced'
#     - model 3: solver='lbfgs', class_weight=None
#     - model 4: solver='lbfgs', class_weight='balanced'
# 6. For each subject, save the average prediction probabilities for each classifier as well as the TR-by-TR prediction probabilities.
#     - classifier=['face_classifier', 'scene_classifier', 'object_classifier']
#     - trial_type=['face_trials', 'scene_trials', 'object_trials']
#         - classifier_scores[classifier][mask][run]
#         - prediction_probabilities[classifier][mask][TR]
#         - avg_classifier_evidence[trial_type][mask][run][classifier]
#         - num_voxels[mask]


# ## Import necessary packages

# In[7]:


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



# In[8]:


from platform import python_version
print('The python version is {}.'.format(python_version()))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The numpy version is {}.'.format(np.__version__))
print('The nilearn version is {}.'.format(nilearn.__version__))
print('The seaborn version is {}.'.format(sns.__version__))

ROIs=['bilateral_hippo','bilateral_oc-temp']

subList = np.array([5,6,7,8,9,10,11,12,13,15,17,18,20,22,23,24])
#subList = np.array([5])
runList = np.array([4,5])

for eachParti in range (len(subList)):
    for eachRun in range(len(runList)):
        
        partiNum = subList[eachParti]
        if partiNum <10:
            subNum = '00'+str(partiNum)
        else:
            subNum = '0'+str(partiNum)
        sub = 'sub-'+subNum
        ses = 'ses-01'
        task='imagine'
        run = runList[eachRun]
        # In[9]:
        
        
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
        n_task= int(countData[run-1,2])
        
        
        # ## Load settings
        
        # In[10]:
        
        
        # Set printing precision
        np.set_printoptions(precision=2, suppress=True)
        
        # load some helper functions
        sys.path.insert(0, '/jukebox/norman/karina/adderzip_fMRI/adderzip/code/analysis/mainanalysis')
        import adderzip_utils_imagine
        from adderzip_utils_imagine import load_adderzip_stim_labels_imagine,load_adderzip_epi_data, shift_timing, label2TR, mask_data
        
        # load some constants
        from adderzip_utils_imagine import adderzip_dir, adderzip_bids_dir, adderzip_label_dict_imagine,adderzip_TR, adderzip_hrf_lag, run_names, run_order_start, n_runs, TRs_run
        
        importlib.reload(adderzip_utils_imagine)
        
        print('TASK:', task)
        print('LIST OF TASKS:', run_names)
        task_index = run_names.index(task)
        print('task index:', task_index)
        print('')
        
        n_runs_imagine = n_runs[task_index]
        
        trials_run_imagine = 32
        trials_run_extra_imagine = 48
        
        anat_dir=adderzip_bids_dir + 'derivatives/deface/'
        out_dir= adderzip_bids_dir + 'derivatives/firstlevel/%s/' % sub
        mask_fold = adderzip_bids_dir + 'derivatives/firstlevel/%s/masks/' % sub
        data_dir='/jukebox/norman/karina/adderzip_fMRI/adderzip/data/mainanalysis/output'
        
        ses1_dir=adderzip_bids_dir + 'derivatives/fmriprep/%s/ses-01/func/' % sub
        
        print('bids dir = %s' % (adderzip_bids_dir))
        print('')
        print('subject dir = %s' % (ses1_dir))
        print('')
        print('output dir = %s' % (out_dir))
        print('')
        print('ROIs = %s' % (ROIs))
        print('Labels = %s' % (adderzip_label_dict_imagine))
        print('number of runs = %d' % (n_runs_imagine))
        print('TR = %s seconds' % (adderzip_TR))
        print('TRs before trimming for run %i= %s' % (run,n_trunc_beginning+n_trunc_end+n_task))
        print('%d volumes trimmed from beginning of each run' % (n_trunc_beginning))
        print('%d volumes trimmed from end of each run' % (n_trunc_end))
        print('TRs per run after trimming = %s' % (n_task))
        
        
        # ## Stimulus labels  - load truncated stimulus labels and shift labels 4.5 sec (3 TRs)
        
        # In[19]:
        
        
        # load stimulus labels from regressor file for each run and concatenate
        # NOTE: Regressor files are already trimmed (beginning only), but not shifted, in Matlab using gen_loc_regressor_0101.m
        
        
        stim_label = load_adderzip_stim_labels_imagine(subNum,run)
                
        print('stim_label has shape: ', np.shape(stim_label))
        print('')
        print('stim_label looks like this')
        print(stim_label[:5,:])
        print('...')
        print(stim_label[-5:,:])
        #print(stim_label)
        
        
        # Plot the labels
        f, ax = plt.subplots(1,1, figsize = (12,5))
        ax.plot(stim_label[:trials_run_imagine,6], c='orange')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('Trial')
        
        
        # In[20]:
        
        
        #reshape and edit stim labels format
        print('original shape', np.shape(stim_label))
        n_col_orig = stim_label.shape[1]
        
        #add new column for new reshaped data
        zeros = np.zeros((stim_label.shape[0],1))
        ones = np.ones((stim_label.shape[0],1))
        
        #trial number and experiment time is restarting at 0 for each run. fix it.
        print('fix trial number and expTime')
        #trial number 
        #stim_label[:,0] = np.arange(stim_label.shape[0])
        
        #exp time
        if stim_label.shape[1] ==n_col_orig:
            stim_label = np.hstack((stim_label,zeros))
            delayTime = stim_label[trials_run_imagine-1,8]
            print('delayTime',delayTime)
            newCol_expTime = stim_label[:,8]
            #newCol_expTime[trials_run_imagine:]= newCol_expTime[trials_run_imagine:] + delayTime
            stim_label[:,10] = newCol_expTime
        
        print('stim_label looks like this')
        print(stim_label[:5,:])
        print('...')
        print(stim_label[-5:,:])
        
        # Plot the labels for 1st run
        f, ax = plt.subplots(1,1, figsize = (12,5))
        ax.plot(stim_label[:trials_run_imagine,10],stim_label[:trials_run_imagine,6], c='orange')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('TargetOnset Time')
        
        #stim_label = np.hstack((stim_label,zeros))
        
        
        #change targetOnset time to TR
        print('change time to TR and trim beginning')
        #truncate initial 
        if stim_label.shape[1]<n_col_orig+2:
            stim_label = np.hstack((stim_label,zeros))
            
            #change time to TR
            newCol_TR = np.float64(stim_label[:,10])
            newCol_TR = newCol_TR / np.float64(adderzip_TR)
            
            #trim TR from beginning
            newCol_TR = newCol_TR - n_trunc_beginning
        
            stim_label[:,11] = newCol_TR
            
            #add TR end
            stim_label = np.hstack((stim_label,zeros))
            
            TRcount = stim_label[:,9]
            TRcount = TRcount/np.float64(adderzip_TR)
            newCol_TR_end = newCol_TR + TRcount
            
            stim_label[:,12] = newCol_TR_end
        
            print('new shape after adding new column with trimmed TRs', np.shape(stim_label))
        
        print('')
        print('stim_label_allruns looks like this')
        print(stim_label[:5,:])
        print('...')
        print(stim_label[-5:,:])
        
        # Plot the labels 
        f, ax = plt.subplots(1,1, figsize = (12,5))
        ax.plot(stim_label[:trials_run_imagine,11:12],stim_label[:trials_run_imagine,6], c='orange')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('TR')
        
        total_TRs = n_task
        label_mat = np.zeros((total_TRs,4))
        label_mat[:,1] = np.arange(total_TRs)
        
        for eachRow in range(stim_label.shape[0]):
            TR_start = int(stim_label[eachRow, 11])
            TR_end = int(stim_label[eachRow, 12])
            
            thisLabel = stim_label[eachRow, 6]
            thisItemGr = stim_label[eachRow,3]
            
            label_mat[TR_start:TR_end,0] = eachRow
            label_mat[TR_start:TR_end,2] = thisLabel
            label_mat[TR_start:TR_end,3] = thisItemGr


        
        print('')
        print('label_mat looks like this')
        print(label_mat[:5,:])
        print('...')
        print(label_mat[-5:,:])
        
        # Plot the labels 
        f, ax = plt.subplots(1,1, figsize = (12,5))
        ax.plot(label_mat[:,1],label_mat[:,2], c='orange')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('TR')
        
        
        # In[22]:
        
        
        # Shift the data labels to account for hemodynamic lag
        shift_size = int(adderzip_hrf_lag / adderzip_TR)  # Convert the shift into TRs
        print('shift by %s TRs' % (shift_size))
        
        #add shift to TRs
        TRs = label_mat[:,1] 
        ones = np.ones(int(np.shape(TRs)[0]))
        shiftBy = ones * shift_size
        shiftBy = np.float64(shiftBy)
        TRs = np.float64(TRs)
        shiftedTRs = TRs + shiftBy
        #shiftBy = int(shift_size+1) #last TR from last run also got truncated from bold data
        shiftBy = int(shift_size) 
        shiftedTRs[-shiftBy:] = 0
        stim_label_shifted = np.copy(label_mat)
        stim_label_shifted[:,1] = shiftedTRs
        
        
        # Plot the original and shifted labels
        f, ax = plt.subplots(1,1, figsize = (20,5))
        ax.plot(label_mat[:trials_run_imagine,1],label_mat[:trials_run_imagine,2], label='original', c='orange')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('TR')
        ax.legend()
        
        f, ax = plt.subplots(1,1, figsize = (20,5))
        ax.plot(stim_label_shifted[:trials_run_imagine,1],stim_label_shifted[:trials_run_imagine,2], label='shifted', c='blue')
        ax.set_ylabel('Stimulus category label')
        ax.set_xlabel('TR')
        ax.legend()
        
        print('stim_label_shifted shape', np.shape(stim_label_shifted))
        np.savetxt('/jukebox/norman/karina/adderzip_fMRI/adderzip/data/behavioral/info_processed_v1/imagineInfo/imagineInfo_'+subNum+'.csv',stim_label_shifted[1:,:],fmt='%10s',delimiter=",")
        
        print('')
        print('stim_label looks like this')
        print(stim_label_shifted[:5,:])
        print(stim_label_shifted[-5:,:])
        
        
        # ## OR if voxel x TR matrix already exists, load matrix:
        
        # In[23]:
        
        
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
        
        
        # In[24]:
        
        
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
        
        
        # In[25]:
        
        
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
            
        
        
        # ## Reshape labels and data 
        # Extract the time points for which we have stimulus labels. For classifier B, we remove all rest TRs. 
        
        # In[26]:
        
        
        # Extract bold data for non-zero labels
        print('label list - shape: ', stim_label_shifted.shape)
        
        def reshape_data(stim_label_shifted, masked_data_roi):
            label_index = np.float64(stim_label_shifted[:,1])
            label_index = label_index.astype(int)
            #print(label_index)
            
            # Pull out the indexes
            indexed_data = np.transpose(masked_data_roi[:,label_index])
            nonzero_labels = stim_label_shifted[:,2]
            itemGr = stim_label_shifted[:,3]
            return indexed_data, nonzero_labels,itemGr
        
        # Pull out the data from this ROI for these time points
        
        bold_data_reshaped = [0] * len(mask_list)
        
        for mask_counter in range(len(mask_list)):
            this_mask = mask_list[mask_counter]
            print(this_mask)
        
            masked_data_roi = masked_data_all[mask_counter]
        
            bold_data, labels, itemGr = reshape_data(stim_label_shifted, masked_data_roi)
        
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
        
            np.savetxt(data_dir+'/bold_data_imagine/'+this_mask+'/bold_run-0'+str(run)+'_'+subNum+'.csv', bold_data, fmt='%f', delimiter=",")
            np.savetxt(data_dir+'/label_list_imagine/labels_run-0'+str(run)+'_'+subNum+'.csv', labels, fmt='%f', delimiter=",")
            np.savetxt(data_dir+'/itemGroup_imagine/itemGroup_run-0'+str(run)+'_'+subNum+'.csv', itemGr, fmt='%f', delimiter=",")                
