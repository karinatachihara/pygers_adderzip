#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:55:26 2022

@author: Work
"""


import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from nilearn.masking import intersect_masks
from nilearn import image
from nilearn.plotting import plot_roi
from nilearn.plotting import plot_anat
from nilearn.plotting import plot_epi
from nilearn.image.image import mean_img
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt 
#import seaborn as sns 
import scipy.io
import importlib
import csv

#subList = np.array([5,6,7,8,9,10,11,12,13,15,17,18,19,20,22,23,24])
subList = np.array([9,10,11,12,13,15,17,18,19,20,22,23,24])
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
        
        # load some helper functions
        sys.path.insert(0, '/jukebox/norman/karina/adderzip_fMRI/adderzip/code/analysis/mainanalysis')
        import adderzip_utils
        from adderzip_utils import load_adderzip_epi_data, load_data
        
        importlib.reload(adderzip_utils)
        
        # load some constants
        from adderzip_utils import adderzip_dir, adderzip_bids_dir, adderzip_TR, adderzip_hrf_lag, run_names, run_order_start, n_runs, TRs_run
        
        print('TASK:', task)
        print('LIST OF TASKS:', run_names)
        task_index = run_names.index(task)
        print('task index:', task_index)
        print('')
        
        adderzip_n_runs = n_runs[task_index]
        adderzip_TRs_run = TRs_run[task_index]
        
        bold_dir=adderzip_bids_dir + 'derivatives/fmriprep/%s/%s/func/' % (sub, ses)
        anat_dir=adderzip_bids_dir + 'derivatives/deface/'
        out_dir= adderzip_bids_dir + 'derivatives/firstlevel/%s/%s/' % (sub, ses)
        mask_fold = adderzip_bids_dir + 'derivatives/firstlevel/%s/masks/' % sub
        
        print('bids dir = %s' % (adderzip_bids_dir))
        print('')
        print('subject dir = %s' % (bold_dir))
        print('')
        print('output dir = %s' % (out_dir))
        print('')
        print('number of runs = %d' % (adderzip_n_runs))
        print('TR = %s seconds' % (adderzip_TR))
        print('TRs per run = %s' % (adderzip_TRs_run))
        print('trim %d volumes from beginning of each run' % (n_trunc_beginning))
        print('trim %d volumes from end of each run depending on n_trunc_end file' % (n_trunc_end))
        
        # Use only the last 6 columns
        confounds=[]
        mc_all=[]
        
        fname='_ses-01_task-localizer_run-%i_desc-confounds_timeseries.tsv' % (run)
        confounds = pd.read_csv(bold_dir + sub + fname,  sep='\t', header=(0))
        
        print('before trimming')
        print(confounds.shape)
        
        confounds_selected=confounds[['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']][n_trunc_beginning:-n_trunc_end]
        confounds_selected=pd.DataFrame(confounds_selected)
        confounds_selected.to_csv(out_dir + sub + '_ses-01_task-localizer_run-0%i_confounds_selected_trim%dand%dTRs.txt' % (run, n_trunc_beginning, n_trunc_end), index=False, sep='\t', mode='w')
        
        print('run #%d' % (run))
        print(confounds_selected)
        
        mask_imgs=[]
        mask_name = bold_dir + sub + '_ses-01_task-localizer_run-%i_space-T1w_desc-brain_mask.nii.gz' % run
        mask_imgs.append(mask_name)
            
        avg_mask=intersect_masks(mask_imgs, threshold=0.5, connected=True)
        
        # plot
        t1_file = anat_dir + sub + '_ses-01_desc-preproc_T1w_defaced.nii.gz'
        t1_img = image.load_img(t1_file)
        plot_roi(avg_mask, bg_img=t1_img)
        
        # Save the mask
        output_name = mask_fold + '%s_%s_brain.nii.gz' % (sub, ses)
        print('Save average mask:', output_name)
        print('')
        
        dimsize=avg_mask.header.get_zooms()
        affine_mat = avg_mask.affine
        print('Mask dimensions:', dimsize)
        print('')
        print('Affine:')
        print(affine_mat)
        
        hdr = avg_mask.header  # get a handle for the .nii file's header
        hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
        nib.save(avg_mask, output_name)
        
        epi_trunc=[]
        
        epi_file=bold_dir + sub + '_ses-01_task-localizer_run-%i_space-T1w_desc-preproc_bold.nii.gz' % run
        epi_data=nib.load(epi_file)
        epi=epi_data.get_fdata()
        #hdr=epi_data.get_data.hdr()
        
        if n_trunc_end <= 0:
            epi_trunc =np.zeros((epi_data.shape[0], epi_data.shape[1], epi_data.shape[2], epi_data.shape[3]-n_trunc_beginning))
            epi_trunc[:, :, :, :] = epi[:,:,:,n_trunc_beginning:]
        else:
            epi_trunc =np.zeros((epi_data.shape[0], epi_data.shape[1], epi_data.shape[2], epi_data.shape[3]-n_trunc_beginning-n_trunc_end))
            epi_trunc[:, :, :, :] = epi[:,:,:,n_trunc_beginning:-n_trunc_end]
        
        #epi_truncated
        print('run #%d' % (run))
        print('Original:', epi_data.shape, '  ', 'Truncated:', epi_trunc.shape)
        dimsize=epi_data.header.get_zooms()
        print('Dimensions:', dimsize)
        orig_dimsize=dimsize
        
        affine_mat = epi_data.affine  # What is the orientation of the data
        print('Affine:')
        print(affine_mat)
        print('')
        
        # Save the volume
        output_name = (out_dir + '%s_ses-01_task-localizer_run-0%i_space-T1w_desc-preproc_bold_trim%dandEndTRs.nii.gz' % (sub, run, n_trunc_beginning))
        bold_nii = nib.Nifti1Image(epi_trunc, affine_mat)
        hdr = bold_nii.header  # get a handle for the .nii file's header
        hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], dimsize[3]))
        nib.save(bold_nii, output_name)
        
        print('bold shape:', bold_nii.shape)
        
        epi_mask_data_all=[]
        
        # 1. Load the fMRI data 
        
        epi_masker= NiftiMasker(
            mask_img=avg_mask,  
            high_pass=1/128,
            standardize=True,  # Are you going to zscore the data across time?
            t_r=adderzip_TR, 
            #memory='nilearn_cache',  # Caches the mask in the directory given as a string here so that it is easier to load and retrieve
            #memory_level=1,  # How much memory will you cache?
            verbose=0
        )
        
        epi_file=out_dir + '%s_ses-01_task-localizer_run-0%i_space-T1w_desc-preproc_bold_trim%dandEndTRs.nii.gz' % (sub, run, n_trunc_beginning)
        
        epi_mask_data = epi_masker.fit_transform(epi_file)
        
        # Save the volume
        print('Saving trimmed and normalized volume for run', run)
        
        affine_mat = avg_mask.affine #should be the same as the epi data
        avg_mask.shape
        coords = np.where(avg_mask.get_fdata())
        bold_vol=[]
        bold_vol =np.zeros((avg_mask.shape[0], avg_mask.shape[1], avg_mask.shape[2], epi_mask_data.shape[0]))
        bold_vol[coords[0], coords[1], coords[2], :] = epi_mask_data.T
        print('epi_mask_data shape:', bold_vol.shape)
        print('')
        
        output_name = (out_dir + '%s_ses-01_task-localizer_run-0%i_space-T1w_desc-preproc_bold_trim%dandEndTRs_normalized.nii.gz' % (sub, run, n_trunc_beginning))
        bold_nii = nib.Nifti1Image(bold_vol, affine_mat)
        hdr = bold_nii.header  # get a handle for the .nii file's header
        hdr.set_zooms((orig_dimsize[0], orig_dimsize[1], orig_dimsize[2], orig_dimsize[3]))
        nib.save(bold_nii, output_name)
        
        print('Volumes saved')
        
        avg_mask.shape
        coords = np.where(avg_mask.get_fdata())
        #print(avg_mask)
        
        dimsize=avg_mask.header.get_zooms()
        print('Voxel dimensions:', dimsize)
        
        affine_mat = avg_mask.affine  # What is the orientation of the data
        
        bold_vol=[]
        bold_vol =np.zeros((avg_mask.shape[0], avg_mask.shape[1], avg_mask.shape[2], epi_mask_data.shape[0]))
        bold_vol[coords[0], coords[1], coords[2], :] = epi_mask_data.T
        
        print('avg_mask shape:', avg_mask.shape)
        print('epi_mask_data shape:', bold_vol.shape)


















































