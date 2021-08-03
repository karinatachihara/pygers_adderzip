#! /bin/bash

# LOAD GLOBAL VARIABLES AND MODULES ON THE CLUSTER
source globals.sh   
module load fsl/6.0.2
module load pydeface/2.0.0

sid=$1
#session=$2

subj_dir=sub-$sid

# deface T1
#T1=`find $bids_dir/sub-$sid/ses-$session/anat -name "*T1w.nii.gz"`
T1=$bids_dir/derivatives/fmriprep/$subj_dir/ses-01/anat/${subj_dir}_ses-01_desc-preproc_T1w.nii.gz
pydeface $T1

# move defaced T1 to extra directory
#T1_defaced=`find $bids_dir/sub-$sid/ses-$session/anat -name "*T1w_defaced.nii.gz"`
T1_defaced=$bids_dir/derivatives/fmriprep/$subj_dir/ses-01/anat/${subj_dir}_ses-01_desc-preproc_T1w_defaced.nii.gz
#mv $T1_defaced $defaced_dir/
mv $T1_defaced $defaced_dir