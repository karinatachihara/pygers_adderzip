module load freesurfer
module load fsl

subj=$1

SUBJ_DIR=sub-$subj
STUDY_DIR=/jukebox/norman/karina/adderzip_fMRI/adderzip
DATA_DIR=$STUDY_DIR/data
BIDS_DIR=$DATA_DIR/bids
SCRIPT_DIR=$STUDY_DIR/code/mainanalysis
DERIV_DIR=$BIDS_DIR/derivatives
FIRSTLEVEL_DIR=$DERIV_DIR/firstlevel

T1=$DERIV_DIR/fmriprep/$SUBJ_DIR/ses-01/anat/${SUBJ_DIR}_ses-01_desc-preproc_T1w.nii.gz
T1_brainmask=$DERIV_DIR/fmriprep/${SUBJ_DIR}/ses-01/anat/${SUBJ_DIR}_ses-01_desc-brain_mask.nii.gz
T1_brain=$FIRSTLEVEL_DIR/${SUBJ_DIR}/masks/${SUBJ_DIR}_T1wbrain.nii.gz

fslmaths $T1 \
-mul $T1_brainmask \
$T1_brain