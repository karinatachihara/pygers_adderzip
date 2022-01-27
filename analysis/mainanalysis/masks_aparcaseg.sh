#! /bin/bash

set -e #stop immediately when error occurs

module load freesurfer
module load fsl

subj=$1

SUBJ_DIR=sub-$subj
STUDY_DIR=/jukebox/norman/karina/adderzip_fMRI/adderzip/
DATA_DIR=$STUDY_DIR/data
BIDS_DIR=$DATA_DIR/bids
SCRIPT_DIR=$STUDY_DIR/code/mainanalysis
DERIV_DIR=$BIDS_DIR/derivatives
FIRSTLEVEL_DIR=$DERIV_DIR/firstlevel
FREESURFER_DIR=$DERIV_DIR/freesurfer/$SUBJ_DIR/mri
MASK_DIR=$FIRSTLEVEL_DIR/$SUBJ_DIR/masks

T1=$DERIV_DIR/fmriprep/$SUBJ_DIR/anat/${SUBJ_DIR}_desc-preproc_T1w.nii.gz
BOLD_REF=$DERIV_DIR/fmriprep/$SUBJ_DIR/ses-01/func/${SUBJ_DIR}_ses-01_task-localizer_run-10_space-T1w_boldref.nii.gz
aPARC=$FREESURFER_DIR/aparc.a2009s+aseg.mgz #parcellation is the cortical ribbon and segmentation are the subcortical volumes

# convert reference image to mgz
mri_convert $BOLD_REF $FIRSTLEVEL_DIR/$SUBJ_DIR/ses-01/${SUBJ_DIR}_ses-01_task-localizer_run-01_space-T1w_boldref.mgz

# convert aparc.aseg mgz label file into volume space (i.e., bold resolution...giving it boldref as the target space)
mri_label2vol --seg $aPARC --temp $BOLD_REF --o $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.mgz --fillthresh 0.5 --regheader $aPARC

# convert resample mgz label file to nifti 
mri_convert $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.mgz $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz
# mri_convert -rl $FIRSTLEVEL_DIR/$SUBJ_DIR/ses-00/${SUBJ_DIR}_ses-00_task-localizer_run-01_space-T1w_boldref.mgz -rt nearest $aPARC $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz


# # open in Freesurfer (example)
# freeview -v derivatives/freesurfer/sub-129/mri/orig.mgz \
# derivatives/freesurfer/sub-129/mri/aparc.a2009s+aseg.mgz:colormap=lut:opacity=0.4 \
# -f derivatives/freesurfer/sub-129/surf/lh.white:annot=aparc.annot.a2009s

# Extract values for occipito-temporal mask - left, right, and bilateral (I used bilateral_oc-temp mask made here for localizer classification analysis)
# lookup table https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
# lateral occipito temporal gyrus (G_oc-temp_lat-fusifor label L:11121 R:12121)
# parahippocampal gyrus (G_oc-temp_med-Parahip label L:11123 R:12123)
# lateral occipito-temporal sulcus (S_oc-temp_lat label L:11161 R:12161)
# medial occipito-temporal sulcus (S_oc-temp_med_and_Lingual L:11162 R:12162)

fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 17 -uthr 17 -bin $MASK_DIR/${SUBJ_DIR}_left_hippo.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 53 -uthr 53 -bin $MASK_DIR/${SUBJ_DIR}_right_hippo.nii.gz
fslmaths $MASK_DIR/${SUBJ_DIR}_left_hippo.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_hippo.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_hippo.nii.gz

# Make masks for RSA sanity check (item-specific scene selective areas, and a frontal and sensory control area)
# calcarine sulcus (S_calcarine label L:11145 R:12145 )
# cuneus (G_cuneus label L:11111 R:12111 )
# lingual gyrus (G_oc-temp_med-Lingual label L:11122 R:12122 )
# Inferior occipital gyrus and sulcus (G_and_S_occipital_inf label L:11102 R:12102 )
# Middle occipital gyrus (G_occipital_middle label L:11119 R:12119 )
# Superior occipital gyrus (G_occipital_sup label L:11120 R:12120 )
# Olfactory sulcus (S_orbital_med-olfact label L:11164 R:12164 )
# Orbital part of the inferior frontal gyrus (G_front_inf-Orbital label L:11113 R:12113 )

# # Calcarine sulcus
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11145 -uthr 11145 -bin $MASK_DIR/${SUBJ_DIR}_left_calcarine.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12145 -uthr 12145 -bin $MASK_DIR/${SUBJ_DIR}_right_calcarine.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_left_calcarine.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_calcarine.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_calcarine.nii.gz

# # Cuneus
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11111 -uthr 11111 -bin $MASK_DIR/${SUBJ_DIR}_left_cuneus.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12111 -uthr 12111 -bin $MASK_DIR/${SUBJ_DIR}_right_cuneus.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_left_cuneus.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_cuneus.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_cuneus.nii.gz

# # Lingual gyrus
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11122 -uthr 11122 -bin $MASK_DIR/${SUBJ_DIR}_left_lingual.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12122 -uthr 12122 -bin $MASK_DIR/${SUBJ_DIR}_right_lingual.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_left_lingual.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_lingual.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_lingual.nii.gz

# # Inferior occipital gyrus and sulcus
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11102 -uthr 11102 -bin $MASK_DIR/${SUBJ_DIR}_left_occipital_inferior.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12102 -uthr 12102 -bin $MASK_DIR/${SUBJ_DIR}_right_occipital_inferior.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_left_occipital_inferior.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_occipital_inferior.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_occipital_inferior.nii.gz

# # Middle occipital gyrus
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11119 -uthr 11119 -bin $MASK_DIR/${SUBJ_DIR}_left_occipital_middle.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12119 -uthr 12119 -bin $MASK_DIR/${SUBJ_DIR}_right_occipital_middle.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_left_occipital_middle.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_occipital_middle.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_occipital_middle.nii.gz

# # Superior occipital gyrus
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11120 -uthr 11120 -bin $MASK_DIR/${SUBJ_DIR}_left_occipital_superior.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12120 -uthr 12120 -bin $MASK_DIR/${SUBJ_DIR}_right_occipital_superior.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_left_occipital_superior.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_occipital_superior.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_occipital_superior.nii.gz

# Olfactory sulcus
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11164 -uthr 11164 -bin $MASK_DIR/${SUBJ_DIR}_left_olfactory.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12164 -uthr 12164 -bin $MASK_DIR/${SUBJ_DIR}_right_olfactory.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_left_olfactory.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_olfactory.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_olfactory.nii.gz

# # Orbital frontal
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11113 -uthr 11113 -bin $MASK_DIR/${SUBJ_DIR}_left_frontal_inf-orbital.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12113 -uthr 12113 -bin $MASK_DIR/${SUBJ_DIR}_right_frontal_inf-orbital.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_left_frontal_inf-orbital.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_frontal_inf-orbital.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_frontal_inf-orbital.nii.gz

# # G_Ins_lg_and_S_cent_ins
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11117 -uthr 11117 -bin $MASK_DIR/${SUBJ_DIR}_left_long-insular.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12117 -uthr 12117 -bin $MASK_DIR/${SUBJ_DIR}_right_long-insular.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_left_long-insular.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_long-insular.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_long-insular.nii.gz

# # G_insular_short
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 11118 -uthr 11118 -bin $MASK_DIR/${SUBJ_DIR}_left_short-insular.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_aparc.a2009s+aseg_CONVERTED2BOLD.nii.gz -thr 12118 -uthr 12118 -bin $MASK_DIR/${SUBJ_DIR}_right_short-insular.nii.gz
# fslmaths $MASK_DIR/${SUBJ_DIR}_left_short-insular.nii.gz -add $MASK_DIR/${SUBJ_DIR}_right_short-insular.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_short-insular.nii.gz

# fslmaths $MASK_DIR/${SUBJ_DIR}_bilateral_long-insular.nii.gz -add $MASK_DIR/${SUBJ_DIR}_bilateral_short-insular.nii.gz -bin $MASK_DIR/${SUBJ_DIR}_bilateral_insula.nii.gz
