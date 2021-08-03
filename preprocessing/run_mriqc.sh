#!/bin/bash

source globals.sh

# singularity run --cleanenv \
#     --bind $bids_dir:/home \
#     /jukebox/hasson/singularity/mriqc/mriqc-v0.10.4.sqsh \
#     --participant-label sub-$1 \
#     --correct-slice-timing --no-sub \
#     --nprocs 8 -w /home/derivatives/work \
#     /home /home/derivatives/mriqc participant

singularity run --cleanenv \
    --bind $bids_dir:/bids \
    --bind $scratch_dir:/scratch \
    --bind /usr/people \
    /jukebox/hasson/singularity/mriqc/mriqc-v0.15.1.simg \
    --participant-label $1 \
    --correct-slice-timing --no-sub \
    --nprocs 8 -w /scratch \
    /bids /bids/derivatives/mriqc participant
