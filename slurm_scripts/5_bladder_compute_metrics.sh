#!/bin/bash
#SBATCH --job-name=segmamba_metrics
#SBATCH --partition=small-long
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/metrics_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/metrics_%j.err
#SBATCH --cpus-per-task=21            # 21 CPU cores

APPTAINER=/software/apptainer/bin/apptainer

################ host-side paths you MAY tweak ################
REPO=/sharedscratch/br61/thesis                       # repo root
DATA_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/ # GT folder
PRED_DIR=/sharedscratch/br61/thesis/predictions/first_try_pred_on_center1
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif
################################################################

"$APPTAINER" exec --nv --cleanenv \
    --bind ${REPO}:/workspace \
    --bind ${DATA_DIR}:/workspace/data/FedBCa_clean \
    --bind $(dirname ${PRED_DIR}):/workspace/predictions \
    "$IMG" \
    bash -lc "set -euo pipefail
              pip install --quiet --no-cache-dir acvl-utils>=0.2.1 \
                                                   medpy SimpleITK \
                                                   scikit-image scikit-learn tqdm nibabel

              python 5_compute_metrics.py \
                     --pred_dir /workspace/predictions/first_try_pred_on_center1/first_try_pred_on_center1 \
                     --gt_dir   /workspace/data/FedBCa_clean/raw_center1"