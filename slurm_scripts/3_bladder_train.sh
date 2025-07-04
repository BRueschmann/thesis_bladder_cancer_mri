#!/bin/bash
#SBATCH --job-name=segmamba_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu01
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/train_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/train_%j.err

APPTAINER=/software/apptainer/bin/apptainer

# Paths
REPO=/sharedscratch/br61/thesis/MVA_SegMamba
DATA=/sharedscratch/br61/thesis/data/FedBCa_clean/no_center1_processed
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif

"$APPTAINER" exec --nv --cleanenv \
        --bind ${REPO}:/workspace \
        --bind ${DATA}:/workspace/data/FedBCa_clean/no_center1_processed \
        --bind /sharedscratch:/sharedscratch \
        "$IMG" \
        bash -lc 'set -euxo pipefail
                  pip install --no-cache-dir --quiet "acvl-utils>=0.2.1"
                  python /workspace/3_bladder_train.py \
                        --data_dir  /workspace/data/FedBCa_clean/no_center1_processed \
                        --logdir    /workspace/logs/segmamba_no_center1 \
                        --max_epoch 1000 \
                        --batch_size 2 \
                        --val_every 2 \
                        --device    cuda:0 \
                        --roi       128 128 128'