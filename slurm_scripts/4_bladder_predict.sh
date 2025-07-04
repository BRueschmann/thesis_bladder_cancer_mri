#!/bin/bash
#SBATCH --job-name=segmamba_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu01
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/predict_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/predict_%j.err

# Use this script with already pre-processed test data!

APPTAINER=/software/apptainer/bin/apptainer

# paths
REPO=/sharedscratch/br61/thesis/MVA_SegMamba
RAW_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/raw_center1
CKPT=/sharedscratch/br61/thesis/trained_models/first_model_no_center1.pt
OUT_DIR=/sharedscratch/br61/thesis/predictions/first_try_pred_on_center1
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif

"$APPTAINER" exec --nv --cleanenv \
        --bind ${REPO}:/workspace \
        --bind ${RAW_DIR}:/workspace/data \
        --bind $(dirname ${CKPT}):/workspace/trained_models \
        --bind ${OUT_DIR}:/workspace/predictions \
        "$IMG" \
        bash -lc 'set -euxo pipefail
                  mkdir -p /workspace/predictions/first_try_pred_on_center1
                  python /workspace/4_bladder_predict.py \
                        --model_path /workspace/trained_models/first_model_no_center1.pt \
                        --save_dir   /workspace/predictions/first_try_pred_on_center1 \
                        --data_dir   /workspace/data'