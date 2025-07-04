#!/bin/bash
#SBATCH --job-name=segmamba_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/run_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/run_%j.err

################# 1. modules ########################
source /etc/profile.d/modules.sh
module use /gpfs01/software/modules/nvidia
module use /gpfs01/software/modules/apps
# module load cuda/12.6.2 apptainer

################# 2. paths ##########################
CUDA_HOME=/software/cuda/12.6.2
REPO=/sharedscratch/br61/thesis/MVA_SegMamba
DATA=/sharedscratch/br61/thesis/data/FedBCa_clean
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif   # ← NEW image

################# 3. run container ##################
apptainer exec --nv --cleanenv \
    --bind ${REPO}:/workspace \
    --bind ${DATA}:/workspace/data/FedBCa_clean \
    "${IMG}" \
    bash -lc "set -euo pipefail
              pip install --no-cache-dir --quiet 'acvl-utils>=0.2.1'
              unset LMOD_SYSTEM_DEFAULT_MODULES MODULESHOME LMOD_CMD
              python /workspace/2_preprocessing_mri.py"
