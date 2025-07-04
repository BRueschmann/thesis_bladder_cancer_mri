#!/bin/bash
#SBATCH --job-name=bladder_experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu01
#SBATCH --time=30:00:00          # 24 h train + 3 h predict + buffer
#SBATCH --mem=40G
#SBATCH --chdir=/sharedscratch/br61/thesis/slurm_scripts
#SBATCH --output=/sharedscratch/br61/thesis/MVA_SegMamba/logs/exp_%j.out
#SBATCH --error=/sharedscratch/br61/thesis/MVA_SegMamba/logs/exp_%j.err

APPTAINER=/software/apptainer/bin/apptainer

########## host-side paths you MAY tweak ######################
REPO=/sharedscratch/br61/thesis/MVA_SegMamba
TRAIN_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/no_center1_processed
TEST_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/center1_processed
GT_DIR=/sharedscratch/br61/thesis/data/FedBCa_clean/raw_center1
IMG=/sharedscratch/br61/containers/segmamba_wheels.sif
################################################################

# Create experiment scaffold (one folder per SLURM job id)
EXP_ROOT=/sharedscratch/br61/thesis/Experiments/Experiment_${SLURM_JOB_ID}
MODEL_DIR=${EXP_ROOT}/model
PRED_DIR=${EXP_ROOT}/predictions
METRICS_DIR=${EXP_ROOT}/metrics
mkdir -p "${MODEL_DIR}" "${PRED_DIR}" "${METRICS_DIR}"

"$APPTAINER" exec --nv --cleanenv \
    --bind ${REPO}:/workspace \
    --bind ${TRAIN_DIR}:/workspace/data/train \
    --bind ${TEST_DIR}:/workspace/data/test \
    --bind ${GT_DIR}:/workspace/data/gt \
    --bind ${EXP_ROOT}:/workspace/experiment \
    "$IMG" \
    bash -lc 'set -euxo pipefail

    ############## 1) TRAIN ####################################
    pip install --quiet --no-cache-dir "acvl-utils>=0.2.1"

    python /workspace/3_bladder_train.py \
          --data_dir  /workspace/data/train \
          --logdir    /workspace/experiment/model \
          --max_epoch 1000 \
          --batch_size 2 \
          --val_every 2 \
          --device    cuda:0 \
          --roi       128 128 128

    # grab the best checkpoint (created by the train script)
    CKPT=$(ls -1 /workspace/experiment/model/best_model_*.pt | sort | tail -n1)

    ############## 2) PREDICT ##################################
    python /workspace/4_bladder_predict.py \
          --model_path "${CKPT}" \
          --save_dir   /workspace/experiment/predictions \
          --data_dir   /workspace/data/test

    ############## 3) METRICS ##################################
    pip install --quiet --no-cache-dir medpy SimpleITK scikit-image scikit-learn tqdm nibabel

    python /workspace/5_compute_metrics.py \
          --pred_dir /workspace/experiment/predictions \
          --gt_dir   /workspace/data/gt \
          --out_dir  /workspace/experiment/metrics
    '
