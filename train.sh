#!/bin/bash
#
# Training script for Enhancement Segmenter
#
# This script trains the nnU-Net model for predicting brain tumour enhancement
# from non-contrast MRI sequences (FLAIR, T1, T2).
#
# Prerequisites:
#   1. Install nnU-Net v2: pip install nnunetv2
#   2. Set environment variables:
#      export nnUNet_raw="/path/to/nnUNet_raw"
#      export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
#      export nnUNet_results="/path/to/nnUNet_results"
#   3. Prepare dataset in nnU-Net format (see split_sequences_abnormality_class.sh)
#
# Dataset structure:
#   $nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/
#   ├── dataset.json
#   ├── imagesTr/
#   │   ├── subject001_0000.nii.gz  (FLAIR)
#   │   ├── subject001_0001.nii.gz  (T1)
#   │   ├── subject001_0002.nii.gz  (T2)
#   │   └── ...
#   └── labelsTr/
#       ├── subject001.nii.gz
#       └── ...
#
# Usage:
#   ./train.sh
#
# Configuration:
#   - Dataset: 003 (Dataset003_enhance_and_abnormality_batchconfig)
#   - Planner: nnUNetPlannerResEncL
#   - GPU memory target: 48GB (adjust for your hardware)
#   - Configuration: 3d_fullres only
#   - Folds: 5-fold cross-validation (0-4)
#
# Author: James K Ruffle
# Paper: https://arxiv.org/abs/2508.16650
#

set -e  # Exit on error

# Configuration
DATASET_ID=003
PLANS_NAME="nnUNetResEncUNetPlans_48G"
GPU_MEMORY_TARGET=48  # GB - adjust based on your GPU

echo "=============================================="
echo "Enhancement Segmenter Training Script"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Dataset ID: $DATASET_ID"
echo "  Plans: $PLANS_NAME"
echo "  GPU Memory Target: ${GPU_MEMORY_TARGET}GB"
echo ""

# Check environment variables
if [ -z "$nnUNet_raw" ] || [ -z "$nnUNet_preprocessed" ] || [ -z "$nnUNet_results" ]; then
    echo "Error: nnU-Net environment variables not set!"
    echo "Please set:"
    echo "  export nnUNet_raw=\"/path/to/nnUNet_raw\""
    echo "  export nnUNet_preprocessed=\"/path/to/nnUNet_preprocessed\""
    echo "  export nnUNet_results=\"/path/to/nnUNet_results\""
    exit 1
fi

# Step 1: Plan and preprocess
echo "Step 1: Planning and preprocessing..."
echo "----------------------------------------------"
nnUNetv2_plan_and_preprocess -d $DATASET_ID -pl nnUNetPlannerResEncL -np 32 --verify_dataset_integrity

# Step 2: Generate custom plans with GPU memory target
echo ""
echo "Step 2: Generating experiment plan with ${GPU_MEMORY_TARGET}GB target..."
echo "----------------------------------------------"
nnUNetv2_plan_experiment -d $DATASET_ID -pl nnUNetPlannerResEncL -gpu_memory_target $GPU_MEMORY_TARGET -overwrite_plans_name $PLANS_NAME

# Step 3: Train all 5 folds
# Adjust CUDA_VISIBLE_DEVICES based on your available GPUs
echo ""
echo "Step 3: Training 5-fold cross-validation..."
echo "----------------------------------------------"
echo "Note: Modify CUDA_VISIBLE_DEVICES below based on your available GPUs"
echo ""

# Train folds 0, 1, 2 in parallel (requires 3 GPUs)
echo "Training folds 0, 1, 2..."
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID 3d_fullres 0 --npz -p $PLANS_NAME &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_ID 3d_fullres 1 --npz -p $PLANS_NAME &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train $DATASET_ID 3d_fullres 2 --npz -p $PLANS_NAME &
wait

# Train folds 3, 4
echo "Training folds 3, 4..."
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train $DATASET_ID 3d_fullres 3 --npz -p $PLANS_NAME &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train $DATASET_ID 3d_fullres 4 --npz -p $PLANS_NAME &
wait

# Step 4: Find best configuration
echo ""
echo "Step 4: Finding best configuration..."
echo "----------------------------------------------"
nnUNetv2_find_best_configuration $DATASET_ID -p $PLANS_NAME

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "To run inference, use:"
echo "  nnUNetv2_predict -d Dataset003_enhance_and_abnormality_batchconfig \\"
echo "    -i INPUT_FOLDER -o OUTPUT_FOLDER \\"
echo "    -f 0 1 2 3 4 -tr nnUNetTrainer \\"
echo "    -c 3d_fullres -p $PLANS_NAME"
