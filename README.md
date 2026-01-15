# Enhancement Segmenter

[![arXiv](https://img.shields.io/badge/arXiv-2508.16650-b31b1b.svg)](https://arxiv.org/abs/2508.16650)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Deep learning model for predicting brain tumour contrast enhancement from non-contrast MRI sequences.

## Overview

This repository contains the code and model weights for the paper:

**"Predicting brain tumour enhancement from non-contrast MR imaging with artificial intelligence"**

Ruffle JK, Mohinta S, Pombo G, Biswas A, Campbell A, Davagnanam I, Doig D, Hamman A, Hyare H, Jabeen F, Lim E, Mallon D, Owen S, Wilkinson S, Brandner S, Nachev P.

[arXiv:2508.16650](https://arxiv.org/abs/2508.16650)

### Key Results

- **83%** balanced accuracy
- **91.5%** sensitivity
- **74.4%** specificity
- Trained on **11,089** brain MRI studies
- Validated across glioma, meningioma, metastases, and post-resection cases in both adult and paediatric populations

## Model Architecture

- **Framework**: [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet)
- **Architecture**: ResidualEncoderUNet (3D fullres)
- **Input**: 3 non-contrast MRI sequences (FLAIR, T1, T2)
- **Output**: 4-class segmentation
  - 0: Background
  - 1: Brain parenchyma
  - 2: Abnormality (non-enhancing lesion)
  - 3: Enhancing tumour (predicted)
- **Training**: 5-fold cross-validation
- **Best Model Dice Score**: 0.7859

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU with 8GB+ VRAM (for inference)
- ~48GB GPU VRAM recommended for training

### Install nnU-Net v2

```bash
pip install nnunetv2
```

### Set Environment Variables

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## Pretrained Model Weights

Download the pretrained model weights from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> **Note**: Update the Zenodo DOI badge once the upload is complete.

### Directory Structure

After downloading, place the model files in the following structure:

```
$nnUNet_results/
└── Dataset003_enhance_and_abnormality_batchconfig/
    └── nnUNetTrainer__nnUNetResEncUNetPlans_80G__3d_fullres/
        ├── fold_0/
        │   └── checkpoint_best.pth
        ├── fold_1/
        │   └── checkpoint_best.pth
        ├── fold_2/
        │   └── checkpoint_best.pth
        ├── fold_3/
        │   └── checkpoint_best.pth
        ├── fold_4/
        │   └── checkpoint_best.pth
        ├── dataset.json
        └── plans.json
```

## Usage

### Input Data Format

Prepare your input data in the nnU-Net format:

```
input_folder/
├── subject001_0000.nii.gz  # FLAIR
├── subject001_0001.nii.gz  # T1
├── subject001_0002.nii.gz  # T2
├── subject002_0000.nii.gz  # FLAIR
├── subject002_0001.nii.gz  # T1
├── subject002_0002.nii.gz  # T2
└── ...
```

**Channel mapping:**
- `_0000.nii.gz`: FLAIR sequence
- `_0001.nii.gz`: T1-weighted sequence (non-contrast)
- `_0002.nii.gz`: T2-weighted sequence

### Running Inference

```bash
nnUNetv2_predict \
    -d Dataset003_enhance_and_abnormality_batchconfig \
    -i /path/to/input_folder \
    -o /path/to/output_folder \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -c 3d_fullres \
    -p nnUNetResEncUNetPlans_80G
```

### Output Interpretation

The model outputs a segmentation mask with the following labels:

| Label | Description |
|-------|-------------|
| 0 | Background |
| 1 | Brain parenchyma |
| 2 | Non-enhancing abnormality |
| 3 | Predicted enhancing tumour |

## Training

### Data Preparation

1. **Organise your raw data** with the following structure:
   - 4D NIFTI files containing [FLAIR, T1, T1CE, T2] sequences
   - Enhancement masks (ground truth segmentations)

2. **Split sequences** using FSL tools:

```bash
# Example: Split 4D sequences into individual 3D volumes
fslsplit input_4d.nii.gz output_prefix_ -t

# Rename channels to match nnUNet format
# _0000 = FLAIR, _0001 = T1, _0002 = T2
# Note: T1CE (channel 3) is excluded as we predict enhancement from non-contrast only
```

See `split_sequences_abnormality_class.sh` for a complete data preparation example.

### Dataset Configuration

Create a `dataset.json` file:

```json
{
    "channel_names": {
        "0": "FLAIR",
        "1": "T1",
        "2": "T2"
    },
    "labels": {
        "background": 0,
        "brain": 1,
        "abnormality": 2,
        "ET": 3
    },
    "numTraining": 9980,
    "file_ending": ".nii.gz"
}
```

### Plan and Preprocess

```bash
# Plan with ResEncL planner and 48GB GPU memory target
nnUNetv2_plan_and_preprocess -d 003 -pl nnUNetPlannerResEncL -np 32 --verify_dataset_integrity

# Generate experiment plan with custom memory target
nnUNetv2_plan_experiment -d 3 -pl nnUNetPlannerResEncL -gpu_memory_target 48 -overwrite_plans_name nnUNetResEncUNetPlans_48G
```

### Training

Train all 5 folds (multi-GPU example):

```bash
# Train folds 0-2 in parallel on 3 GPUs
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 0 --npz -p nnUNetResEncUNetPlans_48G &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 003 3d_fullres 1 --npz -p nnUNetResEncUNetPlans_48G &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 003 3d_fullres 2 --npz -p nnUNetResEncUNetPlans_48G &
wait

# Train folds 3-4
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 3 --npz -p nnUNetResEncUNetPlans_48G &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 003 3d_fullres 4 --npz -p nnUNetResEncUNetPlans_48G &
wait
```

See `train_nnunet_abnormality_class_bigger_batch_L_mod.sh` for the exact training configuration used in the paper.

### Find Best Configuration

```bash
nnUNetv2_find_best_configuration 003 -p nnUNetResEncUNetPlans_48G
```

## Repository Structure

```
enhancement_segmenter/
├── README.md                           # This file
├── LICENSE                             # Apache 2.0 license
├── requirements.txt                    # Python dependencies
├── predict.py                          # Simplified inference wrapper
├── train.sh                            # Training script with exact parameters
├── split_sequences_abnormality_class.sh  # Data preparation example
└── ZENODO_UPLOAD.md                    # Instructions for model weight upload
```

## Citation

If you use this code or model in your research, please cite:

```bibtex
@article{ruffle2025predicting,
  title={Predicting brain tumour enhancement from non-contrast MR imaging with artificial intelligence},
  author={Ruffle, James K and Mohinta, Samia and Pombo, Guilherme and Biswas, Asthik and Campbell, Alan and Davagnanam, Indran and Doig, David and Hamman, Ahmed and Hyare, Harpreet and Jabeen, Farrah and Lim, Emma and Mallon, Dermot and Owen, Stephanie and Wilkinson, Sophie and Brandner, Sebastian and Nachev, Parashkev},
  journal={arXiv preprint arXiv:2508.16650},
  year={2025}
}
```

Please also cite nnU-Net:

```bibtex
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## Funding

This work was supported by:

- [European Society of Radiology (ESR)](https://www.myesr.org/) / [European Institute for Biomedical Imaging Research (EIBIR)](https://www.eibir.org/) Seed Grant
- [Medical Research Council](https://www.ukri.org/councils/mrc/) (MR/X00046X/1)
- [British Society of Neuroradiology](https://www.bsnr.org.uk/)
- [National Brain Appeal](https://www.nationalbrainappeal.org/)
- [Wellcome Trust](https://wellcome.org/) (213038/Z/18/Z)
- [UCLH NIHR Biomedical Research Centre](https://www.uclhospitals.brc.nihr.ac.uk/)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this code, please open an issue on GitHub or contact the corresponding author.
