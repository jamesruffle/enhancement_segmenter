#!/usr/bin/env python3
"""
Simplified inference script for the Enhancement Segmenter model.

This script provides a user-friendly wrapper around nnUNetv2_predict for
predicting brain tumour enhancement from non-contrast MRI sequences.

Usage:
    python predict.py -i /path/to/input -o /path/to/output

Input format:
    The input folder should contain NIFTI files named as:
    - {subject_id}_0000.nii.gz  (FLAIR)
    - {subject_id}_0001.nii.gz  (T1)
    - {subject_id}_0002.nii.gz  (T2)

Output:
    Segmentation masks with labels:
    - 0: Background
    - 1: Brain parenchyma
    - 2: Non-enhancing abnormality
    - 3: Predicted enhancing tumour
"""

import argparse
import os
import subprocess
import sys


# Model configuration
DATASET_NAME = "Dataset003_enhance_and_abnormality_batchconfig"
TRAINER = "nnUNetTrainer"
CONFIGURATION = "3d_fullres"
PLANS = "nnUNetResEncUNetPlans_80G"
FOLDS = "0 1 2 3 4"


def check_environment():
    """Check that nnU-Net environment variables are set."""
    required_vars = ["nnUNet_results"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        print("Error: The following environment variables must be set:")
        for var in missing:
            print(f"  - {var}")
        print("\nExample:")
        print('  export nnUNet_results="/path/to/nnUNet_results"')
        sys.exit(1)

    # Check that model exists
    model_path = os.path.join(
        os.environ["nnUNet_results"],
        DATASET_NAME,
        f"{TRAINER}__{PLANS}__{CONFIGURATION}"
    )

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("\nPlease download the pretrained weights from Zenodo and place them in:")
        print(f"  {model_path}")
        sys.exit(1)


def check_input_format(input_dir):
    """Check that input files are in the correct format."""
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    files = os.listdir(input_dir)
    nifti_files = [f for f in files if f.endswith('.nii.gz')]

    if not nifti_files:
        print(f"Error: No .nii.gz files found in {input_dir}")
        sys.exit(1)

    # Check for proper naming convention
    subjects = set()
    for f in nifti_files:
        if '_0000.nii.gz' in f or '_0001.nii.gz' in f or '_0002.nii.gz' in f:
            subject_id = f.rsplit('_', 1)[0]
            subjects.add(subject_id)

    if not subjects:
        print("Warning: Files may not be in the correct format.")
        print("Expected format: {subject_id}_0000.nii.gz, {subject_id}_0001.nii.gz, {subject_id}_0002.nii.gz")
        print("  - _0000: FLAIR")
        print("  - _0001: T1")
        print("  - _0002: T2")
    else:
        print(f"Found {len(subjects)} subject(s) to process")

    return True


def run_prediction(input_dir, output_dir, device="cuda", save_probabilities=False,
                   num_processes_preprocessing=3, num_processes_segmentation=3):
    """Run nnU-Net prediction."""

    cmd = [
        "nnUNetv2_predict",
        "-d", DATASET_NAME,
        "-i", input_dir,
        "-o", output_dir,
        "-f", *FOLDS.split(),
        "-tr", TRAINER,
        "-c", CONFIGURATION,
        "-p", PLANS,
        "-device", device,
        "-npp", str(num_processes_preprocessing),
        "-nps", str(num_processes_segmentation),
    ]

    if save_probabilities:
        cmd.append("--save_probabilities")

    print("Running nnU-Net prediction...")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
        print(f"\nPrediction complete! Results saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\nError running prediction: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: nnUNetv2_predict not found. Please install nnU-Net v2:")
        print("  pip install nnunetv2")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Predict brain tumour enhancement from non-contrast MRI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python predict.py -i /path/to/input -o /path/to/output

  # Save probability maps
  python predict.py -i /path/to/input -o /path/to/output --save_probabilities

  # Use CPU instead of GPU
  python predict.py -i /path/to/input -o /path/to/output --device cpu

Input format:
  Place your MRI sequences in the input folder with the following naming:
    subject001_0000.nii.gz  (FLAIR)
    subject001_0001.nii.gz  (T1 - non-contrast)
    subject001_0002.nii.gz  (T2)

Output labels:
  0: Background
  1: Brain parenchyma
  2: Non-enhancing abnormality
  3: Predicted enhancing tumour
"""
    )

    parser.add_argument("-i", "--input", required=True,
                        help="Input folder containing NIFTI files")
    parser.add_argument("-o", "--output", required=True,
                        help="Output folder for predictions")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for inference (default: cuda)")
    parser.add_argument("--save_probabilities", action="store_true",
                        help="Save softmax probability maps")
    parser.add_argument("--npp", type=int, default=3,
                        help="Number of processes for preprocessing (default: 3)")
    parser.add_argument("--nps", type=int, default=3,
                        help="Number of processes for segmentation export (default: 3)")

    args = parser.parse_args()

    # Validate environment and inputs
    check_environment()
    check_input_format(args.input)

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Run prediction
    run_prediction(
        input_dir=args.input,
        output_dir=args.output,
        device=args.device,
        save_probabilities=args.save_probabilities,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation=args.nps
    )


if __name__ == "__main__":
    main()
