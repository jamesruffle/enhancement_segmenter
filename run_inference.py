#!/usr/bin/env python3
"""
Run inference on the full validation set and save the output as NIFTI files.

Usage:
    python run_inference.py --model_path /path/to/model/directory --model_type segresnet

For ablation study:
    python run_inference.py --model_path /path/to/model --ablation_study --ablation_output /path/to/output

This script loads a trained model (SegResNet or SwinUNETR) and runs inference on 
the validation set, saving both the raw predictions and binary masks.
When using ablation_study mode, it runs inference with different combinations
of input modalities (FLAIR, T1, T2) to evaluate their individual contributions.
"""

import os
import sys
import glob
import json
import time
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet, SwinUNETR
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ToTensord, SignalFillEmptyd,
    AddCoordinateChannelsd, NormalizeIntensityd, ScaleIntensityd, Compose
)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Enhancement Inference")

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on validation set and save NIFTIs")
    
    # Required parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory containing the model checkpoint and data_split.json")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="segresnet", choices=["segresnet", "swinunetr"],
                        help="Type of model to use: 'segresnet' or 'swinunetr'")
    parser.add_argument("--model_filename", type=str, default="best_model.pth",
                        help="Filename of the model checkpoint in the model_path directory")
    
    # Data parameters
    parser.add_argument("--sequences_dir", type=str, 
                        default="/home/jruffle/Documents/seq-synth/data/sequences_merged",
                        help="Directory containing the sequence NIFTI files")
    parser.add_argument("--segmentations_dir", type=str, 
                        default="/home/jruffle/Documents/seq-synth/data/enhancement_masks",
                        help="Directory containing the ground truth segmentation NIFTI files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the output NIFTI files. If not provided, will use model_path/best_predictions")
    parser.add_argument("--save_probabilities", action="store_true",
                        help="Save the raw prediction probabilities in addition to binary masks")
    parser.add_argument("--xdim", type=int, default=128, help="X dimension of input tensors")
    parser.add_argument("--ydim", type=int, default=128, help="Y dimension of input tensors")
    parser.add_argument("--zdim", type=int, default=128, help="Z dimension of input tensors")
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--sw_batch_size", type=int, default=16,
                        help="Batch size for sliding window inference")
    parser.add_argument("--sw_overlap", type=float, default=0.5,
                        help="Overlap factor for sliding window inference")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for inference ('cuda' or 'cpu'). If not specified, will use cuda if available.")
    parser.add_argument("--metrics_only", action="store_true",
                        help="Calculate metrics only without saving NIFTI files")
    parser.add_argument("--save_validation_metrics", action="store_true",
                        help="Save detailed validation metrics for each sample")
    
    # Ablation study parameters
    parser.add_argument("--ablation_study", action="store_true",
                       help="Run ablation study with different input channel combinations")
    parser.add_argument("--ablation_output", type=str, default=None,
                       help="Output directory for ablation study results")
    parser.add_argument("--multi_gpu", action="store_true",
                       help="Use multiple GPUs for ablation study, distributing combinations across available GPUs")
    parser.add_argument("--multi_gpu_sw_batch_size", type=int, default=4,
                       help="Sliding window batch size for multi-GPU mode (smaller values use less memory)")
    parser.add_argument("--multi_gpu_memory_limit", type=float, default=None,
                       help="Memory limit per GPU in GB. If set, will try to stay below this limit.")
    
    return parser.parse_args()

def setup_transforms():
    """Create a transform pipeline for loading and preprocessing the data."""
    seg_key = ["segmentation"]
    mri_keys = ["sequences"]
    all_keys = seg_key + mri_keys
    cw_application = True  # Channel-wise application
    
    transform_list = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        ToTensord(keys=all_keys),
        SignalFillEmptyd(keys=all_keys),
        AddCoordinateChannelsd(keys=mri_keys, spatial_dims=[0, 1, 2]),
        NormalizeIntensityd(keys=mri_keys, channel_wise=cw_application),
        ScaleIntensityd(keys=mri_keys, channel_wise=cw_application)
    ]
    
    return Compose(transform_list)

def load_model(model_type, model_path, model_filename, xdim, ydim, zdim, device):
    """Load the trained model from the given path."""
    if model_type == "segresnet":
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=3+3,  # FLAIR, T1, T2 + 3 coordinate channels
            out_channels=1,
            dropout_prob=0.1,
        )
    elif model_type == "swinunetr":
        model = SwinUNETR(
            img_size=(xdim, ydim, zdim),
            in_channels=3+3,  # FLAIR, T1, T2 + 3 coordinate channels
            out_channels=1,
            feature_size=48,
            use_checkpoint=False,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            dropout_path_rate=0.1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # Load model checkpoint
    checkpoint_path = os.path.join(model_path, model_filename)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # This is the format saved by the training script
        model_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        # Alternative format with 'model' key
        model_state = checkpoint['model']
    else:
        # Assume it's just the model state dict directly
        model_state = checkpoint
    
    # Handle DDP or DataParallel state dict
    if any(k.startswith('module.') for k in model_state.keys()):
        # Remove 'module.' prefix for DataParallel/DDP models
        from collections import OrderedDict
        new_model_state = OrderedDict()
        for k, v in model_state.items():
            name = k[7:] if k.startswith('module.') else k
            new_model_state[name] = v
        model_state = new_model_state
    
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    return model

def calculate_metrics(y_pred, y, threshold=0.5):
    """Calculate evaluation metrics for predictions against ground truth."""
    # Convert to binary predictions based on threshold
    y_pred_binary = (y_pred > threshold).float()
    
    # Calculate pixel-level metrics
    tp = torch.sum((y_pred_binary == 1) & (y == 1)).float()
    fp = torch.sum((y_pred_binary == 1) & (y == 0)).float()
    tn = torch.sum((y_pred_binary == 0) & (y == 0)).float()
    fn = torch.sum((y_pred_binary == 0) & (y == 1)).float()
    
    # Compute metrics with epsilon to avoid division by zero
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    balanced_acc = 0.5 * (tp/(tp + fn + epsilon) + tn/(tn + fp + epsilon))
    
    metrics = {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'dice': dice.item(),
        'balanced_acc': balanced_acc.item()
    }
    
    return metrics

def save_prediction_as_nifti(image_tensor, metadata, output_dir, filename_prefix="pred", binary=False):
    """
    Save a prediction tensor as a NIFTI file using the original image's metadata.
    
    Args:
        image_tensor (torch.Tensor): The prediction tensor to save [C, H, W, D]
        metadata (dict): Dictionary containing metadata from the original image
        output_dir (str): Directory to save the NIFTI file
        filename_prefix (str): Prefix to add to the filename
        binary (bool): Whether to binarize the predictions (threshold > 0.5)
    
    Returns:
        str: Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy and move channel dimension to the end (required by nibabel)
    image_array = image_tensor.detach().cpu().numpy()
    
    # Apply sigmoid and threshold if needed
    if binary:
        # Apply sigmoid and threshold at 0.5
        image_array = (1 / (1 + np.exp(-image_array))) > 0.5
        image_array = image_array.astype(np.float32)  # Convert binary to float
    
    # Get original filename without path and extension
    orig_filename = os.path.basename(metadata['filename_or_obj'])
    if '.nii' in orig_filename:
        orig_filename = orig_filename.split('.nii')[0]
    
    # Construct output filename
    output_filename = f"{filename_prefix}_{orig_filename}.nii.gz"
    output_path = os.path.join(output_dir, output_filename)
    
    # Get the affine matrix from the original image metadata
    affine = None
    if 'original_affine' in metadata:
        affine = metadata['original_affine']
        # Ensure affine is a numpy array and has shape (4,4)
        if not isinstance(affine, np.ndarray):
            try:
                affine = np.array(affine)
            except:
                logger.warning(f"Could not convert affine to numpy array for {orig_filename}")
                affine = None
        
        # Check affine shape and fix it if it's (1, 4, 4)
        if affine is not None:
            if affine.shape == (1, 4, 4):
                # Remove the extra dimension to get (4, 4)
                affine = affine[0]
                logger.debug(f"Fixed affine shape from (1, 4, 4) to (4, 4) for {orig_filename}")
            elif affine.shape != (4, 4):
                logger.warning(f"Affine has incorrect shape {affine.shape} for {orig_filename}, using identity instead")
                affine = None
    
    # If affine is still None, use identity matrix
    if affine is None:
        logger.warning(f"No valid affine found for {orig_filename}. Using identity.")
        affine = np.eye(4)
    
    # Create NIFTI image and save
    # Need to reshape if there's a channel dimension (nibabel doesn't support channel dimension)
    if image_array.shape[0] == 1:  # If we have a single channel
        image_array = image_array[0]  # Remove channel dimension
    
    nifti_image = nib.Nifti1Image(image_array, affine)
    nib.save(nifti_image, output_path)
    
    return output_path

def create_ablation_masks():
    """
    Create a dictionary of masks for different ablation combinations.
    
    Returns:
        dict: Dictionary of mask combinations where keys are descriptive names and values are
              boolean masks to apply to input channels (FLAIR, T1, T2) where True means keep channel.
    """
    # Define all possible ablation combinations
    # Each mask is for [FLAIR, T1, T2] where True means the channel is kept
    ablation_masks = {
        "FLAIR": [True, False, False],
        "T1": [False, True, False],
        "T2": [False, False, True],
        "FLAIR_T1": [True, True, False],
        "FLAIR_T2": [True, False, True],
        "T1_T2": [False, True, True],
        "FLAIR_T1_T2": [True, True, True],  # This is the full model (no ablation)
    }
    
    return ablation_masks

def apply_channel_mask(input_tensor, mask):
    """
    Apply a boolean mask to specific channels of an input tensor.
    
    Args:
        input_tensor (torch.Tensor): Input tensor with shape [B, C, H, W, D]
        mask (list): Boolean mask list for FLAIR, T1, T2 where True means keep the channel
    
    Returns:
        torch.Tensor: Tensor with masked channels (zeroed out where mask is False)
    """
    # Channel indices: FLAIR=0, T1=1, T2=3 (T1ce=2 is not used in inference)
    channel_indices = {
        "flair": 0,
        "t1": 1,
        "t2": 3
    }
    
    # Copy the input to avoid modifying the original
    masked_tensor = input_tensor.clone()
    
    # Apply mask to each channel
    if not mask[0]:  # FLAIR
        masked_tensor[:, 0:1, ...] = 0.0
    
    if not mask[1]:  # T1
        masked_tensor[:, 1:2, ...] = 0.0
    
    if not mask[2]:  # T2
        masked_tensor[:, 3:4, ...] = 0.0
    
    return masked_tensor

def run_ablation_study(args, model, val_loader, device):
    """
    Run an ablation study by masking different combinations of input channels.
    
    Args:
        args: Command-line arguments
        model: Trained model
        val_loader: DataLoader with validation data
        device: Device for inference
    """
    # Create directory for ablation results
    if args.ablation_output is None:
        args.ablation_output = os.path.join(args.model_path, "ablation_results")
    os.makedirs(args.ablation_output, exist_ok=True)
    
    logger.info(f"Running ablation study. Results will be saved to: {args.ablation_output}")
    
    # Get all ablation mask combinations
    ablation_masks = create_ablation_masks()
    
    # Dictionary to store metrics for each ablation
    ablation_metrics = {}
    
    # For each ablation combination
    for ablation_name, mask in ablation_masks.items():
        logger.info(f"Running ablation: {ablation_name}")
        
        # Create directory for this ablation
        ablation_dir = os.path.join(args.ablation_output, ablation_name)
        os.makedirs(ablation_dir, exist_ok=True)
        
        # List to store metrics for each sample
        all_metrics = []
        
        # Process each batch
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Processing {ablation_name}"):
                # Get input sequences and target segmentation
                sequences = batch_data["sequences"].to(device)
                segmentation = batch_data["segmentation"].to(device)
                
                # Extract channels
                flair = sequences[:, 0:1, ...]
                t1 = sequences[:, 1:2, ...]
                # t1ce = sequences[:, 2:3, ...] # Not used in input
                t2 = sequences[:, 3:4, ...]
                coord_conv = sequences[:, 4:, ...]
                
                # Combine input channels
                input_channels = torch.cat([flair, t1, t2, coord_conv], dim=1)
                
                # Apply channel masking for this ablation
                masked_input = apply_channel_mask(input_channels, mask)
                
                # Perform sliding window inference
                outputs = sliding_window_inference(
                    inputs=masked_input,
                    roi_size=[args.xdim, args.ydim, args.zdim],
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.sw_overlap,
                    mode="gaussian",
                    sigma_scale=0.125,
                    padding_mode="constant",
                    cval=0.0,
                    sw_device=device,
                    device=device
                )
                
                # Apply sigmoid to get probabilities
                outputs_sigmoid = torch.sigmoid(outputs)
                
                # Calculate metrics for each item in the batch
                for i, output in enumerate(decollate_batch(outputs_sigmoid)):
                    metrics = calculate_metrics(output, segmentation[i])
                    
                    # Include filename for reference
                    case_filename = os.path.basename(batch_data["segmentation"].meta["filename_or_obj"][i])
                    case_name = case_filename.split('.nii')[0]
                    metrics["filename"] = case_name
                    all_metrics.append(metrics)
                
                # Save predictions
                for i, (output, output_sigmoid) in enumerate(zip(decollate_batch(outputs), decollate_batch(outputs_sigmoid))):
                    # Get metadata for this sample
                    meta_dict = {
                        'filename_or_obj': batch_data["segmentation"].meta["filename_or_obj"][i],
                        'original_affine': batch_data["segmentation"].meta.get("original_affine", np.eye(4))
                    }
                    
                    # Save binary mask
                    save_prediction_as_nifti(
                        output_sigmoid, 
                        meta_dict, 
                        ablation_dir,
                        filename_prefix="bin",
                        binary=True
                    )
                    
                    # Save probabilities
                    save_prediction_as_nifti(
                        output, 
                        meta_dict, 
                        ablation_dir,
                        filename_prefix="prob",
                        binary=False
                    )
        
        # Calculate average metrics for this ablation
        avg_metrics = {key: sum(item[key] for item in all_metrics) / len(all_metrics) 
                     for key in all_metrics[0] if key != "filename"}
        
        # Store metrics for this ablation
        ablation_metrics[ablation_name] = avg_metrics
        
        # Save detailed metrics for this ablation
        import pandas as pd
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(ablation_dir, f"{ablation_name}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        # Print metrics summary for this ablation
        logger.info(f"\nMetrics for {ablation_name}:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    
    # Save overall ablation summary
    ablation_summary = {
        "ablation_metrics": ablation_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(os.path.join(args.ablation_output, "ablation_results.json"), 'w') as f:
        json.dump(ablation_summary, f, indent=4)
    
    # Create and save summary table
    summary_data = []
    for ablation_name, metrics in ablation_metrics.items():
        row = {"ablation": ablation_name}
        row.update(metrics)
        summary_data.append(row)
    
    import pandas as pd
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.ablation_output, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"\nAblation study completed. Results saved to {args.ablation_output}")
    logger.info(f"Summary saved to {summary_path}")

def process_single_ablation(ablation_name, mask, gpu_id, result_queue, args, model_type, model_path, model_filename, xdim, ydim, zdim, val_files):
    """
    Process a single ablation combination on a specific GPU.
    This function runs in a separate process.
    """
    try:
        logger.info(f"Starting process for ablation {ablation_name} on GPU {gpu_id}")
        # Set the device for this process
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        
        # Clear GPU memory before starting
        torch.cuda.empty_cache()
        
        logger.info(f"Process for ablation {ablation_name}: Using GPU {gpu_id} ({torch.cuda.get_device_name(device)})")
        logger.info(f"GPU {gpu_id} memory before loading model: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB used")
        
        # Create directory for this ablation
        ablation_dir = os.path.join(args.ablation_output, ablation_name)
        os.makedirs(ablation_dir, exist_ok=True)
        
        # Need to create a new model instance for this GPU
        model = load_model(
            model_type=model_type,
            model_path=model_path,
            model_filename=model_filename,
            xdim=xdim,
            ydim=ydim,
            zdim=zdim,
            device=device
        )
        model.eval()  # Ensure model is in evaluation mode
        
        logger.info(f"GPU {gpu_id} memory after loading model: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB used")
        
        # Use smaller batch size and user-specified multi-GPU sliding window batch size
        multi_gpu_batch_size = max(1, args.batch_size // 2)
        multi_gpu_sw_batch_size = args.multi_gpu_sw_batch_size if args.multi_gpu_sw_batch_size is not None else max(1, args.sw_batch_size // 2)
        
        logger.info(f"Using reduced batch size for multi-GPU: batch_size={multi_gpu_batch_size}, sw_batch_size={multi_gpu_sw_batch_size}")
        
        # Create new dataset and dataloader
        transform = setup_transforms()
        val_ds = Dataset(data=val_files, transform=transform)
        val_loader_local = DataLoader(
            val_ds,
            batch_size=multi_gpu_batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // torch.cuda.device_count()),  # Distribute workers
            pin_memory=True
        )
        
        # List to store metrics for each sample
        all_metrics = []
        
        # Process each batch
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_loader_local, desc=f"Processing {ablation_name} on GPU {gpu_id}")):
                try:
                    # Get input sequences and target segmentation
                    sequences = batch_data["sequences"].to(device)
                    segmentation = batch_data["segmentation"].to(device)
                    
                    # Extract channels
                    flair = sequences[:, 0:1, ...]
                    t1 = sequences[:, 1:2, ...]
                    t2 = sequences[:, 3:4, ...]
                    coord_conv = sequences[:, 4:, ...]
                    
                    # Combine input channels
                    input_channels = torch.cat([flair, t1, t2, coord_conv], dim=1)
                    
                    # Apply channel masking for this ablation
                    masked_input = apply_channel_mask(input_channels, mask)
                    
                    # Perform sliding window inference with reduced batch size
                    outputs = sliding_window_inference(
                        inputs=masked_input,
                        roi_size=[args.xdim, args.ydim, args.zdim],
                        sw_batch_size=multi_gpu_sw_batch_size,
                        predictor=model,
                        overlap=args.sw_overlap,
                        mode="gaussian",
                        sigma_scale=0.125,
                        padding_mode="constant",
                        cval=0.0,
                        sw_device=device,
                        device=device
                    )
                    
                    # Apply sigmoid to get probabilities
                    outputs_sigmoid = torch.sigmoid(outputs)
                    
                    # Calculate metrics for each item in the batch
                    for i, output in enumerate(decollate_batch(outputs_sigmoid)):
                        metrics = calculate_metrics(output, segmentation[i])
                        
                        # Include filename for reference
                        case_filename = os.path.basename(batch_data["segmentation"].meta["filename_or_obj"][i])
                        case_name = case_filename.split('.nii')[0]
                        metrics["filename"] = case_name
                        all_metrics.append(metrics)
                    
                    # Save predictions
                    for i, (output, output_sigmoid) in enumerate(zip(decollate_batch(outputs), decollate_batch(outputs_sigmoid))):
                        # Get metadata for this sample
                        meta_dict = {
                            'filename_or_obj': batch_data["segmentation"].meta["filename_or_obj"][i],
                            'original_affine': batch_data["segmentation"].meta.get("original_affine", np.eye(4))
                        }
                        
                        # Save binary mask
                        save_prediction_as_nifti(
                            output_sigmoid, 
                            meta_dict, 
                            ablation_dir,
                            filename_prefix="bin",
                            binary=True
                        )
                        
                        # Save probabilities
                        save_prediction_as_nifti(
                            output, 
                            meta_dict, 
                            ablation_dir,
                            filename_prefix="prob",
                            binary=False
                        )
                    
                    # Explicitly clear some memory
                    del sequences, segmentation, flair, t1, t2, coord_conv, input_channels, outputs, outputs_sigmoid
                    torch.cuda.empty_cache()
                    
                    # Log memory usage periodically
                    if batch_idx % 10 == 0:
                        mem_used = torch.cuda.memory_allocated(device) / 1024**3  # GB
                        logger.info(f"{ablation_name} on GPU {gpu_id}: Batch {batch_idx}, Memory used: {mem_used:.2f} GB")
                
                except torch.cuda.OutOfMemoryError as oom_error:
                    logger.error(f"GPU {gpu_id} ran out of memory on {ablation_name}, batch {batch_idx}! Trying to recover...")
                    torch.cuda.empty_cache()
                    
                    # If we failed with the current sw_batch_size, try with an even smaller one
                    reduced_sw_batch_size = max(1, multi_gpu_sw_batch_size // 2)
                    logger.info(f"Retrying with further reduced sw_batch_size={reduced_sw_batch_size}")
                    
                    # Wait a moment for memory to clear
                    time.sleep(5)
                    
                    try:
                        # Get input again
                        sequences = batch_data["sequences"].to(device)
                        segmentation = batch_data["segmentation"].to(device)
                        
                        # Extract channels
                        flair = sequences[:, 0:1, ...]
                        t1 = sequences[:, 1:2, ...]
                        t2 = sequences[:, 3:4, ...]
                        coord_conv = sequences[:, 4:, ...]
                        
                        # Combine input channels
                        input_channels = torch.cat([flair, t1, t2, coord_conv], dim=1)
                        
                        # Apply channel masking for this ablation
                        masked_input = apply_channel_mask(input_channels, mask)
                        
                        # Perform sliding window inference with further reduced batch size
                        outputs = sliding_window_inference(
                            inputs=masked_input,
                            roi_size=[args.xdim, args.ydim, args.zdim],
                            sw_batch_size=reduced_sw_batch_size,
                            predictor=model,
                            overlap=args.sw_overlap,
                            mode="gaussian",
                            sigma_scale=0.125,
                            padding_mode="constant",
                            cval=0.0,
                            sw_device=device,
                            device=device
                        )
                        
                        # Continue with processing...
                        # Apply sigmoid to get probabilities
                        outputs_sigmoid = torch.sigmoid(outputs)
                        
                        # Update sw_batch_size for future iterations if this worked
                        multi_gpu_sw_batch_size = reduced_sw_batch_size
                        logger.info(f"Recovery successful. Continuing with sw_batch_size={multi_gpu_sw_batch_size}")
                        
                    except Exception as retry_error:
                        logger.error(f"Recovery failed for {ablation_name} on GPU {gpu_id}, batch {batch_idx}: {retry_error}")
                        logger.error("Skipping this batch and continuing...")
                        continue
        
        # Calculate average metrics for this ablation
        if all_metrics:
            avg_metrics = {key: sum(item[key] for item in all_metrics) / len(all_metrics) 
                         for key in all_metrics[0] if key != "filename"}
        else:
            logger.warning(f"No metrics collected for {ablation_name} on GPU {gpu_id}")
            avg_metrics = {"error": "No metrics collected"}
        
        # Save detailed metrics for this ablation
        if all_metrics:
            import pandas as pd
            metrics_df = pd.DataFrame(all_metrics)
            metrics_path = os.path.join(ablation_dir, f"{ablation_name}_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
        
        # Print metrics summary for this ablation
        logger.info(f"\nMetrics for {ablation_name} (GPU {gpu_id}):")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Put results in queue
        result_queue.put((ablation_name, avg_metrics))
        
        # Clean up GPU memory before exiting
        del model
        torch.cuda.empty_cache()
        logger.info(f"GPU {gpu_id} memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB used")
        
        logger.info(f"Completed ablation {ablation_name} on GPU {gpu_id}")
        
    except Exception as e:
        import traceback
        logger.error(f"Error in process for ablation {ablation_name} on GPU {gpu_id}: {e}")
        logger.error(traceback.format_exc())
        result_queue.put((ablation_name, {"error": str(e)}))

def run_ablation_study_multi_gpu(args, model, val_loader, base_device):
    """
    Run an ablation study by masking different combinations of input channels,
    distributing tasks across multiple available GPUs using multiprocessing for true parallelism.
    
    Args:
        args: Command-line arguments
        model: Trained model
        val_loader: DataLoader with validation data
        base_device: Base device for inference (ignored when multiple GPUs used)
    """
    # Create directory for ablation results
    if args.ablation_output is None:
        args.ablation_output = os.path.join(args.model_path, "ablation_results")
    os.makedirs(args.ablation_output, exist_ok=True)
    
    # Get all available GPUs
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to single device ablation study.")
        run_ablation_study(args, model, val_loader, base_device)
        return
    
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        logger.warning("Only one GPU available. Falling back to single device ablation study.")
        run_ablation_study(args, model, val_loader, base_device)
        return
    
    logger.info(f"Running multi-GPU ablation study with {num_gpus} GPUs.")
    logger.info(f"Results will be saved to: {args.ablation_output}")
    
    # Get all ablation mask combinations
    ablation_masks = create_ablation_masks()
    
    # Distribute ablation tasks across GPUs
    ablation_items = list(ablation_masks.items())
    
    # Calculate which GPU each ablation runs on
    gpu_assignments = {}
    for i, (ablation_name, _) in enumerate(ablation_items):
        gpu_id = i % num_gpus
        gpu_assignments[ablation_name] = gpu_id
        logger.info(f"Assigned ablation {ablation_name} to GPU {gpu_id}")
    
    # Dictionary to store metrics for each ablation
    ablation_metrics = {}
    
    # Use multiprocessing to run ablations in parallel
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # Use 'spawn' for better compatibility with CUDA
    
    # Create a manager to share data between processes
    manager = mp.Manager()
    result_queue = manager.Queue()
    
    # Group ablations by GPU
    processes = []
    
    # Create and start processes for each ablation
    for ablation_name, mask in ablation_masks.items():
        gpu_id = gpu_assignments[ablation_name]
        p = mp.Process(
            target=process_single_ablation,
            args=(
                ablation_name, 
                mask, 
                gpu_id, 
                result_queue,
                args,
                args.model_type,
                args.model_path,
                args.model_filename,
                args.xdim,
                args.ydim,
                args.zdim,
                val_loader.dataset.data
            )
        )
        processes.append(p)
        p.start()
        logger.info(f"Started process for ablation {ablation_name} on GPU {gpu_id}")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results from the queue
    while not result_queue.empty():
        name, metrics = result_queue.get()
        ablation_metrics[name] = metrics
    
    # Save overall ablation summary
    ablation_summary = {
        "ablation_metrics": ablation_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(os.path.join(args.ablation_output, "ablation_results.json"), 'w') as f:
        json.dump(ablation_summary, f, indent=4)
    
    # Create and save summary table
    summary_data = []
    for ablation_name, metrics in ablation_metrics.items():
        row = {"ablation": ablation_name}
        row.update(metrics)
        summary_data.append(row)
    
    import pandas as pd
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.ablation_output, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"\nMulti-GPU ablation study completed. Results saved to {args.ablation_output}")
    logger.info(f"Summary saved to {summary_path}")

def run_inference():
    """Run inference on the validation set and save the results."""
    args = parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "best_predictions")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output predictions will be saved to: {args.output_dir}")
    
    # Handle ablation output directory
    if args.ablation_study and args.ablation_output is None:
        args.ablation_output = os.path.join(args.model_path, "ablation_results")
    
    # Load data_split.json to get validation files
    split_path = os.path.join(args.model_path, "data_split.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"data_split.json not found at {split_path}")
    
    with open(split_path, 'r') as f:
        data_split = json.load(f)
    
    val_files = data_split.get('val_files', [])
    if not val_files:
        raise ValueError("No validation files found in data_split.json")
    
    logger.info(f"Found {len(val_files)} validation files")
    
    # Fix paths in val_files if they are relative
    for file_data in val_files:
        # Check if sequences path is relative (doesn't start with '/')
        if 'sequences' in file_data and not os.path.isabs(file_data['sequences']):
            file_data['sequences'] = os.path.join(args.sequences_dir, os.path.basename(file_data['sequences']))
        
        # Check if segmentation path is relative
        if 'segmentation' in file_data and not os.path.isabs(file_data['segmentation']):
            file_data['segmentation'] = os.path.join(args.segmentations_dir, os.path.basename(file_data['segmentation']))
    
    # Create dataset and dataloader
    transform = setup_transforms()
    
    val_ds = Dataset(
        data=val_files,
        transform=transform
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load model
    model = load_model(
        args.model_type, 
        args.model_path, 
        args.model_filename, 
        args.xdim, 
        args.ydim, 
        args.zdim, 
        device
    )
    
    # Check if we're doing an ablation study
    if args.ablation_study:
        # If multi-gpu flag is set and multiple GPUs are available, distribute the ablation tasks
        if args.multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Running ablation study with {torch.cuda.device_count()} GPUs")
            run_ablation_study_multi_gpu(args, model, val_loader, device)
        else:
            if args.multi_gpu and torch.cuda.device_count() <= 1:
                logger.warning("Multi-GPU requested but only one GPU available. Running on single device.")
            run_ablation_study(args, model, val_loader, device)
        return
    
    # Regular inference
    logger.info("Starting inference...")
    start_time = time.time()
    
    # Initialize lists to collect metrics
    all_metrics = []
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Processing validation data"):
            # Get input sequences and target segmentation
            sequences = batch_data["sequences"].to(device)
            segmentation = batch_data["segmentation"].to(device)
            
            # Processing input data
            flair = sequences[:, 0:1, ...]
            t1 = sequences[:, 1:2, ...]
            # t1ce = sequences[:, 2:3, ...] # Not used in input
            t2 = sequences[:, 3:4, ...]
            coord_conv = sequences[:, 4:, ...]
            
            # Combine input channels
            input_channels = torch.cat([flair, t1, t2, coord_conv], dim=1)  # Results in [B, 3+3, H, W, D]
            
            # Perform sliding window inference
            outputs = sliding_window_inference(
                inputs=input_channels,
                roi_size=[args.xdim, args.ydim, args.zdim],
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.sw_overlap,
                mode="gaussian",
                sigma_scale=0.125,
                padding_mode="constant",
                cval=0.0,
                sw_device=device,
                device=device
            )
            
            # Apply sigmoid to get probabilities
            outputs_sigmoid = torch.sigmoid(outputs)
            
            # Calculate metrics for each item in the batch
            for i, output in enumerate(decollate_batch(outputs_sigmoid)):
                metrics = calculate_metrics(output, segmentation[i])
                
                # Include filename for reference
                case_filename = os.path.basename(batch_data["segmentation"].meta["filename_or_obj"][i])
                case_name = case_filename.split('.nii')[0]
                metrics["filename"] = case_name
                all_metrics.append(metrics)
            
            # Save predictions if not metrics_only
            if not args.metrics_only:
                # Process each item in the batch
                for i, (output, output_sigmoid) in enumerate(zip(decollate_batch(outputs), decollate_batch(outputs_sigmoid))):
                    # Get metadata for this sample
                    meta_dict = {
                        'filename_or_obj': batch_data["segmentation"].meta["filename_or_obj"][i],
                        'original_affine': batch_data["segmentation"].meta.get("original_affine", np.eye(4))
                    }
                    
                    # Save binary mask
                    save_prediction_as_nifti(
                        output_sigmoid, 
                        meta_dict, 
                        args.output_dir,
                        filename_prefix="bin",
                        binary=True
                    )
                    
                    # Save probabilities if requested
                    if args.save_probabilities:
                        save_prediction_as_nifti(
                            output, 
                            meta_dict, 
                            args.output_dir,
                            filename_prefix="prob",
                            binary=False
                        )
    
    # Calculate average metrics
    avg_metrics = {key: sum(item[key] for item in all_metrics) / len(all_metrics) 
                 for key in all_metrics[0] if key != "filename"}
    
    # Print metrics summary
    logger.info("\nValidation Metrics Summary:")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save detailed validation metrics
    if args.save_validation_metrics:
        import pandas as pd
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(args.model_path, "validation_sample_metrics.csv")
        
        # Check if file exists - if so, append epoch information
        try:
            existing_df = pd.read_csv(metrics_path)
            max_epoch = existing_df['epoch'].max() if 'epoch' in existing_df.columns else 0
            metrics_df['epoch'] = max_epoch + 1
        except (FileNotFoundError, pd.errors.EmptyDataError):
            metrics_df['epoch'] = 1
        
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved detailed validation metrics to {metrics_path}")
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Inference completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    run_inference()