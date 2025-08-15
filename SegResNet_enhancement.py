# %%
#!/usr/bin/env python3
"""
Run with:
    python enhancement_predictor.py [arguments]

For DDP (distributed training):
    torchrun --nproc_per_node=N enhancement_predictor.py --Use_DDP True
"""
from pathlib import Path
import os
import gc
import random
import math
import time
import logging
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import monai
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from monai import transforms
from monai.config import print_config
from monai.utils import set_determinism
from tqdm import tqdm
from joblib import Parallel, delayed
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from contextlib import nullcontext
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate
import glob
import shutil
import tempfile
from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch, Dataset, CacheDataset, set_track_meta, PersistentDataset, SmartCacheDataset, partition_dataset
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, SwinUNETR
from monai.transforms import Transform, LoadImage
import nibabel as nib # For saving as NIFTI
import pandas as pd
import traceback

# Import wandb if available
try:
    import wandb
except ImportError:
    pass

# Add global variable to store wandb run ID for resume
wandb_run_id = None

class RandomModalityMaskingd(Transform):
    """
    MONAI transform that randomly masks MRI modality channels.
    
    This transform will randomly mask between 0 and 2 of the FLAIR, T1, and T2 channels,
    ensuring at least one of these modalities remains unmasked. The T1CE modality is 
    never masked since it's the target for enhancement prediction.
    
    Args:
        keys (list): List of keys to apply the transform to
        prob (float): Probability of applying the transform
        channel_indices (dict): Dictionary specifying indices of each modality
                               (default assumes channels are [FLAIR, T1, T1CE, T2])
    """
    def __init__(self, keys, prob=0.5, channel_indices=None):
        # Make sure keys is a list even if a single string is passed
        self.keys = [keys] if isinstance(keys, str) else keys
        self.prob = prob
        
        # Default channel indices for [FLAIR, T1, T1CE, T2]
        self.channel_indices = channel_indices or {
            "flair": 0,
            "t1": 1, 
            "t1ce": 2,
            "t2": 3
        }
        
        # Define the channels that can be masked: FLAIR, T1, T2 (not T1CE)
        self.maskable_channels = [
            self.channel_indices["flair"],
            self.channel_indices["t1"],
            self.channel_indices["t2"]
        ]

    def __call__(self, data):
        d = dict(data)
        
        # Apply with probability
        if random.random() < self.prob:
            # Process each key in the keys list, not each character
            for key in self.keys:
                if key not in d:
                    if debug:
                        logger.warning(f"Key {key} not found in data. Skipping.")
                    continue
                
                # Get tensor shape and check if it has channels dimension
                shape = d[key].shape
                if len(shape) < 3:  # Need at least [C, H, W]
                    continue
                
                # Determine how many channels to mask (between 0 and 2)
                num_to_mask = random.randint(0, 2)
                
                # Select which channels to mask (without replacement)
                if num_to_mask > 0:
                    channels_to_mask = random.sample(self.maskable_channels, num_to_mask)
                    
                    # Debug: Print what channels are being masked
                    channel_names = {0: 'FLAIR', 1: 'T1', 3: 'T2'}
                    if debug:
                        masked_names = [channel_names.get(c, f"Channel {c}") for c in channels_to_mask]
                        logger.info(f"Masking channels: {masked_names}")
                    
                    # Create a mask tensor with ones
                    mask = torch.ones_like(d[key])
                    
                    # Zero out the selected channels
                    for channel in channels_to_mask:
                        if d[key].ndim == 5:  # [B, C, H, W, D] format
                            # For 5D tensor (batch, channel, height, width, depth)
                            mask[:, channel:channel+1, ...] = 0
                        elif d[key].ndim == 4:  # [C, H, W, D] format
                            # For 4D tensor (channel, height, width, depth)
                            mask[channel:channel+1, ...] = 0
                        else:
                            # For other formats, try to handle channel dim correctly
                            # Assuming channel is first dim after batch if present
                            slices = [slice(None)] * d[key].ndim
                            if d[key].ndim >= 3:
                                slices[1 if d[key].ndim >= 5 else 0] = slice(channel, channel+1)
                            mask[tuple(slices)] = 0
                    
                    # Apply the mask
                    d[key] = d[key] * mask
                            
        return d

# Early in the file, after imports
def setup_training_mode(debug=False):
    if debug:
        # Debug mode - prioritize determinism
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.autograd.set_detect_anomaly(True)
        logger.setLevel(logging.DEBUG)
    else:
        # Production mode - optimize performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autograd.set_detect_anomaly(False)
        logger.setLevel(logging.INFO)

def clear_memory():
    """Explicitly clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def worker_init_fn(worker_id):
    """Initialize workers with different random seeds"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Initialize with None to be set by argparse later
xdim = None
ydim = None
zdim = None
batch_size = None
lr = None
n_epochs = None
patience = None
gradient_clip = None
augs = None
aug_prob = None
Use_DDP = False
use_dp = False
local_rank = 0
sequences_dir = None
segmentations_dir = None
abnormality_seg_dir = None
data_dir = None
outpath = None
send_to_wandb = False
wandb_location = "online"  # Changed to match the default in argparse
run_name = None
debug = False
steps_per_epoch = None

# Process environment variables for distributed training
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    Use_DDP = True

def save_metrics_to_csv(outpath, train_metrics, val_metrics, codebook_metrics=None, continue_training=False):
    """
    Combines all training and validation metrics along with codebook usage statistics into a single DataFrame 
    and saves it to a CSV file.
    
    Args:
        outpath (str): Directory to save the CSV file
        train_metrics (dict): Dictionary containing training metrics per epoch
                              Keys are metric names, values are lists of values per epoch
        val_metrics (dict): Dictionary containing validation metrics per epoch
                           Keys are metric names, values are lists of values per epoch
        codebook_metrics (dict, optional): Dictionary containing codebook usage statistics
        continue_training (bool): If True, append to existing file instead of overwriting
    
    Returns:
        str or tuple: Path to the saved CSV file, or tuple of (csv_path, max_epoch) when continue_training=True
    """
    try:
        # Check if we have any metrics to save
        if not train_metrics and not val_metrics:
            logger.warning("No metrics available to save to CSV")
            return None
        
        # Create file path
        csv_path = os.path.join(outpath, 'training_metrics.csv')
        
        # Variable to track max epoch when loading existing data
        max_existing_epoch = 0
        
        # Check for existing file when continuing training
        existing_df = None
        if continue_training and os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                logger.info(f"Found existing metrics file with {len(existing_df)} records")
                
                # If we're continuing training, load existing metrics from CSV into train_metrics and val_metrics
                # This ensures metrics are cumulative rather than just from current run
                if 'epoch' in existing_df.columns and len(existing_df) > 0:
                    # Store the max epoch number
                    max_existing_epoch = existing_df['epoch'].max()
                    
                    # Extract train metrics from existing CSV
                    for col in existing_df.columns:
                        if col.startswith('train_'):
                            metric_name = col[6:]  # Remove 'train_' prefix
                            if metric_name in train_metrics:
                                # Convert to list and filter out NaN values
                                values = existing_df[col].tolist()
                                train_metrics[metric_name] = values
                    
                    # Extract val metrics from existing CSV
                    for col in existing_df.columns:
                        if col.startswith('val_'):
                            metric_name = col[4:]  # Remove 'val_' prefix
                            if metric_name in val_metrics:
                                # Convert to list and filter out NaN values
                                values = existing_df[col].tolist()
                                val_metrics[metric_name] = values
                    
                    logger.info(f"Loaded existing metrics from CSV file: {len(existing_df)} epochs, max epoch = {max_existing_epoch}")
            except Exception as e:
                logger.warning(f"Error reading existing metrics file: {str(e)}")
        
        # Get the number of epochs from the longest available metric
        train_lengths = [len(vals) for vals in train_metrics.values() if vals]
        val_lengths = [len(vals) for vals in val_metrics.values() if vals]
        
        if not train_lengths and not val_lengths:
            logger.warning("All metrics lists are empty, cannot create CSV")
            return (None, max_existing_epoch) if continue_training else None
        
        num_epochs = max(max(train_lengths) if train_lengths else 0, 
                         max(val_lengths) if val_lengths else 0)
        
        if num_epochs == 0:
            logger.warning("No valid epochs with metrics data, cannot create CSV")
            return (None, max_existing_epoch) if continue_training else None
        
        # Create a DataFrame with epoch numbers
        # If continuing, adjust epoch numbers to continue from existing data
        start_epoch = 1
        if existing_df is not None and 'epoch' in existing_df.columns and len(existing_df) > 0:
            start_epoch = existing_df['epoch'].max() + 1
            
        df = pd.DataFrame({'epoch': list(range(start_epoch, start_epoch + num_epochs))})
        
        # Add training metrics to the DataFrame
        for metric_name, values in train_metrics.items():
            if values:  # Only add if there are values
                # Pad the list if it's shorter than num_epochs
                padded_values = values + [None] * (num_epochs - len(values))
                df[f'train_{metric_name}'] = padded_values
        
        # Add validation metrics to the DataFrame
        for metric_name, values in val_metrics.items():
            if values:  # Only add if there are values
                # Pad the list if it's shorter than num_epochs
                padded_values = values + [None] * (num_epochs - len(values))
                df[f'val_{metric_name}'] = padded_values
        
        # If continuing training and we have existing data, append to it
        if existing_df is not None:
            # Combine existing and new data
            df = pd.concat([existing_df, df], ignore_index=True)
            logger.info(f"Appending {num_epochs} new records to existing {len(existing_df)} records")
        
        # Save DataFrame to CSV
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved metrics to {csv_path} ({len(df)} total records)")
        
        # Return appropriate value based on continue_training flag
        if continue_training:
            return (csv_path, max_existing_epoch)
        else:
            return csv_path
    except Exception as e:
        logger.warning(f"Failed to save metrics to CSV: {str(e)}")
        return (None, max_existing_epoch) if continue_training else None
    
def calculate_enhancement_losses(inputs, pred, target, device, loss_function, filenames=None, is_validation=False):
    """
    Calculate losses for T1CE enhancement prediction.
    If is_validation is True, also calculates and returns metrics per item in the batch.

    Args:
        inputs (tensor): Input tensor [B, C, H, W, D]
        pred (tensor): Model predictions (logits) [B, 1, H, W, D]
        target (tensor): Target binary mask [B, 1, H, W, D]
        device: Device to run calculations on
        loss_function: The loss function (e.g., DiceLoss)
        filenames (list, optional): List of filenames corresponding to batch items. Required if is_validation=True.
        is_validation (bool): Flag indicating if this is a validation step.

    Returns:
        tuple: 
          - loss (tensor): Batch loss.
          - batch_metrics (dict): Aggregated metrics for the batch.
          - individual_metrics (list or None): List of dicts, each containing metrics for one item 
                                              if is_validation=True, otherwise None.
    """
    
    loss = loss_function(pred, target)
    
    individual_metrics = []
    batch_metrics = {
        'loss': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'dice': 0.0,
        'balanced_acc': 0.0,
    }
    num_valid_items = 0

    # Initialize MONAI DiceMetric for computing Dice coefficient
    dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean", get_not_nans=False, ignore_empty=False)

    # Calculate metrics
    with torch.no_grad():
        # Apply sigmoid to get probabilities
        pred_sigmoid = torch.sigmoid(pred)
        # Threshold predictions at 0.5
        pred_binary = (pred_sigmoid > 0.5).float()
        
        batch_size = pred.shape[0]
        
        # Extract filenames if this is validation and we have metadata
        if is_validation and hasattr(filenames, 'meta') and 'filename_or_obj' in filenames.meta:
            batch_filenames = [os.path.basename(f).split('.nii')[0] for f in filenames.meta['filename_or_obj']]
        else:
            batch_filenames = [f"sample_{i}" for i in range(batch_size)]
            
        # Process each item in the batch individually
        for i in range(batch_size):
            # Get the target and prediction for this item
            target_i = target[i]
            pred_binary_i = pred_binary[i]
            
            # Calculate pixel-level metrics
            tp = torch.sum((pred_binary_i == 1) & (target_i == 1)).float()
            fp = torch.sum((pred_binary_i == 1) & (target_i == 0)).float()
            tn = torch.sum((pred_binary_i == 0) & (target_i == 0)).float()
            fn = torch.sum((pred_binary_i == 0) & (target_i == 1)).float()
            
            # Compute metrics with epsilon to avoid division by zero
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            balanced_acc = 0.5 * (tp/(tp + fn + 1e-8) + tn/(tn + fp + 1e-8))
            
            # Calculate per-sample Dice using MONAI's implementation
            dice_result = dice_metric(y_pred=pred_binary_i.unsqueeze(0), y=target_i.unsqueeze(0))
            # print(f"Sample {i}: Dice = {dice_result}")

            # Check if the result is valid (not empty and not NaN)
            if dice_result.numel() > 0 and not torch.isnan(dice_result).any():
                sample_dice = dice_result.item()
            else:
                logger.warning(f"Sample {i}: Dice result is empty or NaN. Falling back to manual calculation.")
                # If dice_result is empty or contains NaN, calculate manually
                intersection = torch.sum(pred_binary_i * target_i).float()
                union = torch.sum(pred_binary_i).float() + torch.sum(target_i).float()
                
                # If both prediction and target are empty, this is a perfect match with dice=1.0
                if union == 0:
                    sample_dice = 1.0  # Perfect agreement when nothing to detect and nothing detected
                else:
                    sample_dice = (2.0 * intersection) / (union + 1e-5)
                
            # print(f"Sample {i}: Dice = {sample_dice:.4f}")
            dice_metric.reset()  # Reset the metric
                    
            # Add to batch total for averaging
            batch_metrics['precision'] += precision.item()
            batch_metrics['recall'] += recall.item()
            batch_metrics['f1'] += f1.item()
            batch_metrics['balanced_acc'] += balanced_acc.item()
            batch_metrics['dice'] += sample_dice
            num_valid_items += 1
            
            # Store individual metrics for validation
            if is_validation:
                individual_metrics.append({
                    'filename': batch_filenames[i],
                    'precision': precision.item(),
                    'recall': recall.item(),
                    'f1': f1.item(),
                    'dice': sample_dice,
                    'balanced_acc': balanced_acc.item()
                })

    # Average batch metrics (avoid division by zero)
    if num_valid_items > 0:
        batch_metrics['precision'] /= num_valid_items
        batch_metrics['recall'] /= num_valid_items
        batch_metrics['f1'] /= num_valid_items
        batch_metrics['dice'] /= num_valid_items
        batch_metrics['balanced_acc'] /= num_valid_items
    else:        
        logger.warning("No valid items in batch for metrics calculation")

    batch_metrics['loss'] = loss.item() # Assign batch loss
        
    return loss, batch_metrics, individual_metrics if is_validation else None

def setup_transforms(seg_key, mri_keys, abnormality_seg_key, all_keys, xdim, ydim, zdim, augs, aug_prob):
    histogram_shift_control_points=10
    three_d_deform_magnitudes=(0.1,1)
    three_d_deform_sigmas = (1,2)
    scale_range = (0.05, 0.05, 0.05)
    translate_range = (3, 3, 3)
    shear_angle_in_rads = 4 * (2 * 3.14159 / 360)  # the number on the far left is in degrees!
    rot_angle_in_rads = 4 * (2 * 3.14159 / 360)
    nii_target_shape=[128,128,128]
    min_small_crop_size = [int(0.95 * d) for d in nii_target_shape]
    max_inflation_factor = 0.05
    sigma_lower = 0.1
    sigma_upper = 0.5
    cw_application=True

    # Convert keys to strings instead of lists
    seg_key_str = seg_key[0]
    mri_keys_str = mri_keys[0]
    # abnormality_seg_key_str = abnormality_seg_key[0]
    all_seg_keys_str = [seg_key_str]
    all_keys_str = [seg_key_str, mri_keys_str] #abnormality_seg_key_str

    train_transforms = [transforms.LoadImaged(keys=all_keys_str)]
    train_transforms += [transforms.EnsureChannelFirstd(keys=all_keys_str)]
    train_transforms += [transforms.ToTensord(keys=all_keys_str)]
    train_transforms += [transforms.SignalFillEmptyd(keys=all_keys_str)]
    train_transforms += [transforms.AddCoordinateChannelsd(keys=mri_keys_str, spatial_dims=[0,1,2])]
    val_transforms = train_transforms.copy()
    train_transforms += [transforms.CropForegroundd(keys=all_keys_str, source_key=mri_keys_str, margin=1,allow_smaller=False)]
    train_transforms += [transforms.ClipIntensityPercentilesd(keys=mri_keys_str, lower=1, upper=99,channel_wise=cw_application)]
    train_transforms += [transforms.NormalizeIntensityd(keys=mri_keys_str,channel_wise=cw_application)]
    train_transforms += [transforms.ScaleIntensityd(keys=mri_keys_str,channel_wise=cw_application)]
    train_transforms += [transforms.RandFlipd(keys=all_keys_str, prob=0.5, spatial_axis=[0])]

    if augs:
        print('Using augmentations...')
        train_transforms += [transforms.RandFlipd(keys=all_keys_str, prob=aug_prob, spatial_axis=[1])]
        train_transforms += [transforms.RandFlipd(keys=all_keys_str, prob=aug_prob, spatial_axis=[2])]
        train_transforms += [transforms.RandSpatialCropd(keys=all_keys_str, roi_size=[xdim, ydim, zdim], random_size=False)]
        # train_transforms += [transforms.RandCropByPosNegLabeld(keys=all_keys_str,label_key=all_seg_keys_str,spatial_size=(xdim,ydim,zdim),image_key=mri_keys_str,allow_smaller=False)]
        train_transforms += [transforms.RandCoarseDropoutd(keys=mri_keys_str, prob=aug_prob, max_spatial_size=10, holes=1, max_holes=50, spatial_size=1)]
        train_transforms += [transforms.RandShiftIntensityd(keys=mri_keys_str, offsets = 0.15, prob=aug_prob)]
        train_transforms += [transforms.RandScaleIntensityd(keys=mri_keys_str, prob=aug_prob,factors=0.15)]
        train_transforms += [transforms.RandGaussianNoised(keys=mri_keys_str, prob=aug_prob, std=0.025)]
        train_transforms += [transforms.RandStdShiftIntensityd(keys=mri_keys_str, prob=aug_prob, factors=0.1)]
        train_transforms += [transforms.RandBiasFieldd(keys=mri_keys_str, prob=aug_prob)]
        train_transforms += [transforms.RandHistogramShiftd(keys=mri_keys_str, prob=aug_prob, num_control_points=histogram_shift_control_points)]
        train_transforms += [transforms.RandGaussianSmoothd(keys=mri_keys_str, prob=aug_prob, sigma_x=(sigma_lower, sigma_upper), sigma_y=(sigma_lower,sigma_upper), sigma_z=(sigma_lower,sigma_upper))]
        train_transforms += [transforms.RandSimulateLowResolutiond(keys=mri_keys_str, prob=aug_prob, zoom_range=(0.5,1.0), downsample_mode="area", upsample_mode="trilinear")]

        train_transforms += [transforms.RandAffined(keys=all_keys_str,
                                                    prob=aug_prob,
                                                    rotate_range=rot_angle_in_rads,
                                                    shear_range=shear_angle_in_rads,
                                                    translate_range=translate_range,
                                                    scale_range=scale_range,
                                                    spatial_size=None,
                                                    padding_mode="border",
                                                    mode='nearest')]
        train_transforms += [transforms.Rand3DElasticd(keys=all_keys_str,
                                                        sigma_range=three_d_deform_sigmas,
                                                        magnitude_range=three_d_deform_magnitudes,
                                                        prob=aug_prob,
                                                        rotate_range=rot_angle_in_rads,
                                                        shear_range=shear_angle_in_rads,
                                                        translate_range=translate_range,
                                                        scale_range=scale_range,
                                                        spatial_size=None,
                                                        padding_mode="border",
                                                    mode='nearest')]
        train_transforms += [transforms.NormalizeIntensityd(keys=mri_keys_str,channel_wise=cw_application)]
        train_transforms += [transforms.ScaleIntensityd(keys=mri_keys_str,channel_wise=cw_application)]
        
        # # Add random modality masking - specify correct channel mapping
        # # Assumes MRI channels are in order: [FLAIR, T1, T1CE, T2]
        train_transforms += [RandomModalityMaskingd(keys=mri_keys_str, prob=0)] #aug_prob*2

    if not augs:
        print('Not using augmentations')
        train_transforms += [transforms.Resized(keys=mri_keys_str, spatial_size=(xdim, ydim, zdim),mode='trilinear')]
        train_transforms += [transforms.Resized(keys=all_seg_keys_str, spatial_size=(xdim, ydim, zdim),mode='nearest')]
        train_transforms += [transforms.NormalizeIntensityd(keys=mri_keys_str,channel_wise=cw_application)]
        train_transforms += [transforms.ScaleIntensityd(keys=mri_keys_str,channel_wise=cw_application)]
        
    val_transforms += [transforms.NormalizeIntensityd(keys=mri_keys_str,channel_wise=cw_application)]
    val_transforms += [transforms.ScaleIntensityd(keys=mri_keys_str,channel_wise=cw_application)]
    
    train_transforms = transforms.Compose(train_transforms)
    val_transforms = transforms.Compose(val_transforms)
    
    return train_transforms, val_transforms
           
# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Segmentation Enhancement")

# Print MONAI and PyTorch configuration
print_config()
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA version: {torch.version.cuda}")
logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

def save_prediction_as_nifti(image_tensor, metadata, output_dir, filename_prefix="pred", binary=False):
    """
    Saves a prediction tensor as a NIFTI file using the original image's metadata.
    
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
    if 'original_affine' in metadata:
        affine = metadata['original_affine']
        if debug:
            print('Affine', affine)
            print('Affine shape', affine.shape)
    else:
        # If no affine is available, use identity matrix
        logger.warning(f"No affine found in metadata for {orig_filename}. Using identity.")
        affine = np.eye(4)
    
    # print('image_array shape', image_array.shape)
    
    # Create NIFTI image and save
    # Need to reshape if there's a channel dimension (nibabel doesn't support channel dimension)
    if image_array.shape[0] == 1:  # If we have a single channel
        image_array = image_array[0]  # Remove channel dimension
    if image_array.shape[0] == 1:  # If we have a single channel
        image_array = image_array[0]  # Remove channel dimension
        
    # print('image_array shape', image_array.shape)
    
    nifti_image = nib.Nifti1Image(image_array, affine)
    nib.save(nifti_image, output_path)
    
    return output_path

def setup_data():
    """Setup datasets and dataloaders with train/val splitting"""
    # Use the global variables set by argparse
    global sequences_dir, segmentations_dir, abnormality_seg_dir, train_transforms, val_transforms, train_ds, val_ds, Use_DDP, use_dp, training_dataset, persistent_val_dataset, cache_val_dataset, args, local_rank
    
    # Setup transforms
    abnormality_seg_key = ["abnormality_segmentation"]
    seg_key = ["segmentation"]
    all_seg_keys = seg_key # + abnormality_seg_key # Assuming abnormality seg is not used based on check_file_pair
    mri_keys = ["sequences"]
    all_keys = seg_key + mri_keys # + abnormality_seg_key
    train_transforms, val_transforms = setup_transforms(seg_key, mri_keys, abnormality_seg_key, all_keys, xdim, ydim, zdim, augs, aug_prob)

    all_files = []
    removed_count = 0  # Counter for removed files

    os.makedirs(outpath, exist_ok=True)  # Ensure output directory exists

    if not os.path.exists(sequences_dir):
        raise RuntimeError(f"Sequences directory not found: {sequences_dir}")
    if not os.path.exists(segmentations_dir):
        raise RuntimeError(f"Segmentations directory not found: {segmentations_dir}")

    # Path for the data split JSON file
    split_file_path = os.path.join(outpath, "data_split.json")

    # Check if we can bypass file checking and directly load from the existing split
    if args.continue_training and os.path.exists(split_file_path):
        if not Use_DDP or local_rank == 0:
            logger.info(f"Continue training enabled and data_split.json exists. Loading file list directly from {split_file_path}")
        
        try:
            # Load the split file to get all_files, train_files, and val_files
            with open(split_file_path, 'r') as f:
                split_data = json.load(f)
            
            if 'train_files' in split_data and 'val_files' in split_data:
                # Newer format with the full file lists stored
                train_files = split_data['train_files']
                val_files = split_data['val_files']
                all_files = train_files + val_files
                if not Use_DDP or local_rank == 0:
                    logger.info(f"Loaded {len(all_files)} files ({len(train_files)} train, {len(val_files)} val) directly from split file.")
                
                # With DDP, ensure all ranks have the same data
                if Use_DDP:
                    # Broadcast is not needed since all ranks read the same file
                    # But we'll add a barrier to ensure synchronization
                    dist.barrier()
                
                # Skip to dataset creation
                goto_dataset_creation = True
            
            else:
                # Handle older format that only has indices
                if 'train_indices' in split_data and 'val_indices' in split_data:
                    if not Use_DDP or local_rank == 0:
                        logger.info("Found older split format with indices only. Need to scan files first.")
                goto_dataset_creation = False
                # Will continue with regular file checking below
        
        except Exception as e:
            if not Use_DDP or local_rank == 0:
                logger.warning(f"Error loading split file: {e}. Falling back to file scanning.")
            goto_dataset_creation = False
            # Will continue with regular file checking below
    else:
        goto_dataset_creation = False
        # Will continue with regular file checking below

    # Only perform file scanning if needed (not bypassing)
    if not goto_dataset_creation:
        # First get all sequence files (only rank 0 needs the full list initially in DDP)
        sequence_files = []
        if not Use_DDP or local_rank == 0:
            sequence_files = sorted(glob.glob(os.path.join(sequences_dir, "*.nii.gz")))
            logger.info(f"[Rank {local_rank if Use_DDP else 'N/A'}] Found {len(sequence_files)} potential sequence files.")

        # Initialize LoadImage transform for checking segmentation sum (only needed where loading happens)
        image_loader = LoadImage(image_only=True)

        def check_and_load_file_pair(seq_path, load_and_check_data=True): # Add parameter back
            """Helper function to check a single sequence/segmentation pair, optionally loading data."""
            base_name = os.path.basename(seq_path).replace(".nii.gz", "")
            seg_path = os.path.join(segmentations_dir, f"{base_name}.nii.gz")

            if os.path.exists(seg_path):
                if load_and_check_data: # Conditionally load and check
                    try:
                        # Load data for checking
                        seq_data = image_loader(seq_path)
                        seg_data = image_loader(seg_path)

                        # Check for NaN/Inf
                        if np.isnan(seq_data).any() or np.isinf(seq_data).any():
                            # logger.warning(f"Sequence file {seq_path} contains NaN or Inf values. Skipping file pair.") # Reduce verbosity
                            return None, 1
                        if np.isnan(seg_data).any() or np.isinf(seg_data).any():
                            # logger.warning(f"Segmentation file {seg_path} contains NaN or Inf values. Skipping file pair.") # Reduce verbosity
                            return None, 1

                        # Check if segmentation is non-empty
                        if np.sum(seg_data) > -1: # changed to -1 so all include
                            return {"sequences": seq_path, "segmentation": seg_path}, 0
                        else:
                            # logger.warning(f"Segmentation {seg_path} is empty (sum=0). Skipping file pair.") # Reduce verbosity
                            return None, 1
                    except Exception as e:
                        logger.warning(f"Could not load or process files {seq_path}/{seg_path}: {e}. Skipping file pair.")
                        return None, 1 # Return None and indicate removal on error
                else:
                    # If not loading/checking, assume valid if seg file exists
                    return {"sequences": seq_path, "segmentation": seg_path}, 0
            else:
                # logger.warning(f"Segmentation file not found for {seq_path}. Skipping file pair.") # Reduce verbosity
                return None, 1

        # Define a wrapper for joblib that passes necessary args if needed
        # (image_loader might need re-instantiation if not picklable, but LoadImage usually is)
        def check_file_pair_joblib(seq_path):
             # loader = LoadImage(image_only=True) # Re-create if needed
             # Always load and check when using joblib
             return check_and_load_file_pair(seq_path) 

        if Use_DDP:
            object_list = [None, None] # Placeholder for [all_files, removed_count]

            if local_rank == 0:
                logger.info(f"[Rank 0] Checking {len(sequence_files)} potential file pairs using parallel processing...")
                n_jobs = max(1, os.cpu_count() - 10) # Ensure at least 1 job
                logger.info(f"[Rank 0] Using {n_jobs} parallel workers for file checking")

                # Use joblib for parallel checking on Rank 0
                results = Parallel(n_jobs=n_jobs, verbose=10)(
                    delayed(check_file_pair_joblib)(seq_path) for seq_path in sequence_files
                )

                # Collect results on Rank 0
                rank0_all_files = []
                rank0_removed_count = 0
                for result in results: # Iterate through the results from joblib
                    if result is not None: # Check if joblib returned a result (should be a tuple)
                        file_pair, removed = result
                        if file_pair:
                            rank0_all_files.append(file_pair)
                        rank0_removed_count += removed
                    else:
                        # Handle cases where joblib might return None (e.g., error in worker)
                        logger.warning("Joblib worker returned None, potential issue during file check.")
                        # Decide if None implies removal
                        rank0_removed_count += 1 

                logger.info(f"[Rank 0] Found {len(rank0_all_files)} valid files after checking. Removed {rank0_removed_count} pairs.")
                object_list = [rank0_all_files, rank0_removed_count] # Prepare for broadcast

            # Broadcast the list containing all_files and removed_count from rank 0
            logger.info(f"[Rank {local_rank}] Waiting for file list broadcast from Rank 0...")
            dist.broadcast_object_list(object_list, src=0)
            logger.info(f"[Rank {local_rank}] Received file list from Rank 0.")

            # All ranks unpack the broadcasted data
            all_files = object_list[0]
            removed_count = object_list[1]

            # Barrier to ensure all ranks have the data before proceeding
            dist.barrier()
        else:
            # Original Non-DDP logic using joblib
            logger.info(f"Checking {len(sequence_files)} potential file pairs using parallel processing...")
            n_jobs = max(1, os.cpu_count() - 10) # Ensure at least 1 job
            logger.info(f"Using {n_jobs} parallel workers for file checking")

            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(check_file_pair_joblib)(seq_path) for seq_path in sequence_files
            )

            # Collect results
            for result in results: # Iterate through the results from joblib
                if result is not None: # Check if joblib returned a result (should be a tuple)
                    file_pair, removed = result
                    if file_pair:
                        all_files.append(file_pair)
                    removed_count += removed
                else:
                    # Handle cases where joblib might return None
                    logger.warning("Joblib worker returned None, potential issue during file check.")
                    # Decide if None implies removal
                    removed_count += 1

        if removed_count > 0:
            # Log only on rank 0 in DDP to avoid redundant messages
            if not Use_DDP or local_rank == 0:
                logger.info(f"Removed {removed_count} file pairs due to missing files, empty segmentations, or NaN/Inf values.")

        if not all_files:
            # Raise error on all ranks if no files are found
            raise RuntimeError("No valid pairs of sequence and non-empty segmentation files found after filtering.")

        # Log total count (only rank 0 in DDP)
        if not Use_DDP or local_rank == 0:
            logger.info(f"Found {len(all_files)} total valid files for training/validation.")

        # --- Split Logic for creating new train/val split when needed ---
        train_files, val_files = [], []
        save_new_split = False

        if args.continue_training and os.path.exists(split_file_path):
            # Attempt to load existing split (all ranks do this to define train/val files)
            if not Use_DDP or local_rank == 0: # Log only once
                logger.info(f"Loading existing train/validation split from {split_file_path}")
            try:
                with open(split_file_path, 'r') as f:
                    split_data = json.load(f)
                train_indices = split_data.get('train_indices', [])
                val_indices = split_data.get('val_indices', [])

                # Validate indices against the current all_files list
                current_num_files = len(all_files)
                train_indices = [i for i in train_indices if i < current_num_files]
                val_indices = [i for i in val_indices if i < current_num_files]

                if not train_indices or not val_indices:
                     if not Use_DDP or local_rank == 0:
                        logger.warning("Loaded split file contains empty splits or invalid indices for the current dataset. Creating new split.")
                     # Fallback to creating a new split if loaded one is invalid
                     train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=42)
                     save_new_split = True # Mark to save the new split later
                else:
                    train_files = [all_files[idx] for idx in train_indices]
                    val_files = [all_files[idx] for idx in val_indices]
                    if not Use_DDP or local_rank == 0:
                        logger.info(f"Successfully loaded split with {len(train_files)} training and {len(val_files)} validation files")
                    # save_new_split = False # Don't overwrite existing valid split (already default)

            except Exception as e:
                if not Use_DDP or local_rank == 0:
                    logger.warning(f"Error loading split file: {e}. Creating new split.")
                train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=42)
                save_new_split = True
        else:
            # Create a new split
            if not Use_DDP or local_rank == 0:
                logger.info("Creating new train/validation split")
            train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=42)
            save_new_split = True

        # Save the split file only on rank 0 if a new split was created
        if save_new_split and local_rank == 0: # Check local_rank == 0 (works for DDP and non-DDP)
            train_indices = [all_files.index(file) for file in train_files]
            val_indices = [all_files.index(file) for file in val_files]
            split_data = {
                'train_indices': train_indices,
                'val_indices': val_indices,
                'train_files': train_files, # Include filenames
                'val_files': val_files,     # Include filenames
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_size': len(all_files),
                'train_size': len(train_files),
                'val_size': len(val_files)
            }
            try:
                with open(split_file_path, 'w') as f:
                    json.dump(split_data, f, indent=4)
                logger.info(f"Saved new train/validation split (including filenames) to {split_file_path}")
            except Exception as e:
                logger.error(f"Failed to save split file: {e}")

        # Ensure all ranks wait until rank 0 potentially saves the file before proceeding
        if Use_DDP:
            dist.barrier()

    # We're at the dataset creation point
    if not Use_DDP or local_rank == 0:
        logger.info(f"Split into {len(train_files)} training files and {len(val_files)} validation files")

    # --- Dataset and DataLoader creation (remains largely the same) ---
    # Use regular Dataset instead of SmartCache/Cache
    if training_dataset == 'Dataset':
        logger.info("Using Dataset for training data")
        train_ds = monai.data.Dataset(
            data=train_files,
            transform=train_transforms
        )
    
    elif training_dataset == 'CacheDataset':
        logger.info("Using CacheDataset for training data")
        train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=0.01,num_workers=None)
        
    elif training_dataset == 'SmartCacheDataset':
        logger.info("Using SmartCacheDataset for training data")
        if not Use_DDP:
            train_ds = SmartCacheDataset(data=train_files, transform=train_transforms, 
                                        replace_rate=args.smart_cache_replace_rate, # Use args.smart_cache_replace_rate
                                        cache_num=int(len(train_files) * args.smart_cache_num), # Use args.smart_cache_num
                                        num_init_workers = None, #int(os.cpu_count() / 2
                                        num_replace_workers = None,
                                        progress=True if local_rank==0 else False
                                        )
            
        if Use_DDP:
            data_part = partition_dataset(
            data=train_files,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            even_divisible=True,
            )[dist.get_rank()]
            
            train_ds = SmartCacheDataset(
            data=data_part,
            transform=train_transforms,
            replace_rate=args.smart_cache_replace_rate, # Use args.smart_cache_replace_rate
            cache_num=int(len(train_files) * args.smart_cache_num), # Use args.smart_cache_num
            num_init_workers = None, #None
            num_replace_workers = None, #None
            )
    
    if cache_val_dataset:
        # Use CacheDataset for validation when enabled
        val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,  # Cache all validation data
        num_workers=None,
        )
        pw=False
        logger.info("Using CacheDataset for validation data")
    elif persistent_val_dataset:
        # Use PersistentDataset for validation when enabled
        val_ds = PersistentDataset(
        data=val_files,
        transform=val_transforms,
        cache_dir=os.path.join(outpath, "cache"),  # Specify cache directory
        )
        pw=False
        logger.info("Using PersistentDataset for validation data")

    else:
        # Use regular Dataset for validation
        val_ds = monai.data.Dataset(
        data=val_files,
        transform=val_transforms
        )
        pw=True
        logger.info("Using regular Dataset for validation data")

    # Create samplers for distributed training (DDP only)
    train_sampler = None
    val_sampler = None
    if Use_DDP and training_dataset != 'SmartCacheDataset':
        # DDP requires DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) # drop_last recommended for DDP sampler
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    # Determine shuffle for DataLoader based on sampler presence
    train_shuffle = (train_sampler is None) and (training_dataset != 'SmartCacheDataset') # Shuffle if no sampler (DP/Single GPU), except for SmartCache
    
    # Determine appropriate number of workers
    world_size = dist.get_world_size() if Use_DDP else 1
    train_workers = 0 if training_dataset == 'SmartCacheDataset' else args.num_workers
    val_workers = 0 if persistent_val_dataset or cache_val_dataset else args.num_workers
    # Ensure at least 1 worker unless 0 is required, but respect user's choice of 0
    if args.num_workers > 0:
        train_workers = max(1, train_workers) if training_dataset != 'SmartCacheDataset' else 0
        val_workers = max(1, val_workers) if not (persistent_val_dataset or cache_val_dataset) else 0
    else:
        train_workers = 0
        val_workers = 0

    logger.info(f"Using {train_workers} workers for training loader.")
    logger.info(f"Using {val_workers} workers for validation loader.")

    # Determine persistent workers for validation based on dataset type
    val_persistent = val_workers > 0 and not (persistent_val_dataset or cache_val_dataset)

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle, # Use calculated shuffle flag
        num_workers=train_workers, # Use calculated workers
        pin_memory=True, # Enable pin_memory
        sampler=train_sampler, # Pass sampler if created (DDP)
        drop_last=True, # drop_last=True often helps with DDP/batch norm
        persistent_workers=train_workers > 0, # Use persistent if workers > 0
        worker_init_fn=worker_init_fn if train_workers > 0 else None # Add worker init if workers > 0
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size, # Use args.val_batch_size
        shuffle=False, # Validation is never shuffled
        num_workers=val_workers, # Use calculated workers
        pin_memory=True, # Enable pin_memory
        sampler=val_sampler, # Pass sampler if created (DDP)
        drop_last=False, # Keep all samples for validation
        persistent_workers=val_persistent # Use persistent if applicable
    )
    
    # Create directories for visualizations
    if local_rank == 0: # Only create on rank 0
        os.makedirs(os.path.join(outpath, "training_samples"), exist_ok=True)
        os.makedirs(os.path.join(outpath, "validation_samples"), exist_ok=True)
        
    return train_loader, val_loader, train_sampler, val_sampler, train_ds

def visualize_batch_samples(batch_data, outputs, epoch, phase="training", n_examples=10, outpath=None, send_to_wandb=False):
    """
    Visualize a batch of samples with their predictions and ground truth.
    
    Args:
        batch_data (dict): Batch data from the dataloader
        outputs (tensor): Model output tensor
        epoch (int): Current epoch number
        phase (str): Either "training" or "validation"
        n_examples (int): Number of examples to visualize
        outpath (str): Output path to save the visualization
        send_to_wandb (bool): Whether to log the visualization to wandb
        
    Returns:
        matplotlib.figure.Figure: The visualization figure
    """
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    
    if outpath:
        sample_dir = os.path.join(outpath, f"{phase}_samples")
        os.makedirs(sample_dir, exist_ok=True)
    
    # Get batch data
    sequences = batch_data["sequences"]
    segmentation = batch_data["segmentation"]
    
    # Extract filenames from the metadata
    filenames = []
    for filename in segmentation.meta["filename_or_obj"][:n_examples]:
            # Extract just the filename without path and extension
            filenames.append(os.path.basename(filename).split('.nii')[0])
    
    # Move tensors to CPU and convert to numpy
    sequences = sequences.detach().cpu().numpy()
    segmentation = segmentation.detach().cpu().numpy()
    outputs_sigmoid = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu().numpy()
    
    # Threshold outputs to get binary predictions
    outputs_binary = (outputs_sigmoid > 0.5).to(torch.float32).cpu().numpy() # Correct: use .to() for tensor type conversion, then convert to numpy
    
    # Limit the number of examples to the batch size or n_examples
    batch_size = sequences.shape[0]
    n_examples = min(n_examples, batch_size)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_examples, 6, figsize=(21, 3 * n_examples))
    
    # Set a title for the entire figure
    # fig.suptitle(f"{phase.capitalize()} Samples - Epoch {epoch+1:05d}", y=1.1)
    
    # If we have only one example, axes won't be a 2D array
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    # Define column names
    col_names = ['FLAIR', 'T1', 'T2', 'T1ce','Ground Truth', 'Prediction']
    
    # Loop through each example in the batch
    for i in range(n_examples):
        flair = sequences[i, 0, ...]  # FLAIR channel
        t1 = sequences[i, 1, ...]     # T1 channel
        t1ce = sequences[i, 2, ...]   # T1CE channel
        t2 = sequences[i, 3, ...]     # T2 channel
        gt = segmentation[i, 0, ...]  # Ground truth
        pred = outputs_binary[i, 0, ...] # Binary prediction
        pred_prob = outputs_sigmoid[i, 0, ...] # Prediction probabilities
        
        # Get the middle slice for 2D visualization (adjust based on where the tumor typically is)
        # Find the slice with the most nonzero values in the ground truth
        gt_sum_per_slice = np.sum(gt, axis=(0,1))
        if np.sum(gt_sum_per_slice) > 0:  # If there's any segmentation
            z_mid = np.argmax(gt_sum_per_slice)
        else:
            # Fallback to middle slice if no segmentation is found
            z_mid = gt.shape[-1] // 2

        # Get 2D slices for visualization
        flair_slice = flair[:, :, z_mid]
        t1_slice = t1[:, :, z_mid]
        t1ce_slice = t1ce[:, :, z_mid]
        t2_slice = t2[:, :, z_mid]
        gt_slice = gt[:, :, z_mid]
        pred_slice = pred[:, :, z_mid]
        pred_prob_slice = pred_prob[:, :, z_mid]
        # Plot the images - rotated by 90 degrees
        axes[i, 0].imshow(np.rot90(flair_slice), cmap='gray')
        axes[i, 1].imshow(np.rot90(t1_slice), cmap='gray')
        axes[i, 2].imshow(np.rot90(t2_slice), cmap='gray')
        axes[i, 3].imshow(np.rot90(t1ce_slice), cmap='gray')
        axes[i, 4].imshow(np.rot90(t1ce_slice), cmap='gray')
        gt_masked = np.ma.masked_where(gt_slice == 0, gt_slice)
        
        green_cmap = ListedColormap(['none', 'green'])  # 'none' for 0 values (transparent), 'green' for 1 values
        
        # Show background T1CE image
        axes[i, 4].imshow(np.rot90(t1ce_slice), cmap='gray')
        # Overlay the ground truth in green
        axes[i, 4].imshow(np.rot90(gt_slice), cmap=green_cmap, alpha=0.5, vmin=0, vmax=1)
        axes[i, 5].imshow(np.rot90(t1ce_slice), cmap='gray')
        pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
        axes[i, 5].imshow(np.rot90(pred_masked), cmap=green_cmap, alpha=0.5, vmin=0, vmax=1)

        if i == 0:
            axes[i, 0].set_title(f"FLAIR")
            axes[i, 1].set_title(f"T1")
            axes[i, 2].set_title(f"T2")
            axes[i, 3].set_title(f"T1CE\n(Held out from model)")
            axes[i, 4].set_title(f"Ground Truth\n(From T1CE)")
            axes[i, 5].set_title(f"Enhancing Prediction\n(From T1, T2, FLAIR)")
            
        # Add filename on the left instead of generic "Sample X"
        if i < len(filenames):
            axes[i, 0].set_ylabel(filenames[i])
        else:
            # Fall back to generic label if filename not available
            axes[i, 0].set_ylabel(f"Sample {i+1}")
        
        # Remove axis ticks for cleaner look
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    if phase == "training":
        plt.subplots_adjust(wspace=-0.6, hspace=0.0)
    else:
        plt.subplots_adjust(wspace=-0.7, hspace=-0.05)
    
    # Save the figure if outpath is provided
    if outpath:
        fig_path = os.path.join(sample_dir, f"{phase}_samples_epoch_{epoch+1:05d}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
    # Log to wandb if requested
    if send_to_wandb and (not Use_DDP or local_rank == 0):
        try:
            import wandb
            wandb.log({
                f"{phase}_samples": wandb.Image(fig),
                "epoch": epoch + 1
            })
        except ImportError:
            logger.warning("wandb not available for logging visualization")
        except Exception as e:
            logger.warning(f"Failed to log visualization to wandb: {str(e)}")
    
    plt.close('all')
    return fig

def plot_metrics(train_metrics, val_metrics, epoch, outpath=None, send_to_wandb=False):
    """
    Create a comprehensive visualization of training and validation metrics
    
    Args:
        train_metrics (dict): Dictionary with training metrics
        val_metrics (dict): Dictionary with validation metrics
        epoch (int): Current epoch number
        outpath (str): Output directory path
        send_to_wandb (bool): Whether to send the figure to wandb
        
    Returns:
        matplotlib.figure.Figure: The visualization figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create metrics directory if it doesn't exist
    if outpath:
        metrics_dir = os.path.join(outpath, "metrics_plots")
        os.makedirs(metrics_dir, exist_ok=True)
    
    # Set up the figure with subplots (4 x 3 grid)
    fig, axes = plt.subplots(5, 1, figsize=(18, 16))
    fig.suptitle(f"Training and Validation Metrics (Epoch {epoch+1})", fontsize=16, y=0.98)
    
    # Flatten the axes for easier indexing
    axes = axes.flatten()
    
    # Define metrics to plot with nice titles
    metrics_to_plot = [
        ("loss", "Loss"),
        ("balanced_acc", "Balanced Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
    ]
    
    # Create epochs array for x-axis
    epochs = list(range(1, len(train_metrics.get("loss", [])) + 1))
    
    # Plot each metric
    for i, (metric_key, metric_title) in enumerate(metrics_to_plot):
        if i < len(axes):  # Make sure we don't exceed the number of subplots
            ax = axes[i]
            
            # Get the metric values, defaulting to empty list if not found
            train_values = train_metrics.get(metric_key, [])
            val_values = val_metrics.get(metric_key, [])
            
            # Plot training metrics
            if train_values:
                ax.plot(epochs, train_values, 'b-', label=f'Training {metric_title}')
                
            # Plot validation metrics (only for available validation points)
            if val_values:
                # Create x values for validation (only plot at validation intervals)
                val_epochs = list(range(1, len(val_values) + 1))
                ax.plot(val_epochs, val_values, 'r-', label=f'Validation {metric_title}')
            
            # Add title and legend
            ax.set_title(metric_title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set y-limits based on metric type
            if 'acc' in metric_key or 'precision' in metric_key or 'recall' in metric_key or 'f1' in metric_key:
                ax.set_ylim(0, 1.05)  # Accuracy metrics should be between 0 and 1
    
    # Remove any unused subplots
    for i in range(len(metrics_to_plot), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the figure if outpath is provided
    if outpath:
        fig_path = os.path.join(metrics_dir, f"metrics_progress.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        
    # Log to wandb if requested
    if send_to_wandb and (not Use_DDP or local_rank == 0):
        import wandb
        wandb.log({
            "metrics_plot": wandb.Image(fig),
            "epoch": epoch + 1
        })
        
    plt.close('all')
    return fig

def check_for_nan_or_inf(tensor, name="tensor"):
    """
    Check a tensor for NaN or Inf values and log details.
    
    Args:
        tensor: The tensor to check.
        name (str): Name of the tensor for logging.
        
    Returns:
        bool: True if NaN or Inf values are found, False otherwise
    """
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        # Ensure tensor is on CPU for min/max calculation if it might be large or on MPS etc.
        # For CUDA, it's usually fine, but CPU is safer for general case.
        tensor_cpu = tensor.detach().float().cpu() # Use float() to handle potential bfloat16 etc.
        
        # Calculate finite min/max safely
        finite_mask = ~torch.isnan(tensor_cpu) & ~torch.isinf(tensor_cpu)
        min_val = torch.min(tensor_cpu[finite_mask]) if torch.sum(finite_mask) > 0 else 'N/A'
        max_val = torch.max(tensor_cpu[finite_mask]) if torch.sum(finite_mask) > 0 else 'N/A'
        
        nan_count = torch.sum(torch.isnan(tensor_cpu)).item()
        inf_count = torch.sum(torch.isinf(tensor_cpu)).item()
        logger.error(f"!!! NaN or Inf detected in '{name}' !!!")
        logger.error(f"    NaNs: {nan_count}, Infs: {inf_count}")
        logger.error(f"    Min (finite): {min_val}, Max (finite): {max_val}")
        # Log shape as well
        logger.error(f"    Tensor shape: {tensor.shape}") 
        return True # Indicates an issue was found
    
    return False # No issues found

def train_model():
    """
    Main training loop for the Segmentation model to predict enhancement from multimodal MRI sequences.
    Removes T1CE channel (index 2) from the 4-channel input before passing to the model.
    """
    global lr_scheduler, model, optimizer, start_epoch, train_metrics, val_metrics, patience_counter, best_metric, best_metric_epoch, epoch_time_list # Make references to global variables

    # Initialize training storage
    best_metric = float("inf")
    best_metric_epoch = -1
    
    # Metrics dictionaries for training and validation
    train_metrics = {
        "loss": [],
        "balanced_acc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "dice": [], # Add dice metric
    }
    
    val_metrics = {
        "loss": [],
        "balanced_acc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "dice": [], # Add dice metric
    }
    
    # Additional training stats
    epoch_time_list = []
    patience_counter = 0
    
    # Setup gradient scaler for mixed precision training - BUT WE WILL DISABLE ITS USE FOR NOW
    use_amp = False if args.disable_amp else True
    scaler = GradScaler(enabled=use_amp) if torch.cuda.is_available() else None
    if not use_amp and local_rank == 0: # Log only once
        logger.warning("!!! Automatic Mixed Precision (AMP) is temporarily DISABLED for debugging NaNs !!!")

    if training_dataset == 'SmartCacheDataset':
        train_ds.start()
    
    # Training loop
    for epoch in range(start_epoch, n_epochs): # Ensure loop starts from start_epoch
        logger.info(f"Epoch {epoch + 1}/{n_epochs}")
        epoch_start_time = time.time()
        
        if Use_DDP and train_sampler is not None: # Check if train_sampler exists
            train_sampler.set_epoch(epoch)

        # Set model to training mode
        model.train()
        epoch_loss = 0
        epoch_balanced_acc = 0
        epoch_precision = 0
        epoch_recall = 0
        epoch_f1 = 0
        epoch_dice = 0
        
        step = 0
        
        # Determine number of steps for this epoch
        max_steps = len(train_loader) if steps_per_epoch is None else steps_per_epoch
        if local_rank == 0: # Log only once
             logger.info(f"Steps per epoch: {max_steps} (limited by parameter)" if steps_per_epoch is not None else f"Steps per epoch: {max_steps} (full dataset)")
        
        # Set up progress bar only on rank 0 or non-DDP
        if not Use_DDP or local_rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Training{' (Rank '+str(local_rank)+')' if Use_DDP else ''}", total=max_steps)
        else:
            progress_bar = train_loader  # For other ranks, just use the loader without tqdm
        
        # Single consolidated training loop for all cases
        for batch_data in progress_bar:
            
            if step >= max_steps: # Use >= to ensure exact step count
                # Stop the epoch if we've reached the steps_per_epoch limit
                break
                
            # Get input sequences and target segmentation
            sequences = batch_data["sequences"].to(device)
            segmentation = batch_data["segmentation"].to(device)
            # abnormality_segmentation = batch_data["abnormality_segmentation"].to(device)
            
            # Processing input data
            flair = sequences[:,0:1,...]
            t1 = sequences[:,1:2,...]
            # t1ce = sequences[:,2:3,...] # Not used in input
            t2 = sequences[:,3:4,...]
            coord_conv = sequences[:,4:,...]
            
            # Combine input channels
            input_channels = torch.cat([flair, t1, t2, coord_conv], dim=1)  # Results in [B, 3+3, H, W, D]
            
            if check_for_nan_or_inf(input_channels, "input_channels_before_model"):
                logger.error("NaN/Inf detected in input tensor BEFORE model. Skipping batch.")
                clear_memory()
                del input_channels, sequences, segmentation, batch_data, flair, t1, t2, coord_conv
                step += 1 # Increment step even when skipping
                continue # Skip this batch

            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass - with optional automatic mixed precision
            context = autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=use_amp) if torch.cuda.is_available() else nullcontext()
            with context:
                outputs = model(input_channels)
                
                # --- Add check right after model output ---
                if check_for_nan_or_inf(outputs, "model_output_raw"):
                    logger.error("NaN/Inf detected in raw model output. Skipping batch.")
                    clear_memory()
                    del input_channels, outputs, sequences, segmentation, batch_data, flair, t1, t2, coord_conv
                    step += 1 # Increment step even when skipping
                    continue # Skip this batch
                # --- End check ---

                # --- Clamp raw outputs ---
                # Clamp logits to prevent extreme values potentially causing issues in sigmoid (inside DiceLoss)
                outputs_clamped = torch.clamp(outputs, min=-30, max=30) 
                if check_for_nan_or_inf(outputs_clamped, "model_output_clamped"):
                     logger.error("NaN/Inf detected after clamping model output. Skipping batch.")
                     clear_memory()
                     del input_channels, outputs, outputs_clamped, sequences, segmentation, batch_data, flair, t1, t2, coord_conv
                     step += 1 # Increment step even when skipping
                     continue # Skip this batch
                # --- End clamp ---

                loss, batch_metrics, _ = calculate_enhancement_losses(
                    input_channels, outputs, segmentation, device, loss_function=loss_function, # Use clamped outputs
                )

                # --- Modify existing check ---
                nan_inf_found = False
                # input_channels already checked
                # outputs / outputs_clamped already checked
                if check_for_nan_or_inf(loss, "loss"): nan_inf_found = True
                
                if nan_inf_found:
                    logger.error("NaN/Inf detected in loss. Skipping batch.")
                    clear_memory()
                    del input_channels, outputs, outputs_clamped, loss, batch_metrics, sequences, segmentation, batch_data, flair, t1, t2, coord_conv
                    step += 1 # Increment step even when skipping
                    continue
                # --- End modification ---
                    
            # Scale the loss and compute gradients (scaler is disabled if use_amp is False)
            scaler.scale(loss).backward()
            
            # Apply gradient clipping if needed
            if gradient_clip > 0:
                # Unscale the gradients of optimizer's assigned params in-place BEFORE clipping
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_balanced_acc += batch_metrics["balanced_acc"]
            epoch_precision += batch_metrics["precision"]
            epoch_recall += batch_metrics["recall"]
            epoch_f1 += batch_metrics["f1"]
            epoch_dice += batch_metrics["dice"]
            
            # Only update progress bar for rank 0 or non-DDP
            if not Use_DDP or local_rank == 0:
                if isinstance(progress_bar, tqdm):
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"}) # Format loss
            
            # Generate sample visualizations only on rank 0 or non-DDP
            # Visualize the first batch *successfully processed* in the epoch
            if step == 0 and (not Use_DDP or local_rank == 0): # Check step == 0 here, before incrementing
                 visualize_batch_samples(
                     batch_data=batch_data,
                     outputs=outputs_clamped, # Visualize clamped outputs
                     epoch=epoch,
                     phase="training",
                     n_examples=min(10, batch_data["sequences"].shape[0]),
                     outpath=outpath,
                     send_to_wandb=send_to_wandb
                 )
            
            step += 1 # Increment step after successful processing

            clear_memory()
            # Ensure all necessary variables are deleted
            del input_channels, outputs, outputs_clamped, loss, batch_metrics, sequences, segmentation, batch_data, flair, t1, t2, coord_conv

            if Use_DDP:
                torch.cuda.synchronize() # Ensure ops complete before next iteration in DDP
            
        # Calculate epoch metrics on epoch end
        # Avoid division by zero if all batches were skipped
        if step > 0:
            epoch_loss /= step
            epoch_balanced_acc /= step
            epoch_precision /= step
            epoch_recall /= step
            epoch_f1 /= step
            epoch_dice /= step
        else:
            epoch_loss = float('nan') # Or some other indicator that no steps completed
            epoch_balanced_acc = float('nan')
            epoch_precision = float('nan')
            epoch_recall = float('nan')
            epoch_f1 = float('nan')
            epoch_dice = float('nan')
            logger.warning(f"Epoch {epoch + 1}: No batches were successfully processed.")

        
        # Combine training metrics across processes for DDP
        if Use_DDP:
            # Convert metrics to tensors on the current device
            # Handle potential NaN from division by zero
            metrics_list = [epoch_loss, epoch_balanced_acc, epoch_precision, epoch_recall, epoch_f1, epoch_dice]
            # Replace python nan with torch.nan
            metrics_list = [m if not math.isnan(m) else torch.nan for m in metrics_list] 
            train_metrics_tensor = torch.tensor(
                metrics_list,
                dtype=torch.float32, device=device
            )
            
            # All-reduce to sum metrics across all processes
            # Use nanmean equivalent: sum non-nans and count non-nans separately
            non_nan_mask = ~torch.isnan(train_metrics_tensor)
            train_metrics_tensor[torch.isnan(train_metrics_tensor)] = 0 # Replace NaN with 0 for sum
            
            torch.distributed.all_reduce(train_metrics_tensor, op=torch.distributed.ReduceOp.SUM)
            
            # Count non-NaN contributions across ranks
            non_nan_count_tensor = non_nan_mask.float()
            torch.distributed.all_reduce(non_nan_count_tensor, op=torch.distributed.ReduceOp.SUM)
            
            # Divide sum by count (avoid division by zero)
            final_metrics_tensor = train_metrics_tensor / torch.clamp(non_nan_count_tensor, min=1)

            # Extract values back from tensor
            epoch_loss = final_metrics_tensor[0].item()
            epoch_balanced_acc = final_metrics_tensor[1].item()
            epoch_precision = final_metrics_tensor[2].item()
            epoch_recall = final_metrics_tensor[3].item()
            epoch_f1 = final_metrics_tensor[4].item()
            epoch_dice = final_metrics_tensor[5].item()
            
            del train_metrics_tensor, non_nan_mask, non_nan_count_tensor, final_metrics_tensor
                
        if local_rank == 0: # Log only once
            logger.info(f"Epoch {epoch + 1} training loss: {epoch_loss:.4f}")
        
        # Log epoch metrics to wandb (handle potential NaNs)
        if send_to_wandb and (not Use_DDP or local_rank == 0):
            log_dict = {
                "train_epoch_loss": epoch_loss,
                "train_epoch_balanced_acc": epoch_balanced_acc,
                "train_epoch_precision": epoch_precision,
                "train_epoch_recall": epoch_recall,
                "train_epoch_f1": epoch_f1,
                "train_epoch_dice": epoch_dice,
                "epoch": epoch + 1 # Log epoch number with metrics
            }
            # Filter out NaN values before logging to wandb
            log_dict_filtered = {k: v for k, v in log_dict.items() if not math.isnan(v)}
            wandb.log(log_dict_filtered)
        
        # Store epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_time_list.append(epoch_time)
        if local_rank == 0: # Log only once
            logger.info(f"Epoch {epoch + 1} took {epoch_time:.2f}s")
        
        # Store metrics (even if NaN)
        train_metrics['loss'].append(epoch_loss)
        train_metrics['balanced_acc'].append(epoch_balanced_acc)
        train_metrics['precision'].append(epoch_precision)
        train_metrics['recall'].append(epoch_recall)
        train_metrics['f1'].append(epoch_f1)
        train_metrics['dice'].append(epoch_dice)
        
        # Save metrics CSV only on rank 0
        if local_rank == 0:
            csv_path = save_metrics_to_csv(outpath=outpath,train_metrics=train_metrics,val_metrics=val_metrics)

        # Validation loop - perform validation at defined intervals
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_loss = 0
            val_step = 0
            val_balanced_acc = 0
            val_precision = 0
            val_recall = 0
            val_f1 = 0
            val_dice = 0
            
            # Initialize list to collect per-sample metrics for the epoch
            epoch_val_sample_metrics = []
            
            # Determine max validation steps
            val_max_steps = len(val_loader) # Use full validation set unless steps_per_epoch is set
            if steps_per_epoch is not None:
                 val_max_steps = max(1, steps_per_epoch // 2) # Or some fraction if training steps are limited
            
            # Create directories for saving predictions
            if not Use_DDP or local_rank == 0:
                # Directory for all validation predictions for this epoch
                val_predictions_dir = os.path.join(outpath, "cached_val_predictions")
                os.makedirs(val_predictions_dir, exist_ok=True)
                logger.info(f"Will save validation predictions to {val_predictions_dir}")
                
                # Directory for best model predictions (only used if new best model)
                best_predictions_dir = os.path.join(outpath, "best_predictions")
                os.makedirs(best_predictions_dir, exist_ok=True)
            
            with torch.no_grad():
                # Set up progress bar only on rank 0 or non-DDP, similar to training loop
                if not Use_DDP or local_rank == 0:
                    val_bar = tqdm(val_loader, desc=f"Validation{' (Rank '+str(local_rank)+')' if Use_DDP else ''}", total=val_max_steps)
                else:
                    val_bar = val_loader  # For other ranks, just use the loader without tqdm
                    
                val_batch_idx = 0 # Manual index tracking for step limit
                for val_batch in val_bar:
                    
                    if val_batch_idx >= val_max_steps: # Check step limit
                        break
                    
                    # Get input sequences and target segmentation
                    val_sequences = val_batch["sequences"].to(device)
                    val_segmentation = val_batch["segmentation"].to(device)
                    
                    # Extract the channels (FLAIR, T1, T2) - excluding T1CE (at index 2)
                    val_flair = val_sequences[:, 0:1, ...]
                    val_t1 = val_sequences[:, 1:2, ...]
                    val_t2 = val_sequences[:, 3:4, ...]
                    coord_conv = val_sequences[:,4:,...]
                    
                    # Combine the input channels
                    val_input_channels = torch.cat([val_flair, val_t1, val_t2, coord_conv], dim=1)

                    # --- Check validation input ---
                    if check_for_nan_or_inf(val_input_channels, "val_input_channels_before_inference"):
                        logger.error("NaN/Inf detected in validation input tensor BEFORE inference. Skipping batch.")
                        clear_memory()
                        del val_input_channels, val_sequences, val_segmentation, val_batch, val_flair, val_t1, val_t2, coord_conv
                        val_batch_idx += 1 # Increment index even when skipping
                        continue # Skip this validation batch                    # --- End check ---
                    
                    # Use the same context manager for consistency (AMP disabled here too)
                    context = autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=use_amp) if torch.cuda.is_available() else nullcontext()
                    with context:
                        # Forward pass using sliding window inference
                        val_outputs = sliding_window_inference(
                            inputs=val_input_channels,
                            roi_size=(args.xdim, args.ydim, args.zdim),
                            sw_batch_size=args.sw_batch_size,
                            predictor=model,
                            overlap=args.sw_overlap,
                            mode="gaussian", 
                            padding_mode="replicate",
                            device=device,  # Use the current GPU device
                            progress=False # Disable progress bar inside sliding window if desired
                        )

                        # --- Check validation output ---
                        if check_for_nan_or_inf(val_outputs, "val_output_raw"):
                            logger.error("NaN/Inf detected in raw validation output. Skipping batch.")
                            clear_memory()
                            del val_input_channels, val_outputs, val_sequences, val_segmentation, val_batch, val_flair, val_t1, val_t2, coord_conv
                            val_batch_idx += 1 # Increment index even when skipping
                            continue # Skip this validation batch
                        # --- End check ---

                        # --- Clamp validation output ---
                        val_outputs_clamped = torch.clamp(val_outputs, min=-30, max=30)
                        if check_for_nan_or_inf(val_outputs_clamped, "val_output_clamped"):
                            logger.error("NaN/Inf detected after clamping validation output. Skipping batch.")
                            clear_memory()
                            del val_input_channels, val_outputs, val_outputs_clamped, val_sequences, val_segmentation, val_batch, val_flair, val_t1, val_t2, coord_conv
                            val_batch_idx += 1 # Increment index even when skipping
                            continue # Skip this validation batch
                        # --- End clamp ---
                        
                        # Pass val_batch["segmentation"] for filenames and is_validation=True
                        val_batch_loss, val_batch_metrics, individual_metrics = calculate_enhancement_losses(
                            val_input_channels, val_outputs, val_segmentation, device, 
                            loss_function=loss_function, 
                            filenames=val_batch["segmentation"], 
                            is_validation=True
                        )
                        
                        # Store individual metrics for later saving - add epoch number to each sample's metrics
                        if individual_metrics and (not Use_DDP or local_rank == 0):
                            # Add epoch number to each individual metric
                            for metric_dict in individual_metrics:
                                metric_dict['epoch'] = epoch + 1
                            
                            # Extend the overall list with this batch's metrics
                            epoch_val_sample_metrics.extend(individual_metrics)
                        
                        # --- Save prediction outputs as NIFTI files ---
                        if (not Use_DDP or local_rank == 0):
                            # Get sigmoid of outputs for probability map
                            val_outputs_sigmoid = torch.sigmoid(val_outputs_clamped)
                            
                            # Process and save each sample in the batch
                            for b in range(val_outputs_sigmoid.shape[0]):
                                # Get the metadata for this sample from the batch
                                sample_metadata = {
                                    'filename_or_obj': val_batch["segmentation"].meta["filename_or_obj"][b],
                                    'original_affine': val_batch["segmentation"].meta["original_affine"][b],
                                }
                                
                                # Extract single sample from batch 
                                sample_output_sigmoid = val_outputs_sigmoid[b:b+1]  # Keep channel dim
                                
                                try:
                                    # Save probability map (non-binary)
                                    save_prediction_as_nifti(
                                        image_tensor=sample_output_sigmoid,
                                        metadata=sample_metadata,
                                        output_dir=val_predictions_dir,
                                        filename_prefix="prob",
                                        binary=False  # Save raw probabilities
                                    )
                                    
                                    # Save binary prediction
                                    save_prediction_as_nifti(
                                        image_tensor=sample_output_sigmoid,
                                        metadata=sample_metadata,
                                        output_dir=val_predictions_dir,
                                        filename_prefix="bin",
                                        binary=True  # Apply threshold to get binary mask
                                    )
                                except Exception as e:
                                    logger.warning(f"Error saving prediction NIFTI for sample {b}: {str(e)}")
                        # --- End save prediction outputs ---

                    if Use_DDP:
                        torch.cuda.synchronize() # Ensure ops complete before next iteration in DDP
                    
                    # Update validation metrics
                    val_loss += val_batch_loss.item()
                    val_balanced_acc += val_batch_metrics["balanced_acc"]
                    val_precision += val_batch_metrics["precision"]
                    val_recall += val_batch_metrics["recall"]
                    val_f1 += val_batch_metrics["f1"]
                    val_dice += val_batch_metrics["dice"]
                    val_step += 1 # Increment successful step count

                    if not Use_DDP or local_rank == 0:
                         if isinstance(val_bar, tqdm):
                             val_bar.set_postfix({"val_loss": f"{val_batch_loss.item():.4f}"}) # Format loss
                    
                    # Generate sample visualizations for the first *successfully processed* batch of validation
                    if val_step == 1 and (not Use_DDP or local_rank == 0): # Check local_rank for DDP
                        visualize_batch_samples(
                            batch_data=val_batch,
                            outputs=val_outputs_clamped, # Visualize clamped outputs
                            epoch=epoch,
                            phase="validation",
                            n_examples=min(10, val_batch["sequences"].shape[0]),
                            outpath=outpath,
                            send_to_wandb=send_to_wandb
                        )
                        
                    val_batch_idx += 1 # Increment overall batch index

                    clear_memory()
                    # Ensure all necessary validation variables are deleted
                    del val_input_channels, val_outputs, val_outputs_clamped, val_batch_loss, val_batch_metrics, val_sequences, val_segmentation, val_batch, val_flair, val_t1, val_t2, coord_conv

            # Calculate validation metrics
            # Avoid division by zero if all batches were skipped
            if val_step > 0:
                val_loss /= val_step
                val_balanced_acc /= val_step
                val_precision /= val_step
                val_recall /= val_step
                val_f1 /= val_step
                val_dice /= val_step
            else:
                val_loss = float('nan')
                val_balanced_acc = float('nan')
                val_precision = float('nan')
                val_recall = float('nan')
                val_f1 = float('nan')
                val_dice = float('nan')
                logger.warning(f"Epoch {epoch + 1}: No validation batches were successfully processed.")

            
            # Combine validation metrics across processes for DDP
            if Use_DDP:
                # Convert metrics to tensors on the current device (handle potential NaNs)
                val_metrics_list = [val_loss, val_balanced_acc, val_precision, val_recall, val_f1, val_dice]
                val_metrics_list = [m if not math.isnan(m) else torch.nan for m in val_metrics_list]
                val_metrics_tensor = torch.tensor(
                    val_metrics_list,
                    dtype=torch.float32, device=device
                )
                
                # All-reduce using nanmean logic
                val_non_nan_mask = ~torch.isnan(val_metrics_tensor)
                val_metrics_tensor[torch.isnan(val_metrics_tensor)] = 0
                torch.distributed.all_reduce(val_metrics_tensor, op=torch.distributed.ReduceOp.SUM)
                
                val_non_nan_count_tensor = val_non_nan_mask.float()
                torch.distributed.all_reduce(val_non_nan_count_tensor, op=torch.distributed.ReduceOp.SUM)
                
                # Divide sum by count (avoid division by zero)
                final_val_metrics_tensor = val_metrics_tensor / torch.clamp(val_non_nan_count_tensor, min=1)

                # Extract values back from tensor
                val_loss = final_val_metrics_tensor[0].item()
                val_balanced_acc = final_val_metrics_tensor[1].item()
                val_precision = final_val_metrics_tensor[2].item()
                val_recall = final_val_metrics_tensor[3].item()
                val_f1 = final_val_metrics_tensor[4].item()
                val_dice = final_val_metrics_tensor[5].item()
                del val_metrics_tensor, val_non_nan_mask, val_non_nan_count_tensor, final_val_metrics_tensor
            
            if local_rank == 0: # Log only once
                logger.info(f"Epoch {epoch + 1} validation loss: {val_loss:.4f}")
            
            # Log validation metrics to wandb (handle potential NaNs)
            if send_to_wandb and (not Use_DDP or local_rank == 0):
                val_log_dict = {
                    "val_loss": val_loss,
                    "val_balanced_acc": val_balanced_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_dice": val_dice,  # Add Dice metric
                    "epoch": epoch + 1 # Log epoch number
                }
                val_log_dict_filtered = {k: v for k, v in val_log_dict.items() if not math.isnan(v)}
                wandb.log(val_log_dict_filtered)
                
            # Store validation metrics (even if NaN)
            val_metrics['loss'].append(val_loss)
            val_metrics['balanced_acc'].append(val_balanced_acc)
            val_metrics['precision'].append(val_precision)
            val_metrics['recall'].append(val_recall)
            val_metrics['f1'].append(val_f1)
            val_metrics['dice'].append(val_dice)

            # Save per-sample metrics to CSV
            if local_rank == 0 and epoch_val_sample_metrics:
                try:
                    csv_path = os.path.join(outpath, "validation_sample_metrics.csv")
                    
                    # Create DataFrame from current epoch's metrics
                    df_epoch = pd.DataFrame(epoch_val_sample_metrics)
                    # Ensure epoch is included in metrics
                    if 'epoch' not in df_epoch.columns:
                        df_epoch['epoch'] = epoch + 1
                    
                    # Reorder columns for clarity, with epoch first
                    columns = ['epoch', 'filename', 'precision', 'recall', 'f1', 'dice', 'balanced_acc']
                    df_epoch = df_epoch[columns]
                    
                    # If file exists, append without header; otherwise create new file with header
                    if os.path.exists(csv_path):
                        df_epoch.to_csv(csv_path, mode='a', header=False, index=False)
                        logger.info(f"Appended {len(df_epoch)} sample metrics for epoch {epoch+1} to {csv_path}")
                    else:
                        df_epoch.to_csv(csv_path, mode='w', header=True, index=False)
                        logger.info(f"Created new metrics file with {len(df_epoch)} samples for epoch {epoch+1}: {csv_path}")
                except Exception as e:
                    logger.error(f"Failed to save validation sample metrics: {e}")
                    logger.error(traceback.format_exc())

            # Update learning rate scheduler (only step if val_loss is not NaN)
            if not math.isnan(val_loss):
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                # No else needed for CosineAnnealingLR as it steps based on epoch, not metric
            else:
                 logger.warning(f"Epoch {epoch + 1}: Skipping LR scheduler step due to NaN validation loss.")
            
            # Step CosineAnnealingLR regardless of validation loss (based on epoch)
            if not isinstance(lr_scheduler, ReduceLROnPlateau):
                 lr_scheduler.step() # Step per epoch after validation

            # Check for best model (only if val_loss is not NaN)
            if not math.isnan(val_loss) and val_loss < best_metric:
                best_metric = val_loss
                best_metric_epoch = epoch + 1
                
                # Save best model - Handle DP/DDP model structure
                model_state_to_save = model.module.state_dict() if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict()
                
                # Only save on rank 0 for DDP, or always for DP/Single GPU
                if local_rank == 0:
                    model_save_path = os.path.join(outpath, "best_model.pth")
                    # Save optimizer and scheduler state dicts as well
                    scheduler_state = lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else None
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model_state_to_save, # Save unwrapped state
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler_state,
                        "best_metric": best_metric,
                        "val_loss": val_loss, # Save the specific loss that triggered the save
                        # Save full metrics history only on rank 0
                        "train_metrics": train_metrics, 
                        "val_metrics": val_metrics,     
                        "epoch_time_list": epoch_time_list, 
                        "args": vars(args), # Save args used for this run
                        "wandb_run_id": wandb_run_id # Save wandb run ID
                    }, model_save_path)
                    logger.info(f"New best metric model saved at epoch {best_metric_epoch} with loss {best_metric:.4f}")
                    
                    # Copy prediction files from val_predictions_dir to best_predictions_dir
                    logger.info(f"Copying predictions from best model (epoch {best_metric_epoch}) to {best_predictions_dir}")
                    try:
                        # Clear the best_predictions_dir first
                        for file in glob.glob(os.path.join(best_predictions_dir, "*.nii.gz")):
                            os.remove(file)
                        
                        # Copy all nifti files from val_predictions_dir to best_predictions_dir
                        for file in glob.glob(os.path.join(val_predictions_dir, "*.nii.gz")):
                            shutil.copy2(file, best_predictions_dir)
                        logger.info(f"Successfully copied {len(glob.glob(os.path.join(best_predictions_dir, '*.nii.gz')))} prediction files to best_predictions directory")
                    except Exception as e:
                        logger.error(f"Error copying prediction files to best_predictions directory: {str(e)}")
                
                # Reset patience counter
                patience_counter = 0
            elif not math.isnan(val_loss): # Only increment patience if validation was successful
                # Increment patience counter
                patience_counter += 1
                if local_rank == 0: # Log only once
                    logger.info(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")
                
                # Early stopping check
                if patience_counter >= patience:
                    if local_rank == 0: # Log only once
                        logger.info(f"Early stopping triggered after epoch {epoch + 1}")
                    break # Exit the training loop
            else:
                 # Don't increment patience if validation failed (NaN loss)
                 if local_rank == 0:
                     logger.warning(f"Epoch {epoch + 1}: Skipping patience update due to NaN validation loss.")

        
        # Save regular checkpoint (only on rank 0)
        if args.save_checkpoints and (epoch + 1) % 10 == 0 and local_rank == 0:
             # Handle DP/DDP model structure
            model_state_to_save = model.module.state_dict() if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict()
            scheduler_state = lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else None
            
            checkpoint_path = os.path.join(outpath, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_state_to_save, # Save unwrapped state
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler_state,
                # Save full metrics history only on rank 0
                "train_metrics": train_metrics, 
                "val_metrics": val_metrics,     
                "epoch_time_list": epoch_time_list, 
                "patience_counter": patience_counter,
                "best_metric": best_metric,
                "best_metric_epoch": best_metric_epoch,
                "args": vars(args), # Save args used for this run
                "wandb_run_id": wandb_run_id # Save wandb run ID
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch + 1}")
            
        # Clear memory at the end of the epoch
        clear_memory()
        if training_dataset == 'SmartCacheDataset':
            gc.collect()
            # Check if train_ds exists and has update_cache method
            if 'train_ds' in globals() and hasattr(train_ds, 'update_cache'):
                # Only update cache on rank 0? Or all ranks? Check SmartCache docs/behavior
                # Assuming update_cache needs to be called by all ranks using it
                train_ds.update_cache() 
            gc.collect()
            
        # Break the loop if early stopping was triggered in the validation section
        if patience_counter >= patience:
            break

    # End of training
    if local_rank == 0: # Log only once
        logger.info(f"Training completed. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
    
    # Save metrics to CSV (rank 0)
    if local_rank == 0:
        csv_path = save_metrics_to_csv(
            outpath=outpath,
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )
        if csv_path: # Check if saving was successful
             logger.info(f"Metrics saved to {csv_path}")
    
    # Create metrics visualization after training (rank 0)
    if local_rank == 0:
        try:
            # Ensure epoch number is correct (last completed epoch)
            final_epoch_idx = epoch # Use the last value of epoch from the loop
            plot_metrics(
                train_metrics=train_metrics, 
                val_metrics=val_metrics, 
                epoch=final_epoch_idx, # Use the actual last epoch index
                outpath=outpath, 
                send_to_wandb=send_to_wandb
            )
            logger.info("Final metrics visualization generated successfully")
        except Exception as e:
            logger.warning(f"Failed to generate final metrics visualization: {str(e)}")
            logger.warning(traceback.format_exc()) # Log traceback for plotting error
    
    return model, best_metric, best_metric_epoch, train_metrics, val_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="T1CE Enhancement Prediction")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=6) ##reduce this to 1 if SwinUNETR #6 is OK for SegResNet
    parser.add_argument("--val_batch_size", type=int, default=8, ###28 for segresnet xx2 4090 
                        help="Batch size for validation.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--gradient_clip", type=float, default=10.0)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "plateau"],
                        help="Learning rate scheduler type: 'cosine' or 'plateau'")
    parser.add_argument("--steps_per_epoch", type=int, default=None, 
                        help="Limit number of steps per epoch. If None, use full dataset length.")
    parser.add_argument("--continue_training", action="store_true",
                        help="Continue training from the last checkpoint if available")
    parser.add_argument("--disable_amp", action="store_true",
                        help="Disable automatic mixed precision (AMP) training")
    
    # Data loading parameters
    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of worker processes for data loading")
    parser.add_argument('--training_dataset', type=str, default='Dataset', 
                       choices=['Dataset', 'CacheDataset', 'SmartCacheDataset'], 
                       help='Type of dataset implementation to use')
    parser.add_argument('--smart_cache_replace_rate', type=float, default=0.5, 
                       help='Replace rate for SmartCacheDataset (fraction of samples to replace)')
    parser.add_argument('--smart_cache_num', type=float, default=0.003, 
                       help='Fraction of dataset to cache in SmartCacheDataset')
    parser.add_argument('--cache_val_dataset', action='store_true',
                       help='Cache the validation dataset')
    parser.add_argument('--persistent_val_dataset', action='store_true')
    
    # Data parameters
    parser.add_argument("--sequences_dir", type=str, default="/home/jruffle/Documents/seq-synth/data/sequences_merged")
    parser.add_argument("--segmentations_dir", type=str, default="/home/jruffle/Documents/seq-synth/data/enhancement_masks")
    parser.add_argument("--abnormality_seg_dir", type=str, default="/home/jruffle/Documents/seq-synth/data/lesion_masks_augmented")

    parser.add_argument("--xdim", type=int, default=128)
    parser.add_argument("--ydim", type=int, default=128)
    parser.add_argument("--zdim", type=int, default=128)
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="segresnet", choices=["segresnet", "swinunetr"])
    
    # Augmentation parameters
    parser.add_argument("--augs", action="store_true")
    parser.add_argument("--aug_prob", type=float, default=0.2)
    
    # Distribution parameters - Make DDP and DP mutually exclusive
    dist_group = parser.add_mutually_exclusive_group()
    dist_group.add_argument("--Use_DDP", action="store_true", help="Use Distributed Data Parallel (DDP)")
    dist_group.add_argument("--use_dp", action="store_true", help="Use Data Parallel (DP)")
    
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DDP (set automatically by torchrun)")
    
    # Logging parameters
    parser.add_argument("--outpath", type=str, default="/home/jruffle/Documents/seq-synth/enhancement_predictor")
    parser.add_argument("--send_to_wandb", action="store_true")
    parser.add_argument("--wandb_location", type=str, default="online")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_checkpoints", action="store_true", 
                       help="Save checkpoint models every 10 epochs. Best model is always saved regardless of this setting.")
    
    # Add sliding window parameters to argparse
    parser.add_argument("--sw_batch_size", type=int, default=20,
                        help="Batch size for sliding window inference") ##10 for segresnet
    parser.add_argument("--sw_overlap", type=float, default=0.5,
                        help="Overlap factor for sliding window inference")
    
    # Ablation study parameters
    parser.add_argument("--ablation_study", action="store_true",
                       help="Run ablation study with different input channel combinations")
    parser.add_argument("--ablation_output", type=str, default=None,
                       help="Output directory for ablation study results")

    
    args = parser.parse_args()
    
    # Update global Use_DDP based on environment if torchrun is used
    if "LOCAL_RANK" in os.environ:
        args.Use_DDP = True # Force DDP if launched with torchrun
        if args.use_dp:
            logger.warning("WARNING: Both DDP and DP flags were set. Since LOCAL_RANK is in environment, DDP will be used.")
            args.use_dp = False
            
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set global variables from args
    for k, v in vars(args).items():
        # Handle boolean flags correctly
        if isinstance(v, bool):
            globals()[k] = v
        else:
            globals()[k] = v if v is not None else globals().get(k) # Keep default if arg is None

    # Set up training mode (debug vs production)
    setup_training_mode(debug)
        
    # Set up distributed training or DP if needed
    if Use_DDP:
        # Configure NCCL parameters to handle potential deadlocks
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_TIMEOUT"] = "180"  # 3 minutes timeout instead of 10
        
        # Initialize process group with timeout settings
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(minutes=30),  # 30 minutes timeout
            )
        local_rank = int(os.environ["LOCAL_RANK"])

        is_rtx_6000_ada = False
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "RTX 6000 Ada" in device_name:
                is_rtx_6000_ada = True

        if is_rtx_6000_ada:
            os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable P2P communication for Ada
            
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        logger.info(f"Using DDP. Rank {local_rank} assigned to device {device}")
    elif use_dp:
        # DP uses cuda:0 as the main device, but will utilize multiple GPUs
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
             logger.warning("DP requires multiple GPUs. Falling back to single GPU.")
             use_dp = False # Disable DP if conditions not met
             device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
             local_rank = 0 # Ensure local_rank is 0 for single GPU fallback
        else:
             device = torch.device("cuda:0") # Main device for DP
             logger.info(f"Using DP across {torch.cuda.device_count()} GPUs. Main device: {device}")
             local_rank = 0 # DP runs conceptually on rank 0
    else:
        # Single GPU or CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using single device: {device}")
        local_rank = 0 # Ensure local_rank is 0 for single device

    # Create output directory (only needed once)
    # Use local_rank which defaults to 0 if not DDP
    if local_rank == 0: 
        os.makedirs(outpath, exist_ok=True)
        
    # Initialize wandb_run_id if it will be referenced later
    wandb_run_id = None
        
    # Instantiate model
    if args.model_type == 'segresnet':
        model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=3+3, # 3 channels (FLAIR, T1, T2) + 3 coordconv
        out_channels=1,
        dropout_prob=0.1,
        ).to(device)
    elif args.model_type == 'swinunetr':
        model = SwinUNETR(
        img_size=(xdim, ydim, zdim),
        in_channels=3+3, # 3 channels (FLAIR, T1, T2) + 3 coordconv
        out_channels=1,
        feature_size=48,
        use_checkpoint=False,
        drop_rate=0.1, 
        attn_drop_rate=0.1, 
        dropout_path_rate=0.1,
    ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    model = model.to(device) # Move model to the primary device first
    
    # Wrap model based on parallel mode
    if Use_DDP:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank) # Specify output_device for clarity
        logger.info(f"Model wrapped with DDP on rank {local_rank}")
    elif use_dp:
        model = nn.DataParallel(model) # Default uses all available GPUs
        logger.info(f"Model wrapped with DP across {torch.cuda.device_count()} GPUs.")
        
    loss_function = DiceLoss(to_onehot_y=False, sigmoid=True) # Increased smoothing
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-5) # Use args.lr
    # Use ReduceLROnPlateau if specified, otherwise CosineAnnealingLR
    if args.lr_scheduler == 'plateau':
         lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True if local_rank==0 else False) # verbose only on rank 0
    else: # Default to CosineAnnealingLR
         # Make T_max the total number of epochs
         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs) # Use args.n_epochs

    # Load checkpoint if continue_training is enabled
    start_epoch = 0
    # Initialize metrics/patience here in case checkpoint doesn't have them
    # Ensure keys match those used in train_model
    train_metrics = {"loss": [], "balanced_acc": [], "precision": [], "recall": [], "f1": []}
    val_metrics = {"loss": [], "balanced_acc": [], "precision": [], "recall": [], "f1": []}
    patience_counter = 0
    best_metric = float('inf')
    best_metric_epoch = -1
    epoch_time_list = []
    checkpoint_to_load = None # Initialize checkpoint_to_load to None by default

    if args.continue_training: # Use args.continue_training
        # First load CSV metrics and get max epoch
        csv_result = save_metrics_to_csv(
            outpath=outpath, 
            train_metrics=train_metrics, 
            val_metrics=val_metrics, 
            continue_training=True
        )
        
        # Extract max epoch from csv_result tuple if available
        if isinstance(csv_result, tuple) and len(csv_result) == 2:
            _, max_epoch_from_csv = csv_result
            if max_epoch_from_csv > 0:
                # Update start_epoch based on the CSV data
                logger.info(f"Setting start_epoch to {max_epoch_from_csv} based on CSV metrics file")
                start_epoch = max_epoch_from_csv
                
        # --- DDP Checkpoint Discovery (Rank 0 finds, broadcasts path) ---
        if Use_DDP:
            checkpoint_path_list = [None] # Use list to pass by reference for broadcast_object_list
            if local_rank == 0:
                checkpoint_files = sorted(glob.glob(os.path.join(outpath, "checkpoint_epoch_*.pth")))
                best_model_path = os.path.join(outpath, "best_model.pth")
                
                # Prioritize latest checkpoint, then best model
                if checkpoint_files:
                    checkpoint_to_load = checkpoint_files[-1] # Load latest checkpoint
                    logger.info(f"Found latest checkpoint: {checkpoint_to_load}")
                elif os.path.exists(best_model_path):
                    checkpoint_to_load = best_model_path
                    logger.info(f"Found best model checkpoint: {checkpoint_to_load}")
                else:
                    logger.info("No checkpoint or best model found to continue from.")
                    checkpoint_to_load = None # Explicitly set to None
                
                checkpoint_path_list[0] = checkpoint_to_load # Put path in the list for broadcasting

            # Broadcast the path object (string or None) from rank 0 to all other ranks
            torch.distributed.broadcast_object_list(checkpoint_path_list, src=0)
            checkpoint_to_load = checkpoint_path_list[0] # Retrieve path from list

            # Barrier to ensure all ranks have the path before proceeding
            torch.distributed.barrier()

        # --- DP / Single GPU Checkpoint Discovery ---
        else: # Not DDP (DP or Single GPU)
            checkpoint_files = sorted(glob.glob(os.path.join(outpath, "checkpoint_epoch_*.pth")))
            best_model_path = os.path.join(outpath, "best_model.pth")
            if checkpoint_files:
                checkpoint_to_load = checkpoint_files[-1] # Load latest checkpoint
                logger.info(f"Found latest checkpoint: {checkpoint_to_load}")
            elif os.path.exists(best_model_path):
                checkpoint_to_load = best_model_path
                logger.info(f"Found best model checkpoint: {checkpoint_to_load}")
            else:
                logger.info("No checkpoint or best model found to continue from.")
                checkpoint_to_load = None

        # --- Common Loading Logic (All Ranks for DDP, only rank 0 effectively for DP/Single) ---
        # Check if checkpoint_to_load is a non-empty string or path object
        if checkpoint_to_load and os.path.exists(checkpoint_to_load):
            logger.info(f"Rank {local_rank} loading checkpoint: {checkpoint_to_load}")
            # Load checkpoint onto the correct device
            checkpoint = torch.load(checkpoint_to_load, map_location=device)
            
            # Load model state - handles potential 'module.' prefix mismatch
            ckpt_state_dict = checkpoint["model_state_dict"]
            model_to_load = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
            
            try:
                 model_to_load.load_state_dict(ckpt_state_dict)
                 logger.info(f"Rank {local_rank}: Successfully loaded model state_dict.")
            except RuntimeError as e:
                 logger.warning(f"Rank {local_rank}: State dict mismatch, attempting fix: {e}")
                 # Try adding/removing 'module.' prefix
                 new_state_dict = {}
                 is_ddp_dp_checkpoint = list(ckpt_state_dict.keys())[0].startswith('module.')
                 is_current_model_wrapped = isinstance(model, (DDP, nn.DataParallel))

                 if is_ddp_dp_checkpoint and not is_current_model_wrapped:
                     # Strip 'module.' from checkpoint keys
                     for k, v in ckpt_state_dict.items():
                         new_state_dict[k.replace('module.', '')] = v
                     logger.info(f"Rank {local_rank}: Stripped 'module.' prefix from checkpoint keys.")
                 elif not is_ddp_dp_checkpoint and is_current_model_wrapped:
                     # Add 'module.' to checkpoint keys
                     for k, v in ckpt_state_dict.items():
                         new_state_dict[f'module.{k}'] = v
                     logger.info(f"Rank {local_rank}: Added 'module.' prefix to checkpoint keys.")
                 else:
                      # No obvious fix, re-raise or log error
                      logger.error(f"Rank {local_rank}: Could not automatically fix state dict mismatch.")
                      new_state_dict = ckpt_state_dict # Keep original to see load error again

                 try:
                     model_to_load.load_state_dict(new_state_dict, strict=False) # Use strict=False initially
                     logger.info(f"Rank {local_rank}: Successfully loaded model state_dict after key adjustment (strict=False).")
                 except Exception as load_err:
                     logger.error(f"Rank {local_rank}: Failed to load state_dict even after adjustment: {load_err}")
                     # Consider raising error or exiting if model load fails critically


            # Load optimizer and scheduler state (only needed on rank 0 for DDP logic/logging)
            # It's safer to load on all ranks, especially for schedulers like CosineAnnealingLR
            try:
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    logger.info(f"Rank {local_rank}: Loaded optimizer state_dict.")
                else:
                    logger.warning(f"Rank {local_rank}: Optimizer state_dict not found in checkpoint.")

                if "scheduler_state_dict" in checkpoint and hasattr(lr_scheduler, "load_state_dict"):
                    # Check for T_max mismatch in CosineAnnealingLR before loading
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                        logger.info(f"Current T_max: {lr_scheduler.T_max}, handling scheduler loading")
                    
                    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    logger.info(f"Rank {local_rank}: Loaded LR scheduler state_dict.")
                elif hasattr(lr_scheduler, "load_state_dict"):
                     logger.warning(f"Rank {local_rank}: LR scheduler state_dict not found in checkpoint.")

            except Exception as opt_load_err:
                 logger.warning(f"Rank {local_rank}: Error loading optimizer/scheduler state: {opt_load_err}")

            start_epoch = checkpoint.get("epoch", -1) + 1 # Use .get for safety
            
            # Load metric history, patience, etc. (only needed on rank 0 for DDP logic/logging)
            if local_rank == 0:
                train_metrics = checkpoint.get("train_metrics", train_metrics)
                val_metrics = checkpoint.get("val_metrics", val_metrics)
                patience_counter = checkpoint.get("patience_counter", patience_counter)
                best_metric = checkpoint.get("best_metric", best_metric)
                best_metric_epoch = checkpoint.get("best_metric_epoch", best_metric_epoch)
                epoch_time_list = checkpoint.get("epoch_time_list", epoch_time_list)
                
                # Get wandb_run_id from checkpoint for continuing training with wandb
                if send_to_wandb and "wandb_run_id" in checkpoint:
                    wandb_run_id = checkpoint.get("wandb_run_id")
                    logger.info(f"Found wandb run ID in checkpoint: {wandb_run_id}")

            logger.info(f"Rank {local_rank} resuming training from epoch {start_epoch}")
        else:
            if local_rank == 0: # Log only once
                logger.info(f"Rank {local_rank} starting from scratch (no valid checkpoint found or specified)")
            start_epoch = 0 # Ensure start_epoch is 0 if not loading

        # Barrier to ensure all DDP processes have loaded before starting
        if Use_DDP:
            torch.distributed.barrier()
            
    # Setup wandb after loading checkpoint (to get the wandb_run_id if needed)
    if send_to_wandb and local_rank == 0:
        if args.continue_training and wandb_run_id:
            logger.info(f"Resuming wandb run with ID: {wandb_run_id}")
            wandb.init(
                project="t1ce-enhancement-prediction",
                id=wandb_run_id,
                resume="must",
                config=vars(args),
                mode=wandb_location
            )
        else:
            if run_name is None:
                run_name = f"enhancement_pred_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project="t1ce-enhancement-prediction",
                name=run_name,
                config=vars(args),
                mode=wandb_location
            )
            # Store the new run ID for future resume
            wandb_run_id = wandb.run.id
            
    # ... Initialize metrics tracking (already done before checkpoint loading) ...

    # Setup data loaders
    train_loader, val_loader, train_sampler, val_sampler, train_ds = setup_data()
    
    # ... T_max calculation (already handled in scheduler init) ...

    # ... validation interval check (redundant) ...

    # Import traceback for error logging
    import traceback 

    try:
        # Train model (pass start_epoch if needed by train_model)
        # Assuming train_model uses the global start_epoch
        train_model() 
    except Exception as e:
        logger.error(f"Rank {local_rank}: Training failed: {str(e)}")
        # Optionally log traceback
        logger.error(f"Rank {local_rank}: Traceback:\n{traceback.format_exc()}")
        # Ensure cleanup happens even on error
    finally:
        # Cleanup
        if Use_DDP:
            # Ensure synchronization before cleanup
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
        # Each rank needs to clean up its own dataset if it's SmartCache
        # Check if train_ds was assigned and is SmartCache
        if 'train_ds' in globals() and isinstance(train_ds, SmartCacheDataset):
            logger.info(f"Rank {local_rank}: Shutting down SmartCacheDataset")
            try:
                train_ds.shutdown()
            except Exception as shutdown_err:
                logger.error(f"Rank {local_rank}: Error shutting down SmartCacheDataset: {str(shutdown_err)}")

        if Use_DDP and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            
        if send_to_wandb and local_rank == 0: # Finish wandb only on rank 0
            try:
                wandb.finish()
            except:
                pass
            
        # Clear any remaining GPU memory
        clear_memory()
        logger.info(f"Rank {local_rank}: Cleanup finished.")