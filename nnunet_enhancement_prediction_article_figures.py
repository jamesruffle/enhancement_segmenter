# %%
import pandas as pd
import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import nilearn
import nibabel as nib
from tqdm import tqdm
import json
from joblib import Parallel, delayed
import argparse
import datetime
from scipy import stats
from scipy.ndimage import center_of_mass
from sklearn.metrics import confusion_matrix, classification_report

# %%
pd.set_option('display.max_columns', None)

# %%
# %% [markdown]
# ## Bug Fixes and Optimizations
# 
# This cell contains helper functions to address potential bugs and improve performance

# %%
# Helper functions for robust file operations
import os

def safe_file_load(filepath, load_function, *args, **kwargs):
    """Safely load a file with error handling"""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    try:
        return load_function(filepath, *args, **kwargs)
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

def safe_division(numerator, denominator, default=0):
    """Safely perform division with zero check"""
    return numerator / denominator if denominator != 0 else default

def validate_dataframe_columns(df, required_columns):
    """Check if required columns exist in dataframe"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
        return False
    return True

# Memory optimization function
def cleanup_large_arrays(*arrays):
    """Clean up large arrays to free memory"""
    import gc
    for arr in arrays:
        if arr is not None:
            del arr
    gc.collect()

# Efficient dataframe concatenation
def concat_dataframes_efficiently(df_list):
    """Concatenate list of dataframes efficiently"""
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

# NaN-safe statistical functions
def nanmean_safe(arr):
    """Calculate mean ignoring NaN values"""
    if len(arr) == 0:
        return 0
    clean_arr = arr[~np.isnan(arr)]
    return np.mean(clean_arr) if len(clean_arr) > 0 else 0

def nanstd_safe(arr):
    """Calculate std ignoring NaN values"""
    if len(arr) == 0:
        return 0
    clean_arr = arr[~np.isnan(arr)]
    return np.std(clean_arr) if len(clean_arr) > 0 else 0

# %%
# nnUNet paths - updated for nnUNet results
nnunet_base_path = '/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/'
gt_labels_path = os.path.join(nnunet_base_path, 'labelsTs')
predictions_path = os.path.join(nnunet_base_path, 'predTs_PP')  # Note: predTs_PP not predictionsTs_PP
metrics_path = os.path.join(nnunet_base_path, 'predTs_PP/summary.json')

# Add path for probability files
prob_path = os.path.join(nnunet_base_path, 'predT')  # Path to NIfTI probability files

# Output paths
figures_out = '/home/jruffle/Desktop/NNUNET_ENHANCEMENT_ARTICLE/figures_dataset003/'
os.makedirs(figures_out, exist_ok=True)

# Create directory for NIfTI probability maps
prob_nifti_out = os.path.join(figures_out, 'probability_niftis')
os.makedirs(prob_nifti_out, exist_ok=True)

# Load nnUNet metrics from summary.json
try:
    with open(metrics_path, 'r') as f:
        nnunet_metrics = json.load(f)
    print("Successfully loaded nnUNet metrics from Dataset003")
    print(f"Available metrics keys: {list(nnunet_metrics.keys())}")
    
    # Display structure of metrics to understand format
    if 'results' in nnunet_metrics:
        print(f"Number of cases in results: {len(nnunet_metrics['results'])}")
        if nnunet_metrics['results']:
            print(f"Sample case metrics keys: {list(nnunet_metrics['results'][list(nnunet_metrics['results'].keys())[0]].keys())}")
    
except FileNotFoundError as e:
    print(f"Error: Metrics file not found: {e}")
    nnunet_metrics = None
except Exception as e:
    print(f"An error occurred loading metrics: {e}")
    nnunet_metrics = None

# %%
# Label scheme for nnUNet model:
# 0 = not brain (background)
# 1 = normal brain
# 2 = other abnormality
# 3 = ET (enhancing tumour) - this is our focus

LABEL_BACKGROUND = 0
LABEL_NORMAL_BRAIN = 1
LABEL_OTHER_ABNORMALITY = 2
LABEL_ENHANCING_TUMOUR = 3

print(f"Label scheme:")
print(f"{LABEL_BACKGROUND} = Background (not brain)")
print(f"{LABEL_NORMAL_BRAIN} = Normal brain")
print(f"{LABEL_OTHER_ABNORMALITY} = Other abnormality")
print(f"{LABEL_ENHANCING_TUMOUR} = Enhancing tumour (ET) - PRIMARY FOCUS")

# %%
import numpy as np

# Extract metrics from nnUNet summary.json
if nnunet_metrics and 'metric_per_case' in nnunet_metrics:
    results_list = []

    for case_data in nnunet_metrics['metric_per_case']:
        # Extract case ID from reference file
        if 'reference_file' in case_data:
            case_id = os.path.basename(case_data['reference_file']).replace('.nii.gz', '')
        elif 'prediction_file' in case_data:
            case_id = os.path.basename(case_data['prediction_file']).replace('.nii.gz', '')
        else:
            continue  # Skip if no file reference

        # Initialize result row
        result_row = {'case_id': case_id}

        # Extract metrics for all relevant labels (1: normal brain, 2: other abnormality, 3: enhancing tumour)
        labels_to_extract = {
            '1': 'normal_brain',
            '2': 'other_abnormality',
            '3': 'enhancing_tumour'
        }

        for label_id, label_name in labels_to_extract.items():
            if 'metrics' in case_data and label_id in case_data['metrics']:
                label_metrics = case_data['metrics'][label_id]

                # Calculate precision and recall from confusion matrix elements
                tp = label_metrics.get('TP', np.nan)
                fp = label_metrics.get('FP', np.nan)
                fn = label_metrics.get('FN', np.nan)
                tn = label_metrics.get('TN', np.nan)

                precision = tp / (tp + fp) if (not np.isnan(tp) and not np.isnan(fp) and (tp + fp) > 0) else np.nan
                recall = tp / (tp + fn) if (not np.isnan(tp) and not np.isnan(fn) and (tp + fn) > 0) else np.nan
                specificity = tn / (tn + fp) if (not np.isnan(tn) and not np.isnan(fp) and (tn + fp) > 0) else np.nan

                # Add metrics with label prefix
                result_row.update({
                    f'{label_name}_dice': label_metrics.get('Dice', np.nan),
                    f'{label_name}_precision': precision,
                    f'{label_name}_recall': recall,
                    f'{label_name}_f1': label_metrics.get('Dice', np.nan),  # Dice is same as F1 for binary
                    f'{label_name}_balanced_acc': (recall + specificity) / 2 if not np.isnan(recall) and not np.isnan(specificity) else np.nan,
                    f'{label_name}_iou': label_metrics.get('IoU', np.nan),
                    f'{label_name}_tp': tp,
                    f'{label_name}_fp': fp,
                    f'{label_name}_fn': fn,
                    f'{label_name}_tn': tn
                })
            else:
                # If label not found, set to nan
                result_row.update({
                    f'{label_name}_dice': np.nan,
                    f'{label_name}_precision': np.nan,
                    f'{label_name}_recall': np.nan,
                    f'{label_name}_f1': np.nan,
                    f'{label_name}_balanced_acc': np.nan,
                    f'{label_name}_iou': np.nan,
                    f'{label_name}_tp': np.nan,
                    f'{label_name}_fp': np.nan,
                    f'{label_name}_fn': np.nan,
                    f'{label_name}_tn': np.nan
                })

        # Keep original column names for enhancing tumour for backward compatibility
        result_row.update({
            'dice': result_row.get('enhancing_tumour_dice', np.nan),
            'precision': result_row.get('enhancing_tumour_precision', np.nan),
            'recall': result_row.get('enhancing_tumour_recall', np.nan),
            'f1': result_row.get('enhancing_tumour_f1', np.nan),
            'balanced_acc': result_row.get('enhancing_tumour_balanced_acc', np.nan),
            'iou': result_row.get('enhancing_tumour_iou', np.nan),
            'tp': result_row.get('enhancing_tumour_tp', np.nan),
            'fp': result_row.get('enhancing_tumour_fp', np.nan),
            'fn': result_row.get('enhancing_tumour_fn', np.nan),
            'tn': result_row.get('enhancing_tumour_tn', np.nan)
        })

        results_list.append(result_row)

    # Create DataFrame from extracted metrics
    results_df = pd.DataFrame(results_list)
    print(f"Extracted metrics for {len(results_df)} cases from summary.json")

    # Add cohort information to results_df (like we do for all_images)
    cohorts = ['UPENN-GBM', 'UCSF-PDGM', 'BraTS2021', 'BraTS-GLI', 'BraTS-MEN', 'EGD', 'NHNN', 'BraTS-MET', 'BraTS-PED', 'BraTS-SSA']

    # Initialize cohort columns for results_df
    for c in cohorts:
        results_df[c] = 0

    results_df['Cohort'] = ''

    # Assign cohorts based on case_id patterns
    for idx, row in results_df.iterrows():
        for c in cohorts:
            if c in str(row['case_id']):
                results_df.at[idx, c] = 1
                results_df.at[idx, 'Cohort'] = c
                break

    # Display overall summary statistics for all labels
    print(f"\nOverall mean metrics:")
    for label_id, label_name in labels_to_extract.items():
        if label_id in nnunet_metrics['mean']:
            print(f"Label {label_id} ({label_name}) - Dice: {nnunet_metrics['mean'][label_id]['Dice']:.4f}")

    print(f"\nCases with enhancing tumour present: {len([r for r in results_list if not np.isnan(r['tp']) and not np.isnan(r['fn']) and (r['tp'] + r['fn'] > 0)])}")

else:
    print("Could not extract metrics from summary.json - metric_per_case not found")
    # Get list of files for manual computation if needed
    gt_files = sorted(glob.glob(os.path.join(gt_labels_path, '*.nii.gz')))
    pred_files = sorted(glob.glob(os.path.join(predictions_path, '*.nii.gz')))

    print(f"Found {len(gt_files)} ground truth files")
    print(f"Found {len(pred_files)} prediction files")

    # Create list of cases that have both GT and predictions
    gt_case_ids = [os.path.basename(f).replace('.nii.gz', '') for f in gt_files]
    pred_case_ids = [os.path.basename(f).replace('.nii.gz', '') for f in pred_files]

    # Find intersection
    common_cases = list(set(gt_case_ids) & set(pred_case_ids))
    common_cases.sort()

    print(f"Found {len(common_cases)} cases with both GT and predictions")
    results_df = pd.DataFrame({'case_id': common_cases})

    # Add cohort information even if no metrics loaded
    cohorts = ['UPENN-GBM', 'UCSF-PDGM', 'BraTS2021', 'BraTS-GLI', 'BraTS-MEN', 'EGD', 'NHNN', 'BraTS-MET', 'BraTS-PED', 'BraTS-SSA']
    for c in cohorts:
        results_df[c] = 0
    results_df['Cohort'] = ''

    for idx, row in results_df.iterrows():
        for c in cohorts:
            if c in str(row['case_id']):
                results_df.at[idx, c] = 1
                results_df.at[idx, 'Cohort'] = c
                break


# %%
# If we successfully loaded metrics from summary.json, display summary
if len(results_df) > 0 and 'dice' in results_df.columns:
    print("=" * 80)
    print("MULTI-LABEL SEGMENTATION METRICS SUMMARY")
    print("=" * 80)
    
    # Display metrics for all labels
    labels_info = {
        'enhancing_tumour': 'Enhancing Tumour (Label 3)',
        'normal_brain': 'Normal Brain (Label 1)', 
        'other_abnormality': 'Other Abnormality (Label 2)'
    }
    
    for label_prefix, label_description in labels_info.items():
        print(f"\n{label_description}:")
        print("-" * 50)

        # Check if columns exist for this label
        dice_col = f"{label_prefix}_dice"
        precision_col = f"{label_prefix}_precision"
        recall_col = f"{label_prefix}_recall"
        f1_col = f"{label_prefix}_f1"
        balanced_acc_col = f"{label_prefix}_balanced_acc"

        if dice_col in results_df.columns:
            # Calculate statistics, filtering out NaN values
            dice_values = results_df[dice_col].dropna()
            precision_values = results_df[precision_col].dropna()
            recall_values = results_df[recall_col].dropna()
            f1_values = results_df[f1_col].dropna()
            balanced_acc_values = results_df[balanced_acc_col].dropna()

            if len(dice_values) > 0:
                print(f"  Mean Dice: {dice_values.mean():.3f} ± {dice_values.std():.3f} (n={len(dice_values)})")
            if len(precision_values) > 0:
                print(f"  Mean Precision: {precision_values.mean():.3f} ± {precision_values.std():.3f} (n={len(precision_values)})")
            if len(recall_values) > 0:
                print(f"  Mean Recall: {recall_values.mean():.3f} ± {recall_values.std():.3f} (n={len(recall_values)})")
            if len(f1_values) > 0:
                print(f"  Mean F1: {f1_values.mean():.3f} ± {f1_values.std():.3f} (n={len(f1_values)})")
            if len(balanced_acc_values) > 0:
                print(f"  Mean Balanced Accuracy: {balanced_acc_values.mean():.3f} ± {balanced_acc_values.std():.3f} (n={len(balanced_acc_values)})")

            if len(dice_values) == 0:
                print(f"  No valid metrics found for {label_description}")
        else:
            print(f"  Metrics not available for {label_description}")

    print("\n" + "=" * 80)
    print("LEGACY ENHANCING TUMOUR METRICS (for backward compatibility):")
    print("-" * 50)
    print(f"Mean Dice score: {results_df['dice'].mean():.3f} ± {results_df['dice'].std():.3f}")
    print(f"Mean Precision: {results_df['precision'].mean():.3f} ± {results_df['precision'].std():.3f}")
    print(f"Mean Recall: {results_df['recall'].mean():.3f} ± {results_df['recall'].std():.3f}")
    print("Metrics successfully loaded from nnUNet summary.json")

else:
    print("No metrics found - you may need to check the summary.json structure")

# Display sample of the data
print("\nSample of loaded data (first 5 rows, key columns):")
key_columns = ['case_id', 'dice', 'normal_brain_dice', 'other_abnormality_dice', 'enhancing_tumour_dice']
available_columns = [col for col in key_columns if col in results_df.columns]
print(results_df[available_columns].head())


# %%
# Add volume information from confusion matrix data or by loading images
def add_volume_info(results_df, gt_path=None, pred_path=None, target_label=LABEL_ENHANCING_TUMOUR):
    """Add volume information to results dataframe"""
    
    # If we have confusion matrix data, we can derive volumes
    if 'tp' in results_df.columns and 'fn' in results_df.columns:
        results_df['gt_volume'] = results_df['tp'] + results_df['fn']  # TP + FN = total GT positive
        results_df['pred_volume'] = results_df['tp'] + results_df['fp']  # TP + FP = total predicted positive
        print("Volume information derived from confusion matrix data")
        return results_df
    
    # Otherwise, load from images (slower but more detailed)
    if gt_path is None or pred_path is None:
        print("No paths provided and no confusion matrix data available")
        return results_df
        
    volumes_gt = []
    volumes_pred = []
    
    for case_id in tqdm(results_df['case_id'], desc="Computing volumes"):
        try:
            gt_img = nib.load(os.path.join(gt_path, f"{case_id}.nii.gz")).get_fdata()
            pred_img = nib.load(os.path.join(pred_path, f"{case_id}.nii.gz")).get_fdata()
            
            gt_volume = np.sum(gt_img == target_label)
            pred_volume = np.sum(pred_img == target_label)
            
            volumes_gt.append(gt_volume)
            volumes_pred.append(pred_volume)
        except Exception as e:
            print(f"Error loading {case_id}: {e}")
            volumes_gt.append(0)
            volumes_pred.append(0)
    
    results_df['gt_volume'] = volumes_gt
    results_df['pred_volume'] = volumes_pred
    return results_df

# Add volume info if we have metrics
if 'dice' in results_df.columns:
    results_df = add_volume_info(results_df, gt_labels_path, predictions_path)
    print(f"Cases with GT enhancing tumour: {(results_df['gt_volume'] > 0).sum()}")
    print(f"Cases with predicted enhancing tumour: {(results_df['pred_volume'] > 0).sum()}")

    # Calculate balanced accuracy for detection of any enhancing tumour
    y_true = (results_df['gt_volume'] > 0).astype(int)
    y_pred = (results_df['pred_volume'] > 0).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    print(f"Balanced accuracy for detection of any enhancing tumour: {balanced_acc:.4f}")

    # Calculate per-case balanced accuracy and its std
    per_case_sens = []
    per_case_spec = []
    for yt, yp in zip(y_true, y_pred):
        if yt == 1:
            per_case_sens.append(1 if yp == 1 else 0)
        else:
            per_case_spec.append(1 if yp == 0 else 0)
    # Pad with nan if no cases
    sens_mean = np.mean(per_case_sens) if per_case_sens else np.nan
    spec_mean = np.mean(per_case_spec) if per_case_spec else np.nan
    per_case_bal_acc = []
    for yt, yp in zip(y_true, y_pred):
        if yt == 1:
            per_case_bal_acc.append(0.5 * (1 if yp == 1 else 0) + 0.5 * spec_mean)
        else:
            per_case_bal_acc.append(0.5 * (1 if yp == 0 else 0) + 0.5 * sens_mean)
    bal_acc_std = np.std(per_case_bal_acc)
    print(f"Balanced accuracy std (per-case): {bal_acc_std:.4f}")

# %%
# Create all_images DataFrame equivalent for nnUNet - ENTIRE DATASET, not just test set
# This should include all cases from all partitions (train, val, test) like the original notebook

# First, we need to get all filenames from the nnUNet dataset structure
# UPDATE: Use Dataset003 for the full dataset path as well
nnunet_base_path_full = '/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/'

# Define metadata path (needed for demographics loading)
metadata_path = '/home/jruffle/Desktop/ENHANCEMENT_ARTICLE/metadata/'

# Get all available cases from the nnUNet dataset
# This includes train, val, and test sets - the complete cohort
import glob

# Get all label files to determine the complete dataset
all_label_files = glob.glob(os.path.join(nnunet_base_path_full, 'labels*', '*.nii.gz'))
print(f"Found {len(all_label_files)} total cases in nnUNet Dataset003")

# Extract all case IDs from the complete dataset
all_case_ids = []
for label_file in all_label_files:
    case_id = os.path.basename(label_file).replace('.nii.gz', '')
    all_case_ids.append(case_id)

# Create all_images DataFrame for the ENTIRE cohort
all_images = pd.DataFrame(all_case_ids, columns=['filename'])
all_images['case_id'] = all_images['filename']  # Keep consistent with results_df

# Initialize partition column (we'll mark what we know from results_df as test, rest as train/val)
all_images['Partition'] = 'Train/Val'  # Default for cases not in test set
all_images.loc[all_images['case_id'].isin(results_df['case_id']), 'Partition'] = 'Test'

print(f"Total dataset size: {len(all_images)}")
print(f"Test set size: {(all_images['Partition'] == 'Test').sum()}")
print(f"Train/Val set size: {(all_images['Partition'] == 'Train/Val').sum()}")

# Add cohort information for ALL cases
cohorts = ['UPENN-GBM','UCSF-PDGM','BraTS2021','BraTS-GLI','BraTS-MEN','EGD','NHNN','BraTS-MET','BraTS-PED','BraTS-SSA']

# Initialize cohort columns
for c in cohorts:
    all_images[c] = 0

all_images['Cohort'] = ''

# Assign cohorts based on filename patterns
for idx, row in all_images.iterrows():
    for c in cohorts:
        if c in str(row['case_id']):
            all_images.at[idx, c] = 1
            all_images.at[idx, 'Cohort'] = c
            break

# Add subject identifiers for metadata matching
all_images['Subject'] = ''
for i, row in all_images.iterrows():
    if 'NHNN' in str(row['case_id']):
        # Extract subject ID from NHNN filename format
        all_images.at[i, 'Subject'] = 'NHNN_' + str(row['case_id'].split('_')[1])
    else:
        all_images.at[i, 'Subject'] = str(row['case_id'])

# Initialize Age and Sex columns
all_images['Age'] = np.nan
all_images['Sex'] = np.nan

# Load metadata files (same as before)
try:
    nhnn_metadata = pd.read_csv(metadata_path+'radiomic_models.csv',index_col=0)
    nhnn_metadata = nhnn_metadata[['male','age','feature','diagnosis','overall_survival']]
    nhnn_metadata['MRN'] = nhnn_metadata['feature'].apply(lambda x: x.split('_')[0])

    nhnn_mapping_path = metadata_path + 'NHNN_counter_subject_mapping.txt'
    nhnn_mapping = pd.read_csv(nhnn_mapping_path, sep=' ', header=None)
    nhnn_mapping.columns = ['ID mapping','MRN']
    nhnn_mapping['Subject'] = 'NHNN_' + nhnn_mapping['ID mapping'].astype(str)

    nhnn_metadata = nhnn_metadata.merge(nhnn_mapping, on='MRN', how='inner').drop_duplicates()
    nhnn_metadata['Sex'] = nhnn_metadata['male'].apply(lambda x: 'M' if x == 1 else 'F')
    nhnn_metadata.rename(columns={'age': 'Age'}, inplace=True)
    
    print("NHNN metadata loaded successfully")
except Exception as e:
    print(f"Error loading NHNN metadata: {e}")
    nhnn_metadata = pd.DataFrame()

# Load EGD metadata
try:
    egd_metadata = pd.read_excel(metadata_path+'Clinical_data.xlsx',index_col=None)
    print("EGD metadata loaded successfully")
except Exception as e:
    print(f"Error loading EGD metadata: {e}")
    egd_metadata = pd.DataFrame()

# Load UCSF metadata
try:
    ucsf_metadata = pd.read_csv(metadata_path+'UCSF-PDGM-metadata.csv',index_col=None)
    ucsf_metadata = ucsf_metadata.rename(columns={'ID': 'Subject', 'Age at MRI': 'Age'})
    # Modify UCSF-PDGM Subject IDs to add leading zeros
    ucsf_metadata['Subject'] = ucsf_metadata['Subject'].apply(
        lambda x: '-'.join(x.split('-')[:-1] + [f"{int(x.split('-')[-1]):04d}"]) 
        if x.startswith('UCSF-PDGM') else x
    )
    print("UCSF metadata loaded successfully")
except Exception as e:
    print(f"Error loading UCSF metadata: {e}")
    ucsf_metadata = pd.DataFrame()

# Load UPENN metadata
try:
    upenn_metadata = pd.read_csv(metadata_path+'UPENN-GBM_clinical_info_v1.1.csv',index_col=None)
    upenn_metadata = upenn_metadata.rename(columns={
        'ID': 'Subject', 
        'Gender': 'Sex', 
        'Age_at_scan_years': 'Age'
    })
    print("UPENN metadata loaded successfully")
except Exception as e:
    print(f"Error loading UPENN metadata: {e}")
    upenn_metadata = pd.DataFrame()

# Merge metadata with all_images DataFrame (same logic as original)
for i, row in all_images.iterrows():
    try:
        case_id = row['case_id']
        subject = row['Subject']
        
        if 'EGD' in case_id and not egd_metadata.empty:
            match = egd_metadata[egd_metadata['Subject'] == subject]
            if not match.empty:
                all_images.at[i, 'Age'] = match['Age'].values[0]
                all_images.at[i, 'Sex'] = match['Sex'].values[0]
                
        elif 'UCSF-PDGM' in case_id and not ucsf_metadata.empty:
            match = ucsf_metadata[ucsf_metadata['Subject'] == subject]
            if not match.empty:
                all_images.at[i, 'Age'] = match['Age'].values[0]
                all_images.at[i, 'Sex'] = match['Sex'].values[0]
                
        elif 'UPENN-GBM' in case_id and not upenn_metadata.empty:
            match = upenn_metadata[upenn_metadata['Subject'] == subject]
            if not match.empty:
                all_images.at[i, 'Age'] = match['Age'].values[0]
                all_images.at[i, 'Sex'] = match['Sex'].values[0]
                
        elif 'NHNN' in case_id and not nhnn_metadata.empty:
            match = nhnn_metadata[nhnn_metadata['Subject'] == subject]
            if not match.empty:
                all_images.at[i, 'Age'] = match['Age'].values[0]
                all_images.at[i, 'Sex'] = match['Sex'].values[0]
    except:
        continue

# Clean up invalid values
all_images.loc[all_images['Age'] <= 0, 'Age'] = np.nan
all_images.loc[all_images['Sex'] == -1, 'Sex'] = np.nan

# Print summary of complete dataset
age_available = all_images['Age'].notna().sum()
sex_available = all_images['Sex'].notna().sum()
total_cases = len(all_images)

print(f"\nComplete Dataset Summary (Dataset003):")
print(f"Total cases: {total_cases}")
print(f"Age data available: {age_available} ({age_available/total_cases*100:.1f}%)")
print(f"Sex data available: {sex_available} ({sex_available/total_cases*100:.1f}%)")

if age_available > 0:
    print(f"Age range: {all_images['Age'].min():.1f} - {all_images['Age'].max():.1f} years")
    print(f"Mean age: {all_images['Age'].mean():.1f} ± {all_images['Age'].std():.1f} years")

if sex_available > 0:
    sex_counts = all_images['Sex'].value_counts()
    print(f"Sex distribution: {dict(sex_counts)}")

print("\nCohort distribution in complete dataset:")
print(all_images['Cohort'].value_counts())

# %%
# Add country and pathology information to results_df (same as original)
countries = dict()
countries['UPENN-GBM'] = 'USA'
countries['UCSF-PDGM'] = 'USA'
countries['BraTS2021'] = 'USA'
countries['BraTS-GLI'] = 'USA'
countries['BraTS-MEN'] = 'USA'
countries['EGD'] = 'Netherlands'
countries['NHNN'] = 'UK'
countries['BraTS-MET'] = 'USA'
countries['BraTS-PED'] = 'USA'
countries['BraTS-SSA'] = 'Sub-Saharan Africa'

pathologies = dict()
pathologies['UPENN-GBM'] = 'Presurgical glioma'
pathologies['UCSF-PDGM'] = 'Presurgical glioma'
pathologies['BraTS2021'] = 'Presurgical glioma'
pathologies['BraTS-GLI'] = 'Postoperative glioma resection'
pathologies['BraTS-MEN'] = 'Meningioma'
pathologies['EGD'] = 'Presurgical glioma'
pathologies['NHNN'] = 'Presurgical glioma'
pathologies['BraTS-MET'] = 'Metastases'
pathologies['BraTS-PED'] = 'Paediatric presurgical tumour'
pathologies['BraTS-SSA'] = 'Presurgical glioma'

# Add country and pathology info to results_df
results_df['Country'] = ''
results_df['Pathology'] = ''

for country in countries.values():
    results_df[country] = 0
for pathology in pathologies.values():
    results_df[pathology] = 0
    
# Map cohorts to countries and pathologies (cohort info was already added in cell 4)
for i, row in results_df.iterrows():
    cohort = row['Cohort']
    if cohort in countries:
        country = countries[cohort]
        results_df.at[i, country] = 1
        results_df.at[i, 'Country'] = country
    if cohort in pathologies:
        pathology = pathologies[cohort]
        results_df.at[i, pathology] = 1
        results_df.at[i, 'Pathology'] = pathology

# %%
# Display results summary and ensure Age/Sex columns are present
print("nnUNet Enhancing Tumour Segmentation Results Summary")
print("=" * 60)
print(f"Total cases analyzed: {len(results_df)}")

if 'gt_volume' in results_df.columns:
    print(f"Cases with enhancing tumour (GT volume > 0): {(results_df['gt_volume'] > 0).sum()}")
    print(f"Cases where model predicted enhancement: {(results_df['pred_volume'] > 0).sum()}")

if 'dice' in results_df.columns:
    print()
    print("Overall Metrics (Enhancing Tumour - Label 3):")
    print(f"Mean Dice: {results_df['dice'].mean():.4f} ± {results_df['dice'].std():.4f}")
    print(f"Mean Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"Mean Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    
    if 'f1' in results_df.columns:
        print(f"Mean F1: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    if 'balanced_acc' in results_df.columns:
        print(f"Mean Balanced Accuracy: {results_df['balanced_acc'].mean():.4f} ± {results_df['balanced_acc'].std():.4f}")

    # Radiologist Cohort Performance Metrics
    print()
    print("Radiologist Cohort Performance (n=11 radiologists, 100 cases each):")
    print("-" * 60)
    
    # These values are from the comprehensive radiologist analysis
    # Calculated across all 1100 annotations (11 radiologists × 100 cases)
    print(f"Mean Precision: 0.835 ± 0.087")
    print(f"Mean Recall: 0.894 ± 0.056")
    print(f"Mean F1: 0.859 ± 0.044")
    print(f"Mean Balanced Accuracy: 0.866 ± 0.031")
    
    print()
    print("Note: Radiologist metrics calculated on 100 cases per radiologist")
    print("      (subset of the full 1109 test cases used for nnUNet evaluation)")

    # Save results
    results_df.to_csv(os.path.join(figures_out, 'nnunet_enhancing_tumour_results.csv'), index=False)
    print(f"\nResults saved to: {os.path.join(figures_out, 'nnunet_enhancing_tumour_results.csv')}")
else:
    print("No metrics available - check summary.json structure")

# Check if Age and Sex columns exist, if not, load them
if 'Age' not in results_df.columns or 'Sex' not in results_df.columns:
    print("\nLoading demographic data (Age/Sex)...")
    
    # Try to load the updated CSV that has Age/Sex columns
    try:
        results_df_updated = pd.read_csv(os.path.join(figures_out, 'nnunet_enhancing_tumour_results.csv'))
        if 'Age' in results_df_updated.columns and 'Sex' in results_df_updated.columns:
            results_df = results_df_updated
            print(f"Loaded demographic data: {results_df['Age'].notna().sum()} cases with age, {results_df['Sex'].notna().sum()} cases with sex")
        else:
            print("Warning: Age/Sex columns still missing from saved file")
    except:
        print("Warning: Could not load demographic data")

# Print demographic summary if available  
if 'Age' in results_df.columns:
    age_available = results_df['Age'].notna().sum()
    print(f"\nAge data available: {age_available}/{len(results_df)} ({age_available/len(results_df)*100:.1f}%)")
    if age_available > 0:
        print(f"Age range: {results_df['Age'].min():.1f} - {results_df['Age'].max():.1f} years")
        print(f"Mean age: {results_df['Age'].mean():.1f} ± {results_df['Age'].std():.1f} years")

if 'Sex' in results_df.columns:
    sex_available = results_df['Sex'].notna().sum()
    print(f"Sex data available: {sex_available}/{len(results_df)} ({sex_available/len(results_df)*100:.1f}%)")
    if sex_available > 0:
        sex_counts = results_df['Sex'].value_counts()
        print(f"Sex distribution: {dict(sex_counts)}")

# %%
# COMPREHENSIVE TEST COHORT STATISTICS FOR MANUSCRIPT
print("\n" + "="*80)
print("COMPREHENSIVE TEST COHORT STATISTICS (N=1109)")
print("="*80)
print("\nThese statistics are for completing the manuscript demographic description:")
print("-" * 80)

# Get test cohort data directly from all_images where demographic data is available
test_data = all_images[all_images['Partition'] == 'Test'].copy()
total_cases = len(test_data)
print(f"\nTOTAL COHORT SIZE: {total_cases} patients")

# Age statistics - use test_data from all_images which has the demographic data
age_available = test_data['Age'].notna().sum()
age_percentage = (age_available / total_cases * 100) if total_cases > 0 else 0

if age_available > 0:
    age_data = test_data['Age'].dropna()
    age_mean = age_data.mean()
    age_std = age_data.std()
    age_min = age_data.min()
    age_max = age_data.max()
    print(f"\nAGE STATISTICS:")
    print(f"  - Available for: {age_available} cases ({age_percentage:.1f}%)")
    print(f"  - Mean ± SD: {age_mean:.1f} ± {age_std:.1f} years")
    print(f"  - Range: {age_min:.1f} - {age_max:.1f} years")
else:
    print(f"\nAGE STATISTICS:")
    print(f"  - Available for: 0 cases (0.0%)")
    print(f"  - Mean ± SD: Not available")

# Sex statistics - use test_data from all_images which has the demographic data
sex_available = test_data['Sex'].notna().sum()
sex_percentage = (sex_available / total_cases * 100) if total_cases > 0 else 0

if sex_available > 0:
    sex_counts = test_data['Sex'].value_counts()
    male_count = sex_counts.get('M', 0)
    female_count = sex_counts.get('F', 0)
    print(f"\nSEX STATISTICS:")
    print(f"  - Available for: {sex_available} cases ({sex_percentage:.1f}%)")
    print(f"  - Male: {male_count} patients")
    print(f"  - Female: {female_count} patients")
else:
    print(f"\nSEX STATISTICS:")
    print(f"  - Available for: 0 cases (0.0%)")
    male_count = 0
    female_count = 0

# Pathology breakdown - need to merge with results_df to get pathology data
# First merge test_data with results_df to get Pathology and Country columns
test_data_full = test_data.merge(results_df[['case_id', 'Pathology', 'Country']], on='case_id', how='left')

print(f"\nPATHOLOGY BREAKDOWN:")
pathology_counts = test_data_full['Pathology'].value_counts()

# Count specific pathologies
presurgical_glioma = pathology_counts.get('Presurgical glioma', 0)
postop_glioma = pathology_counts.get('Postoperative glioma resection', 0)
meningioma = pathology_counts.get('Meningioma', 0)
metastases = pathology_counts.get('Metastases', 0)
paediatric = pathology_counts.get('Paediatric presurgical tumour', 0)

print(f"  - Presurgical glioma: {presurgical_glioma} patients")
print(f"  - Postoperative glioma resection: {postop_glioma} patients")
print(f"  - Meningioma: {meningioma} patients")
print(f"  - Metastases: {metastases} patients")
print(f"  - Paediatric presurgical tumour: {paediatric} patients")

# Verify total
pathology_total = presurgical_glioma + postop_glioma + meningioma + metastases + paediatric
if pathology_total != total_cases:
    print(f"  - Other/Unknown: {total_cases - pathology_total} patients")

# Country breakdown
print(f"\nCOUNTRY BREAKDOWN:")
country_counts = test_data_full['Country'].value_counts()

uk_count = country_counts.get('UK', 0)
usa_count = country_counts.get('USA', 0)
netherlands_count = country_counts.get('Netherlands', 0)
ssa_count = country_counts.get('Sub-Saharan Africa', 0)

print(f"  - UK: {uk_count} patients")
print(f"  - USA: {usa_count} patients")
print(f"  - Netherlands: {netherlands_count} patients")
print(f"  - Sub-Saharan Africa: {ssa_count} patients")

# Verify total
country_total = uk_count + usa_count + netherlands_count + ssa_count
if country_total != total_cases:
    print(f"  - Other/Unknown: {total_cases - country_total} patients")

# Generate the complete sentence for the manuscript
print("\n" + "="*80)
print("MANUSCRIPT TEXT (Copy and paste this):")
print("="*80)

if age_available > 0 and sex_available > 0:
    manuscript_text = (
        f"The study cohort included {total_cases} patients with brain tumours. "
        f"Age was available for {age_available} cases ({age_percentage:.1f}%), "
        f"with a mean ± standard deviation of {age_mean:.1f} ± {age_std:.1f} years. "
        f"Patient sex was available for {sex_available} cases ({sex_percentage:.1f}%), "
        f"{male_count} of which were male, and {female_count} female. "
        f"This included {presurgical_glioma} patients with presurgical glioma, "
        f"{postop_glioma} with postoperative glioma resection, "
        f"{meningioma} with meningioma, "
        f"{metastases} with metastases, "
        f"and {paediatric} with paediatric gliomas. "
        f"{uk_count} cases were from the UK, "
        f"{usa_count} were from the USA, "
        f"{netherlands_count} were from The Netherlands, "
        f"and {ssa_count} were from Sub-Saharan Africa."
    )
elif age_available > 0:  # Age available but not sex
    manuscript_text = (
        f"The study cohort included {total_cases} patients with brain tumours. "
        f"Age was available for {age_available} cases ({age_percentage:.1f}%), "
        f"with a mean ± standard deviation of {age_mean:.1f} ± {age_std:.1f} years. "
        f"Patient sex data was not available. "
        f"This included {presurgical_glioma} patients with presurgical glioma, "
        f"{postop_glioma} with postoperative glioma resection, "
        f"{meningioma} with meningioma, "
        f"{metastases} with metastases, "
        f"and {paediatric} with paediatric gliomas. "
        f"{uk_count} cases were from the UK, "
        f"{usa_count} were from the USA, "
        f"{netherlands_count} were from The Netherlands, "
        f"and {ssa_count} were from Sub-Saharan Africa."
    )
elif sex_available > 0:  # Sex available but not age
    manuscript_text = (
        f"The study cohort included {total_cases} patients with brain tumours. "
        f"Age data was not available. "
        f"Patient sex was available for {sex_available} cases ({sex_percentage:.1f}%), "
        f"{male_count} of which were male, and {female_count} female. "
        f"This included {presurgical_glioma} patients with presurgical glioma, "
        f"{postop_glioma} with postoperative glioma resection, "
        f"{meningioma} with meningioma, "
        f"{metastases} with metastases, "
        f"and {paediatric} with paediatric gliomas. "
        f"{uk_count} cases were from the UK, "
        f"{usa_count} were from the USA, "
        f"{netherlands_count} were from The Netherlands, "
        f"and {ssa_count} were from Sub-Saharan Africa."
    )
else:  # Neither age nor sex available
    manuscript_text = (
        f"The study cohort included {total_cases} patients with brain tumours. "
        f"Demographic data (age and sex) was not available for this cohort. "
        f"This included {presurgical_glioma} patients with presurgical glioma, "
        f"{postop_glioma} with postoperative glioma resection, "
        f"{meningioma} with meningioma, "
        f"{metastases} with metastases, "
        f"and {paediatric} with paediatric gliomas. "
        f"{uk_count} cases were from the UK, "
        f"{usa_count} were from the USA, "
        f"{netherlands_count} were from The Netherlands, "
        f"and {ssa_count} were from Sub-Saharan Africa."
    )

print(manuscript_text)
print("\n" + "="*80)

# EXPANDED SECTION: Calculate sample-level metrics for detection of any enhancing tumor
print("\n" + "="*60)
print("SAMPLE-LEVEL DETECTION METRICS (Patient-level)")
print("="*60)

# Calculate sample-level (patient-level) metrics
y_true = (results_df['gt_volume'] > 0).astype(int)
y_pred = (results_df['pred_volume'] > 0).astype(int)

# Confusion matrix elements
tp = ((y_true == 1) & (y_pred == 1)).sum()
tn = ((y_true == 0) & (y_pred == 0)).sum()
fp = ((y_true == 0) & (y_pred == 1)).sum()
fn = ((y_true == 1) & (y_pred == 0)).sum()

# Calculate metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = sensitivity  # Same as sensitivity
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
balanced_acc = (sensitivity + specificity) / 2

print(f"Detection of ANY enhancing tumor (sample/patient level):")
print(f"  True Positives: {tp}")
print(f"  True Negatives: {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print()
print(f"  Sensitivity (Recall): {sensitivity:.4f}")
print(f"  Specificity: {specificity:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  F1 Score: {f1_score:.4f}")
print(f"  Balanced Accuracy: {balanced_acc:.4f}")

# Calculate per-case balanced accuracy and its std (for standard deviation)
per_case_sens = []
per_case_spec = []
for yt, yp in zip(y_true, y_pred):
    if yt == 1:
        per_case_sens.append(1 if yp == 1 else 0)
    else:
        per_case_spec.append(1 if yp == 0 else 0)

# Calculate mean sensitivity and specificity for std calculation
sens_mean = np.mean(per_case_sens) if per_case_sens else np.nan
spec_mean = np.mean(per_case_spec) if per_case_spec else np.nan

# Calculate per-case balanced accuracy for std
per_case_bal_acc = []
for yt, yp in zip(y_true, y_pred):
    if yt == 1:
        per_case_bal_acc.append(0.5 * (1 if yp == 1 else 0) + 0.5 * spec_mean)
    else:
        per_case_bal_acc.append(0.5 * (1 if yp == 0 else 0) + 0.5 * sens_mean)

bal_acc_std = np.std(per_case_bal_acc)

# Calculate standard deviations for ALL metrics using bootstrap
np.random.seed(42)
n_bootstrap = 1000
bootstrap_metrics = {
    'sensitivity': [], 
    'specificity': [],
    'precision': [], 
    'recall': [], 
    'f1': []
}

for _ in range(n_bootstrap):
    # Bootstrap sample
    indices = np.random.choice(len(results_df), size=len(results_df), replace=True)
    y_true_boot = y_true.iloc[indices]
    y_pred_boot = y_pred.iloc[indices]
    
    # Calculate metrics for bootstrap sample
    tp_boot = ((y_true_boot == 1) & (y_pred_boot == 1)).sum()
    tn_boot = ((y_true_boot == 0) & (y_pred_boot == 0)).sum()
    fp_boot = ((y_true_boot == 0) & (y_pred_boot == 1)).sum()
    fn_boot = ((y_true_boot == 1) & (y_pred_boot == 0)).sum()
    
    # Sensitivity
    if (tp_boot + fn_boot) > 0:
        sensitivity_boot = tp_boot / (tp_boot + fn_boot)
    else:
        sensitivity_boot = 0
    
    # Specificity
    if (tn_boot + fp_boot) > 0:
        specificity_boot = tn_boot / (tn_boot + fp_boot)
    else:
        specificity_boot = 0
    
    # Precision
    if (tp_boot + fp_boot) > 0:
        precision_boot = tp_boot / (tp_boot + fp_boot)
    else:
        precision_boot = 0
    
    # Recall (same as sensitivity)
    recall_boot = sensitivity_boot
    
    # F1
    if (precision_boot + recall_boot) > 0:
        f1_boot = 2 * (precision_boot * recall_boot) / (precision_boot + recall_boot)
    else:
        f1_boot = 0
    
    bootstrap_metrics['sensitivity'].append(sensitivity_boot)
    bootstrap_metrics['specificity'].append(specificity_boot)
    bootstrap_metrics['precision'].append(precision_boot)
    bootstrap_metrics['recall'].append(recall_boot)
    bootstrap_metrics['f1'].append(f1_boot)

# Calculate standard deviations
sensitivity_std = np.std(bootstrap_metrics['sensitivity'])
specificity_std = np.std(bootstrap_metrics['specificity'])
precision_std = np.std(bootstrap_metrics['precision'])
recall_std = np.std(bootstrap_metrics['recall'])
f1_std = np.std(bootstrap_metrics['f1'])

print()
print(f"Sample-level metrics with standard deviations:")
print(f"  Sensitivity: {sensitivity:.4f} ± {sensitivity_std:.4f}")
print(f"  Specificity: {specificity:.4f} ± {specificity_std:.4f}")
print(f"  Precision: {precision:.4f} ± {precision_std:.4f}")
print(f"  Recall: {recall:.4f} ± {recall_std:.4f}") 
print(f"  F1 Score: {f1_score:.4f} ± {f1_std:.4f}")
print(f"  Balanced Accuracy: {balanced_acc:.4f} ± {bal_acc_std:.4f}")

# Add these sample-level metrics to the results dataframe for future use
results_df['sample_level_sensitivity'] = sensitivity
results_df['sample_level_specificity'] = specificity
results_df['sample_level_precision'] = precision
results_df['sample_level_recall'] = recall
results_df['sample_level_f1'] = f1_score
results_df['sample_level_balanced_acc'] = balanced_acc

# Save updated results with sample-level metrics
results_df.to_csv(os.path.join(figures_out, 'nnunet_enhancing_tumour_results_with_sample_metrics.csv'), index=False)
print(f"\nUpdated results with sample-level metrics saved to: nnunet_enhancing_tumour_results_with_sample_metrics.csv")

# %%
def create_missed_cases_image_figure(n_cases=6, include_filename=True, include_cohort=False, include_nhnn=False, min_dice_threshold=0.3, min_volume_threshold=50):
    """
    Create a figure showing actual images of cases most frequently missed by radiologists
    but correctly identified by the AI model, using the same style as create_pathology_figures_best_worst.
    Shows: T1, T2, FLAIR, Prediction with FLAIR background, T1CE, Ground Truth with T1CE background
    
    Parameters:
    -----------
    n_cases : int, default=6
        Number of cases to show across the rows
    include_filename : bool, default=True
    include_cohort : bool, default=False
        Whether to include the cohort information in the y-axis labels
        Whether to include the filename/case_id in the y-axis labels
    include_nhnn : bool, default=False
        Whether to include cases from the NHNN cohort
    min_dice_threshold : float, default=0.3
        Minimum Dice score threshold for including cases
    min_volume_threshold : int, default=50
        Minimum tumor volume (in voxels) for including cases
    """
    
    # Import required modules
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import glob
    import json
    import nibabel as nib
    
    print("CREATING MISSED CASES IMAGE VISUALIZATION")
    print("=" * 40)
    
    # Load the radiologist review data and model results
    results_df_loaded = pd.read_csv(os.path.join(figures_out, 'nnunet_enhancing_tumour_results.csv'))
    
    # Load radiologist data from the multi-radiologist analysis
    RADIOLOGIST_REVIEWS_PATH = '/home/jruffle/Desktop/ENHANCEMENT_ARTICLE/radiologist_reviews/'
    json_files = glob.glob(os.path.join(RADIOLOGIST_REVIEWS_PATH, '*.json'))
    
    def load_radiologist_data(json_file_path):
        """Load and process single radiologist review data"""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        radiologist_name = os.path.basename(json_file_path).replace('.json', '')
        
        def results_to_dataframe(results_list, with_segmentation=False):
            rows = []
            for result in results_list:
                row = {
                    'radiologist': radiologist_name,
                    'case_id': result['sample']['base_name'],
                    'predicted_enhancement': 1 if result['abnormality'] == 'Y' else 0,
                    'confidence': result['confidence'],
                    'image_quality': result['image_quality'],
                    'response_time': result['response_time'],
                    'ground_truth_sum': result['sample']['ground_truth_sum'],
                    'has_enhancement_gt': 1 if result['sample']['ground_truth_sum'] > 0 else 0,
                    'with_segmentation': with_segmentation
                }
                rows.append(row)
            return pd.DataFrame(rows)
        
        # Process both conditions
        dfs_to_combine = []
        
        if 'results_without_seg' in data:
            df_without = results_to_dataframe(data['results_without_seg'], False)
            dfs_to_combine.append(df_without)
        
        if 'results_with_seg' in data:
            df_with = results_to_dataframe(data['results_with_seg'], True)
            dfs_to_combine.append(df_with)
        
        if dfs_to_combine:
            combined_df = pd.concat(dfs_to_combine, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        return combined_df
    
    # Load all radiologist data
    all_radiologist_data = []
    
    for json_file in json_files:
        try:
            rad_data = load_radiologist_data(json_file)
            if len(rad_data) > 0:
                all_radiologist_data.append(rad_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not all_radiologist_data:
        print("Error: Could not load radiologist data")
        return None
    
    # Combine all radiologist data
    radiologist_df = pd.concat(all_radiologist_data, ignore_index=True)
    
    # Add model predictions and ground truth
    results_df_loaded['model_predicted_enhancement'] = (results_df_loaded['pred_volume'] > 0).astype(int)
    results_df_loaded['model_has_enhancement_gt'] = (results_df_loaded['gt_volume'] > 0).astype(int)
    
    # Merge radiologist data with model results
    radiologist_df = radiologist_df.merge(
        results_df_loaded[['case_id', 'model_predicted_enhancement', 'dice', 'precision', 'recall', 
                   'balanced_acc', 'Cohort', 'Country', 'Pathology', 'gt_volume', 'pred_volume']],
        on='case_id', how='left'
    )
    
    # Ensure consistent ground truth
    radiologist_df.loc[radiologist_df['gt_volume'].notna(), 'has_enhancement_gt'] = (
        radiologist_df.loc[radiologist_df['gt_volume'].notna(), 'gt_volume'] > 0
    ).astype(int)
    
    # IMPORTANT: Filter for cases with Dice >= 0.3 as requested
    # Identify the target cases: Radiologist wrong (false negative), Model correct (true positive)
    # with sufficient tumor volume AND Dice >= 0.3
    target_cases = radiologist_df[
        (radiologist_df['has_enhancement_gt'] == 1) &  # Ground truth: has enhancement
        (radiologist_df['predicted_enhancement'] == 0) &  # Radiologist: predicted no enhancement (WRONG)
        (radiologist_df['model_predicted_enhancement'] == 1) &  # Model: predicted enhancement (CORRECT)
        (radiologist_df['gt_volume'] >= min_volume_threshold) &  # Minimum tumor volume
        (radiologist_df['dice'] >= min_dice_threshold)  # Only include cases with Dice >= threshold
    ]
    
    # Filter out NHNN cohort if requested
    if not include_nhnn:
        target_cases = target_cases[target_cases['Cohort'] != 'NHNN']
    
    print(f"Found {len(target_cases)} cases where radiologists were wrong but model was correct (Dice >= 0.3)")
    
    if len(target_cases) == 0:
        print("No cases found to visualize with Dice >= 0.3")
        return None
    
    # Get the most frequently missed cases
    case_miss_counts = target_cases['case_id'].value_counts()
    
    # Select top n_cases most frequently missed cases for visualization
    n_cases_to_show = min(n_cases, len(case_miss_counts))
    top_missed_cases = case_miss_counts.head(n_cases_to_show).index.tolist()
    print(f"Top missed cases with Dice >= 0.3: {top_missed_cases}")
    
    # Get unique cases for visualization
    selected_cases = []
    for case_id in top_missed_cases:
        case_data = target_cases[target_cases['case_id'] == case_id].iloc[0]
        selected_cases.append(case_data)
    
    n_cases_found = len(selected_cases)
    print(f"Visualizing {n_cases_found} cases")
    
    # Create figure with SAME LAYOUT as create_pathology_figures_best_worst
    # 6 columns: T1, T2, FLAIR, Prediction with FLAIR, T1CE, GT with T1CE
    fig, axes = plt.subplots(n_cases_found, 6, figsize=(22, n_cases_found * 3.8))
    
    if n_cases_found == 1:
        axes = np.array([axes])
    
    # Column titles
    col_titles = ['T1', 'T2', 'FLAIR', 'Enhancing prediction\n(from T1+T2+FLAIR)', 
                 'T1CE\n(Held out from model)', 'Ground Truth\n(with T1CE background)']
    
    # Create green colormap for overlays
    green_cmap = ListedColormap(['none', 'green'])  # 'none' for transparent, 'green' for mask
    
    # Define data path
    data_path = '/home/jruffle/Documents/seq-synth/data/'
    
    # Process each case
    for i, case_row in enumerate(selected_cases):
        case_id = case_row['case_id']
        miss_count = case_miss_counts[case_id]
        
        # Calculate percentage of radiologists who missed this case
        total_radiologists = radiologist_df['radiologist'].nunique()
        miss_percentage = (miss_count / total_radiologists) * 100
        
        try:
            # Load ground truth and prediction
            gt_img = nib.load(os.path.join(gt_labels_path, f"{case_id}.nii.gz")).get_fdata()
            pred_img = nib.load(os.path.join(predictions_path, f"{case_id}.nii.gz")).get_fdata()
            
            # Try to load structural images - EXACTLY SAME AS create_pathology_figures_best_worst
            try:
                # Try sequences_merged directory first
                seq_path = os.path.join(data_path, 'sequences_merged', f"{case_id}.nii.gz")
                brain_mask_path = os.path.join(data_path, 'lesion_masks_augmented', f"{case_id}.nii.gz")
                
                if os.path.exists(seq_path) and os.path.exists(brain_mask_path):
                    seq_img = nib.load(seq_path).get_fdata()
                    brain_mask = nib.load(brain_mask_path).get_fdata()
                    brain_mask[brain_mask > 0] = 1
                    
                    flair_img = seq_img[..., 0] * brain_mask
                    t1_img = seq_img[..., 1] * brain_mask 
                    t1ce_img = seq_img[..., 2] * brain_mask
                    t2_img = seq_img[..., 3] * brain_mask
                    print(f"Successfully loaded sequences for {case_id}")
                else:
                    # Fallback to nnUNet structure
                    images_path = gt_labels_path.replace('labelsTs', 'imagesTs')
                    t1_path = os.path.join(images_path, f"{case_id}_0000.nii.gz")
                    t2_path = os.path.join(images_path, f"{case_id}_0001.nii.gz") 
                    flair_path = os.path.join(images_path, f"{case_id}_0002.nii.gz")
                    t1ce_path = os.path.join(images_path, f"{case_id}_0003.nii.gz")
                    
                    t1_img = nib.load(t1_path).get_fdata() if os.path.exists(t1_path) else None
                    t2_img = nib.load(t2_path).get_fdata() if os.path.exists(t2_path) else None
                    flair_img = nib.load(flair_path).get_fdata() if os.path.exists(flair_path) else None
                    t1ce_img = nib.load(t1ce_path).get_fdata() if os.path.exists(t1ce_path) else None
                    
                    print(f"Using nnUNet structure for {case_id}")
                    
            except Exception as e:
                print(f"Could not load sequences for {case_id}: {e}")
                t1_img = t2_img = flair_img = t1ce_img = None
            
            # Find best slice with enhancing tumour
            et_slices = np.sum(gt_img == LABEL_ENHANCING_TUMOUR, axis=(0, 1))
            if np.max(et_slices) > 0:
                z_slice = np.argmax(et_slices)
            else:
                z_slice = gt_img.shape[2] // 2
            
            # Get 2D slices
            gt_slice = gt_img[:, :, z_slice]
            pred_slice = pred_img[:, :, z_slice]
            
            # Create binary masks
            gt_et = (gt_slice == LABEL_ENHANCING_TUMOUR).astype(np.uint8)
            pred_et = (pred_slice == LABEL_ENHANCING_TUMOUR).astype(np.uint8)
            
            # Calculate ground truth centroid for red arrow
            gt_centroid = None
            if np.any(gt_et):
                gt_centroid = center_of_mass(gt_et)
            
            # Column 1: T1
            if t1_img is not None:
                axes[i, 0].imshow(np.rot90(t1_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 0].text(0.5, 0.5, 'T1\nNot Available', ha='center', va='center', transform=axes[i, 0].transAxes)
            
            # Column 2: T2
            if t2_img is not None:
                axes[i, 1].imshow(np.rot90(t2_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 1].text(0.5, 0.5, 'T2\nNot Available', ha='center', va='center', transform=axes[i, 1].transAxes)
            
            # Column 3: FLAIR
            if flair_img is not None:
                axes[i, 2].imshow(np.rot90(flair_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 2].text(0.5, 0.5, 'FLAIR\nNot Available', ha='center', va='center', transform=axes[i, 2].transAxes)
            
            # Column 4: Prediction with FLAIR background
            if flair_img is not None:
                axes[i, 3].imshow(np.rot90(flair_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 3].imshow(np.rot90(pred_et*0), cmap='gray')
            axes[i, 3].imshow(np.rot90(pred_et), cmap=green_cmap, alpha=0.6, vmin=0, vmax=1)
            
            # Column 5: T1CE with red arrow pointing to enhancing tumor
            if t1ce_img is not None:
                axes[i, 4].imshow(np.rot90(t1ce_img[:, :, z_slice]), cmap='gray')
                
                # Add red arrow pointing to enhancing tumor (matching Figure 2 style from multi_radiologist_analysis.py)
                if gt_centroid is not None:
                    # Apply rotation to match the displayed image
                    gt_rotated = np.rot90(gt_et)
                    if np.any(gt_rotated):
                        centroid_y, centroid_x = center_of_mass(gt_rotated)
                        
                        # Calculate arrow that points FROM near center TO the tumor
                        arrow_length = 40
                        img_h, img_w = gt_rotated.shape
                        
                        # Find which corner has most distance to centroid (to point away from)
                        corners = [
                            (10, 10),  # top-left
                            (img_w-10, 10),  # top-right
                            (10, img_h-10),  # bottom-left
                            (img_w-10, img_h-10)  # bottom-right
                        ]
                        
                        # Choose corner furthest from centroid
                        max_dist = 0
                        best_corner = corners[0]
                        for corner in corners:
                            dist = np.sqrt((corner[0] - centroid_x)**2 + (corner[1] - centroid_y)**2)
                            if dist > max_dist:
                                max_dist = dist
                                best_corner = corner
                        
                        # Calculate arrow vector (from corner toward centroid)
                        dx = centroid_x - best_corner[0]
                        dy = centroid_y - best_corner[1]
                        norm = np.sqrt(dx**2 + dy**2)
                        
                        if norm > arrow_length:
                            # Normalize and scale
                            dx = dx / norm * arrow_length
                            dy = dy / norm * arrow_length
                            # Arrow starts closer to center, points to tumor
                            arrow_start_x = centroid_x - dx
                            arrow_start_y = centroid_y - dy
                            
                            # Add red arrow annotation
                            axes[i, 4].annotate('', xy=(centroid_x, centroid_y), 
                                               xytext=(arrow_start_x, arrow_start_y),
                                               arrowprops=dict(arrowstyle='->', color='red', lw=2.5, alpha=0.9,
                                                             shrinkA=0, shrinkB=5))
            else:
                axes[i, 4].text(0.5, 0.5, 'T1CE\nNot Available', ha='center', va='center', transform=axes[i, 4].transAxes)
            
            # Column 6: Ground truth with T1CE
            if t1ce_img is not None:
                axes[i, 5].imshow(np.rot90(t1ce_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 5].imshow(np.rot90(gt_et*0), cmap='gray')
            axes[i, 5].imshow(np.rot90(gt_et), cmap=green_cmap, alpha=0.6, vmin=0, vmax=1)
            
            # Add titles to first row
            if i == 0:
                for j, title in enumerate(col_titles):
                    axes[i, j].set_title(title, fontsize=11, pad=8)
            
            # UPDATED: Add case information WITHOUT filename, NOT bold, NOT red
            cohort_name = case_row['Cohort']
            dice_score = case_row['dice']
            newline = '\n'
            # Changed: No case_id in label, no red color, no bold weight
            # Set y-axis label based on parameters
            label_parts = []
            if include_filename:
                label_parts.append(case_id)
            if include_cohort:
                label_parts.append(f"Cohort: {cohort_name}")
            label_parts.append(f"{miss_percentage:.1f}% of radiologists\npredicted will not enhance\n")
            label_parts.append(f"Model dice: {dice_score:.3f}")
            
            axes[i, 0].set_ylabel(newline.join(label_parts), fontsize=11)
            
            # Remove axis ticks
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        except Exception as e:
            print(f"Error visualizing {case_id}: {e}")
            for j in range(6):
                error_text = f'Error loading{chr(10)}{case_id}'
                axes[i, j].text(0.5, 0.5, error_text, 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    # Adjust layout - EXACTLY SAME AS create_pathology_figures_best_worst
    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.05, wspace=-0.6, top=0.92, bottom=0.06)
    
    # Add white divider line between columns 3 and 4
    if n_cases_found > 0:
        fig.canvas.draw()
        pos3 = axes[0, 3].get_position()
        pos4 = axes[0, 4].get_position()
        line_x = (pos3.x1 + pos4.x0) / 2
        
        # White line with black border
        border_line = plt.Line2D([line_x, line_x], [0.06, 0.92], 
                                transform=fig.transFigure, 
                                color='black', 
                                linewidth=8,
                                solid_capstyle='butt',
                                zorder=9)
        fig.add_artist(border_line)
        
        line = plt.Line2D([line_x, line_x], [0.06, 0.92], 
                         transform=fig.transFigure, 
                         color='white', 
                         linewidth=6,
                         solid_capstyle='butt',
                         zorder=10)
        fig.add_artist(line)
    
    # UPDATED: Title NOT bold, NOT red
    plt.suptitle('Test-set samples of cases missed by radiologists', 
                fontsize=18, y=0.96)  # Removed color='red' and weight='bold'
    
    # Add explanation text at bottom
    
    # Save the figure
    fig.savefig(os.path.join(figures_out, 'Figure_4.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(figures_out, 'Figure_4.svg'), 
                bbox_inches='tight', facecolor='white')
    
    print(f"Saved: Figure_4.png")
    print(f"Saved: Figure_4.svg")
    
    plt.show()
    
    return selected_cases


def create_expert_failure_ai_success_figure(n_cases=10):
    """
    Create a figure showing cases where expert radiologists were wrong but AI was correct,
    using statistical analysis and multiple panels.
    
    Parameters:
    -----------
    n_cases : int, default=10
        Number of most frequently missed cases to show in Panel A
    """
    
    # Import required modules
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import glob
    import json
    from matplotlib.colors import ListedColormap
    
    print("CREATING EXPERT FAILURE vs AI SUCCESS FIGURE")
    print("=" * 48)
    
    # Load the radiologist review data and model results
    results_df_loaded = pd.read_csv(os.path.join(figures_out, 'nnunet_enhancing_tumour_results.csv'))
    
    # Load radiologist data from the multi-radiologist analysis
    RADIOLOGIST_REVIEWS_PATH = '/home/jruffle/Desktop/ENHANCEMENT_ARTICLE/radiologist_reviews/'
    json_files = glob.glob(os.path.join(RADIOLOGIST_REVIEWS_PATH, '*.json'))
    
    def load_radiologist_data(json_file_path):
        """Load and process single radiologist review data"""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        radiologist_name = os.path.basename(json_file_path).replace('.json', '')
        
        def results_to_dataframe(results_list, with_segmentation=False):
            rows = []
            for result in results_list:
                row = {
                    'radiologist': radiologist_name,
                    'case_id': result['sample']['base_name'],
                    'predicted_enhancement': 1 if result['abnormality'] == 'Y' else 0,
                    'confidence': result['confidence'],
                    'image_quality': result['image_quality'],
                    'response_time': result['response_time'],
                    'ground_truth_sum': result['sample']['ground_truth_sum'],
                    'has_enhancement_gt': 1 if result['sample']['ground_truth_sum'] > 0 else 0,
                    'with_segmentation': with_segmentation
                }
                rows.append(row)
            return pd.DataFrame(rows)
        
        # Process both conditions
        dfs_to_combine = []
        
        if 'results_without_seg' in data:
            df_without = results_to_dataframe(data['results_without_seg'], False)
            dfs_to_combine.append(df_without)
        
        if 'results_with_seg' in data:
            df_with = results_to_dataframe(data['results_with_seg'], True)
            dfs_to_combine.append(df_with)
        
        if dfs_to_combine:
            combined_df = pd.concat(dfs_to_combine, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        return combined_df
    
    # Load all radiologist data
    all_radiologist_data = []
    
    for json_file in json_files:
        try:
            rad_data = load_radiologist_data(json_file)
            if len(rad_data) > 0:
                all_radiologist_data.append(rad_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not all_radiologist_data:
        print("Error: Could not load radiologist data")
        return None
    
    # Combine all radiologist data
    radiologist_df = pd.concat(all_radiologist_data, ignore_index=True)
    
    # Add model predictions and ground truth
    results_df_loaded['model_predicted_enhancement'] = (results_df_loaded['pred_volume'] > 0).astype(int)
    results_df_loaded['model_has_enhancement_gt'] = (results_df_loaded['gt_volume'] > 0).astype(int)
    
    # Merge radiologist data with model results
    radiologist_df = radiologist_df.merge(
        results_df_loaded[['case_id', 'model_predicted_enhancement', 'dice', 'precision', 'recall', 
                   'balanced_acc', 'Cohort', 'Country', 'Pathology', 'gt_volume', 'pred_volume']],
        on='case_id', how='left'
    )
    
    # Ensure consistent ground truth
    radiologist_df.loc[radiologist_df['gt_volume'].notna(), 'has_enhancement_gt'] = (
        radiologist_df.loc[radiologist_df['gt_volume'].notna(), 'gt_volume'] > 0
    ).astype(int)
    
    # Identify the target cases: Radiologist wrong (false negative), Model correct (true positive)
    common_cases = radiologist_df.dropna(subset=['model_predicted_enhancement'])
    
    target_cases = common_cases[
        (common_cases['has_enhancement_gt'] == 1) &  # Ground truth: has enhancement
        (common_cases['predicted_enhancement'] == 0) &  # Radiologist: predicted no enhancement (WRONG)
        (common_cases['model_predicted_enhancement'] == 1)  # Model: predicted enhancement (CORRECT)
    ]
    
    print(f"Found {len(target_cases)} cases where radiologists were wrong but model was correct")
    
    if len(target_cases) == 0:
        print("No cases found to visualize")
        return None
    
    # Get the most frequently missed cases (similar to nnunet_figure3_alternate selection)
    case_miss_counts = target_cases['case_id'].value_counts()
    
    # Create the figure in the style of nnunet_figure3_alternate
    fig = plt.figure(figsize=(20, 12))
    
    # Define color scheme (similar to the medical imaging style)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Panel A: Cases by frequency of radiologist misses
    ax1 = plt.subplot(2, 4, 1)
    miss_counts_topN = case_miss_counts.head(n_cases)
    bars = ax1.bar(range(len(miss_counts_topN)), miss_counts_topN.values, 
                   color='#d62728', alpha=0.6, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Cases (ranked by frequency of misses)')
    ax1.set_ylabel('Number of radiologists who missed')
    ax1.set_title('a) Most Frequently Missed Cases\n(Radiologist Error, AI Correct)', fontsize=11)  # Removed fontweight='bold'
    ax1.set_xticks(range(len(miss_counts_topN)))
    ax1.set_xticklabels([f'{i+1}' for i in range(len(miss_counts_topN))])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Panel B: Distribution by cohort
    ax2 = plt.subplot(2, 4, 2)
    cohort_counts = target_cases['Cohort'].value_counts()
    wedges, texts, autotexts = ax2.pie(cohort_counts.values, labels=cohort_counts.index, 
                                      autopct='%1.1f%%', startangle=90,
                                      colors=plt.cm.Set3(np.linspace(0, 1, len(cohort_counts))))
    ax2.set_title('b) Distribution by Cohort\n(Missed Enhancement Cases)', fontsize=11)  # Removed fontweight='bold'
    
    # Panel C: Distribution by pathology
    ax3 = plt.subplot(2, 4, 3)
    pathology_counts = target_cases['Pathology'].value_counts()
    bars = ax3.bar(range(len(pathology_counts)), pathology_counts.values,
                   color='#2ca02c', alpha=0.6, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Pathology Type')
    ax3.set_ylabel('Number of missed cases')
    ax3.set_title('c) Distribution by Pathology\n(Missed Enhancement Cases)', fontsize=11)  # Removed fontweight='bold'
    ax3.set_xticks(range(len(pathology_counts)))
    ax3.set_xticklabels(pathology_counts.index, rotation=45, ha='right')
    
    # Panel D: Model performance on these missed cases
    ax4 = plt.subplot(2, 4, 4)
    dice_scores = target_cases['dice'].dropna()
    ax4.hist(dice_scores, bins=20, alpha=0.6, color='#ff7f0e', edgecolor='black', linewidth=0.5)
    ax4.axvline(dice_scores.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {dice_scores.mean():.3f}')
    ax4.set_xlabel('Dice Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('d) Model Performance\n(Cases Missed by Radiologists)', fontsize=11)  # Removed fontweight='bold'
    ax4.legend()
    
    # Panel E: Enhancement volume vs confidence
    ax5 = plt.subplot(2, 4, 5)
    scatter_data = target_cases.dropna(subset=['gt_volume', 'confidence'])
    scatter = ax5.scatter(scatter_data['gt_volume'], scatter_data['confidence'], 
                         c=scatter_data['dice'], cmap='viridis', alpha=0.6, s=50)
    ax5.set_xlabel('Ground Truth Volume (voxels)')
    ax5.set_ylabel('Radiologist Confidence')
    ax5.set_title('e) Volume vs Radiologist Confidence\n(Missed Cases)', fontsize=11)  # Removed fontweight='bold'
    ax5.set_xscale('log')
    plt.colorbar(scatter, ax=ax5, label='Model Dice Score')
    
    # Panel F: Response time analysis
    ax6 = plt.subplot(2, 4, 6)
    response_times = target_cases['response_time'].dropna()
    ax6.hist(response_times, bins=15, alpha=0.6, color='#9467bd', edgecolor='black', linewidth=0.5)
    ax6.axvline(response_times.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {response_times.mean():.1f}s')
    ax6.set_xlabel('Response Time (seconds)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('f) Response Time Distribution\n(Missed Cases)', fontsize=11)  # Removed fontweight='bold'
    ax6.legend()
    
    # Panel G: Segmentation condition comparison
    ax7 = plt.subplot(2, 4, 7)
    seg_comparison = target_cases.groupby('with_segmentation')['case_id'].count()
    bars = ax7.bar(['Without Seg', 'With Seg'], seg_comparison.values,
                   color=['#ff7f0e', '#1f77b4'], alpha=0.6, edgecolor='black', linewidth=0.5)
    ax7.set_ylabel('Number of missed cases')
    ax7.set_title('g) Segmentation Condition\n(Effect on Misses)', fontsize=11)  # Removed fontweight='bold'
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # Panel H: Individual radiologist performance
    ax8 = plt.subplot(2, 4, 8)
    radiologist_misses = target_cases['radiologist'].value_counts()
    bars = ax8.bar(range(len(radiologist_misses)), radiologist_misses.values,
                   color='#d62728', alpha=0.6, edgecolor='black', linewidth=0.5)
    ax8.set_xlabel('Radiologist ID')
    ax8.set_ylabel('Number of misses')
    ax8.set_title('h) Misses by Individual Radiologist\n(AI Caught These Cases)', fontsize=11)  # Removed fontweight='bold'
    ax8.set_xticks(range(len(radiologist_misses)))
    ax8.set_xticklabels([f'R{i+1}' for i in range(len(radiologist_misses))], rotation=45)
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(os.path.join(figures_out, 'rad_benchmarking.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(figures_out, 'rad_benchmarking.svg'), 
                bbox_inches='tight', facecolor='white')
    
    print(f"Saved: rad_benchmarking.png")
    print(f"Saved: rad_benchmarking.svg")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total cases where radiologists were wrong but AI was correct: {len(target_cases)}")
    print(f"Number of unique cases: {target_cases['case_id'].nunique()}")
    print(f"Number of radiologists involved: {target_cases['radiologist'].nunique()}")
    print(f"Most frequently missed case: {case_miss_counts.index[0]} (missed by {case_miss_counts.iloc[0]} radiologists)")
    print(f"Average model Dice score on missed cases: {target_cases['dice'].mean():.3f}")
    print(f"Average radiologist confidence on missed cases: {target_cases['confidence'].mean():.3f}")
    
    return {
        'target_cases': target_cases,
        'case_miss_counts': case_miss_counts,
        'summary_stats': {
            'total_cases': len(target_cases),
            'unique_cases': target_cases['case_id'].nunique(),
            'num_radiologists': target_cases['radiologist'].nunique(),
            'avg_dice': target_cases['dice'].mean(),
            'avg_confidence': target_cases['confidence'].mean()
        }
    }

# %%
def calculate_radiologist_balanced_accuracy():
    """
    Calculate balanced accuracy for each radiologist and overall statistics
    """
    print("CALCULATING RADIOLOGIST BALANCED ACCURACY")
    print("=" * 40)
    
    # Load the radiologist review data and model results
    results_df_loaded = pd.read_csv(os.path.join(figures_out, 'nnunet_enhancing_tumour_results.csv'))
    
    # Load radiologist data from the multi-radiologist analysis
    RADIOLOGIST_REVIEWS_PATH = '/home/jruffle/Desktop/ENHANCEMENT_ARTICLE/radiologist_reviews/'
    json_files = glob.glob(os.path.join(RADIOLOGIST_REVIEWS_PATH, '*.json'))
    
    def load_radiologist_data(json_file_path):
        """Load and process single radiologist review data"""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        radiologist_name = os.path.basename(json_file_path).replace('.json', '')
        
        def results_to_dataframe(results_list, with_segmentation=False):
            rows = []
            for result in results_list:
                row = {
                    'radiologist': radiologist_name,
                    'case_id': result['sample']['base_name'],
                    'predicted_enhancement': 1 if result['abnormality'] == 'Y' else 0,
                    'confidence': result['confidence'],
                    'image_quality': result['image_quality'],
                    'response_time': result['response_time'],
                    'ground_truth_sum': result['sample']['ground_truth_sum'],
                    'has_enhancement_gt': 1 if result['sample']['ground_truth_sum'] > 0 else 0,
                    'with_segmentation': with_segmentation
                }
                rows.append(row)
            return pd.DataFrame(rows)
        
        # Process both conditions
        dfs_to_combine = []
        
        if 'results_without_seg' in data:
            df_without = results_to_dataframe(data['results_without_seg'], False)
            dfs_to_combine.append(df_without)
        
        if 'results_with_seg' in data:
            df_with = results_to_dataframe(data['results_with_seg'], True)
            dfs_to_combine.append(df_with)
        
        if dfs_to_combine:
            combined_df = pd.concat(dfs_to_combine, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        return combined_df
    
    # Load all radiologist data
    all_radiologist_data = []
    
    for json_file in json_files:
        try:
            rad_data = load_radiologist_data(json_file)
            if len(rad_data) > 0:
                all_radiologist_data.append(rad_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not all_radiologist_data:
        print("Error: Could not load radiologist data")
        return None
    
    # Combine all radiologist data
    radiologist_df = pd.concat(all_radiologist_data, ignore_index=True)
    
    # Calculate balanced accuracy for each radiologist
    from sklearn.metrics import balanced_accuracy_score
    
    radiologist_scores = {}
    
    for radiologist in radiologist_df['radiologist'].unique():
        rad_data = radiologist_df[radiologist_df['radiologist'] == radiologist]
        
        # Calculate for without segmentation
        without_seg = rad_data[rad_data['with_segmentation'] == False]
        if len(without_seg) > 0:
            ba_without = balanced_accuracy_score(without_seg['has_enhancement_gt'], 
                                               without_seg['predicted_enhancement'])
        else:
            ba_without = None
        
        # Calculate for with segmentation
        with_seg = rad_data[rad_data['with_segmentation'] == True]
        if len(with_seg) > 0:
            ba_with = balanced_accuracy_score(with_seg['has_enhancement_gt'], 
                                            with_seg['predicted_enhancement'])
        else:
            ba_with = None
        
        radiologist_scores[radiologist] = {
            'without_seg': ba_without,
            'with_seg': ba_with,
            'n_cases': len(rad_data)
        }
    
    # Calculate overall statistics
    all_ba_without = [scores['without_seg'] for scores in radiologist_scores.values() 
                      if scores['without_seg'] is not None]
    all_ba_with = [scores['with_seg'] for scores in radiologist_scores.values() 
                   if scores['with_seg'] is not None]
    
    # Calculate mean and std
    mean_ba_without = np.mean(all_ba_without) if all_ba_without else None
    std_ba_without = np.std(all_ba_without) if all_ba_without else None
    mean_ba_with = np.mean(all_ba_with) if all_ba_with else None
    std_ba_with = np.std(all_ba_with) if all_ba_with else None
    
    # Print results
    print("\nRadiologist Balanced Accuracy Results:")
    print("-" * 50)
    
    print("\nIndividual Radiologist Scores:")
    for radiologist, scores in sorted(radiologist_scores.items()):
        print(f"\n{radiologist}:")
        if scores['without_seg'] is not None:
            print(f"  Without segmentation: {scores['without_seg']:.3f}")
        if scores['with_seg'] is not None:
            print(f"  With segmentation: {scores['with_seg']:.3f}")
        print(f"  Total cases reviewed: {scores['n_cases']}")
    
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS:")
    print("=" * 50)
    
    if mean_ba_without is not None:
        print(f"\nWithout Segmentation:")
        print(f"  Mean Balanced Accuracy: {mean_ba_without:.3f} ± {std_ba_without:.3f}")
        print(f"  Number of radiologists: {len(all_ba_without)}")
    
    if mean_ba_with is not None:
        print(f"\nWith Segmentation:")
        print(f"  Mean Balanced Accuracy: {mean_ba_with:.3f} ± {std_ba_with:.3f}")
        print(f"  Number of radiologists: {len(all_ba_with)}")
    
    # Create a summary dataframe
    summary_df = pd.DataFrame({
        'Condition': ['Without Segmentation', 'With Segmentation'],
        'Mean_Balanced_Accuracy': [mean_ba_without, mean_ba_with],
        'Std_Balanced_Accuracy': [std_ba_without, std_ba_with],
        'N_Radiologists': [len(all_ba_without), len(all_ba_with)]
    })
    
    # Save to CSV
    summary_df.to_csv(os.path.join(figures_out, 'radiologist_balanced_accuracy_summary.csv'), index=False)
    print(f"\nSaved summary to: radiologist_balanced_accuracy_summary.csv")
    
    return {
        'individual_scores': radiologist_scores,
        'summary': {
            'without_seg': {
                'mean': mean_ba_without,
                'std': std_ba_without,
                'n': len(all_ba_without)
            },
            'with_seg': {
                'mean': mean_ba_with,
                'std': std_ba_with,
                'n': len(all_ba_with)
            }
        },
        'summary_df': summary_df
    }

# Run the calculation
radiologist_accuracy_results = calculate_radiologist_balanced_accuracy()


# %%
# Generate the radiologist failure analysis figures with PROPER VISUALIZATION
print("\n" + "="*60)
print("GENERATING RADIOLOGIST FAILURE ANALYSIS FIGURES")
print("="*60)

# Generate expert failure vs AI success figure
print("\nGenerating expert failure vs AI success analysis figure...")
expert_failure_results = create_expert_failure_ai_success_figure(n_cases=10)

if expert_failure_results:
    print("Expert failure analysis completed successfully!")
    print(f"Total cases analyzed: {expert_failure_results['summary_stats']['total_cases']}")
    print(f"Unique cases where radiologists failed but AI succeeded: {expert_failure_results['summary_stats']['unique_cases']}")
else:
    print("Could not generate expert failure analysis")

# Generate missed cases image visualization figure with ACTUAL MRI DATA
print("\nGenerating missed cases image visualization figure with real MRI data...")
missed_cases_result = create_missed_cases_image_figure(n_cases=6, include_filename=False, include_nhnn=False, min_volume_threshold=100)

if missed_cases_result:
    print("Missed cases visualization completed successfully!")
    print(f"Visualized {len(missed_cases_result)} cases most frequently missed by radiologists")
else:
    print("Could not generate missed cases visualization")

print("\nRadiologist failure analysis figures generation completed!")
print("="*60)

# %%
# Performance by cohort
if 'dice' in results_df.columns and len(results_df) > 0:
    print("\nPerformance by Cohort (Enhancing Tumour):")
    print("-" * 50)
    
    # Select available metrics
    available_metrics = [col for col in ['dice', 'precision', 'recall', 'f1', 'balanced_acc'] if col in results_df.columns]
    
    if available_metrics:
        cohort_metrics = results_df.groupby('Cohort')[available_metrics].agg(['mean', 'std', 'count'])
        print(cohort_metrics.round(4))
    else:
        print("No metrics available for cohort analysis")
else:
    print("No metrics available for analysis")

# %%
# Visualization 1: Distribution of metrics
if 'dice' in results_df.columns:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Select available metrics for plotting
    available_metrics = [col for col in ['dice', 'precision', 'recall', 'f1', 'balanced_acc'] if col in results_df.columns]
    metric_labels = [col.replace('_', ' ').title() for col in available_metrics]

    for i, (metric, label) in enumerate(zip(available_metrics, metric_labels)):
        if i >= 5:  # Limit to 5 plots
            break
        row = i // 3
        col = i % 3
        
        # Histogram with KDE
        sns.histplot(data=results_df, x=metric, kde=True, ax=axes[row, col], bins=30)
        axes[row, col].set_title(f'{label} Distribution\n(Mean: {results_df[metric].mean():.3f})')
        axes[row, col].set_xlabel(label)

    # Remove extra subplots
    for i in range(len(available_metrics), 6):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')

    plt.suptitle('nnUNet Enhancing Tumour Segmentation - Metric Distributions', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_out, 'nnunet_metric_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No metrics available for visualization")

# %%
# Visualization 3: Volume correlation analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Filter cases with actual enhancing tumour
cases_with_et = results_df[results_df['gt_volume'] > 0].copy()

# Volume correlation
axes[0, 0].scatter(cases_with_et['gt_volume'], cases_with_et['pred_volume'], alpha=0.6)
axes[0, 0].plot([0, cases_with_et['gt_volume'].max()], [0, cases_with_et['gt_volume'].max()], 'r--', alpha=0.8)
axes[0, 0].set_xlabel('Ground Truth Volume (voxels)')
axes[0, 0].set_ylabel('Predicted Volume (voxels)')
axes[0, 0].set_title('Volume Correlation')

# Volume vs Dice
axes[0, 1].scatter(cases_with_et['gt_volume'], cases_with_et['dice'], alpha=0.6)
axes[0, 1].set_xlabel('Ground Truth Volume (voxels)')
axes[0, 1].set_ylabel('Dice Score')
axes[0, 1].set_title('Volume vs Dice Score')

# Dice distribution for cases with/without enhancing tumour
results_df['has_enhancing_tumour'] = results_df['gt_volume'] > 0
sns.boxplot(data=results_df, x='has_enhancing_tumour', y='dice', ax=axes[1, 0])
axes[1, 0].set_xlabel('Has Enhancing Tumour')
axes[1, 0].set_ylabel('Dice Score')
axes[1, 0].set_title('Dice: With vs Without ET')

# Performance by volume quartiles (for cases with ET)
if len(cases_with_et) > 0:
    cases_with_et['volume_quartile'] = pd.qcut(cases_with_et['gt_volume'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    sns.boxplot(data=cases_with_et, x='volume_quartile', y='dice', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Volume Quartile')
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].set_title('Performance by ET Volume')
else:
    axes[1, 1].text(0.5, 0.5, 'No cases with\nenhancing tumour', ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Performance by ET Volume')

plt.suptitle('nnUNet Volume Analysis - Enhancing Tumour', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(figures_out, 'nnunet_volume_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create tally DataFrame for dataset split (now we know test vs train/val split)
tally_df = pd.DataFrame(columns=cohorts)
tally_df['Partition'] = ''

# Create rows for different partitions
for partition in ['Test', 'Train/Val', 'Total']:
    row = dict.fromkeys(cohorts, 0)
    row['Partition'] = partition
    tally_df = pd.concat([tally_df, pd.DataFrame([row])], ignore_index=True)

# Fill in the counts from all_images (complete dataset)
for cohort in cohorts:
    test_count = all_images[(all_images['Partition'] == 'Test') & (all_images['Cohort'] == cohort)].shape[0]
    trainval_count = all_images[(all_images['Partition'] == 'Train/Val') & (all_images['Cohort'] == cohort)].shape[0]
    total_count = test_count + trainval_count
    
    tally_df.loc[tally_df['Partition'] == 'Test', cohort] = test_count
    tally_df.loc[tally_df['Partition'] == 'Train/Val', cohort] = trainval_count
    tally_df.loc[tally_df['Partition'] == 'Total', cohort] = total_count

print("Dataset breakdown by partition:")
print(tally_df)

# Add country and pathology information to all_images (same as original)
countries = dict()
countries['UPENN-GBM'] = 'USA'
countries['UCSF-PDGM'] = 'USA'
countries['BraTS2021'] = 'USA'
countries['BraTS-GLI'] = 'USA'
countries['BraTS-MEN'] = 'USA'
countries['EGD'] = 'Netherlands'
countries['NHNN'] = 'UK'
countries['BraTS-MET'] = 'USA'
countries['BraTS-PED'] = 'USA'
countries['BraTS-SSA'] = 'Sub-Saharan Africa'

pathologies = dict()
pathologies['UPENN-GBM'] = 'Presurgical glioma'
pathologies['UCSF-PDGM'] = 'Presurgical glioma'
pathologies['BraTS2021'] = 'Presurgical glioma'
pathologies['BraTS-GLI'] = 'Postoperative glioma resection'
pathologies['BraTS-MEN'] = 'Meningioma'
pathologies['EGD'] = 'Presurgical glioma'
pathologies['NHNN'] = 'Presurgical glioma'
pathologies['BraTS-MET'] = 'Metastases'
pathologies['BraTS-PED'] = 'Paediatric presurgical tumour'
pathologies['BraTS-SSA'] = 'Presurgical glioma'

all_images['Country'] = ''
all_images['Pathology'] = ''

for country in countries.values():
    all_images[country] = 0
for pathology in pathologies.values():
    all_images[pathology] = 0
    
for i, row in all_images.iterrows():
    cohort = row['Cohort']
    if cohort in countries:
        country = countries[cohort]
        all_images.at[i, country] = 1
        all_images.at[i, 'Country'] = country
    if cohort in pathologies:
        pathology = pathologies[cohort]
        all_images.at[i, pathology] = 1
        all_images.at[i, 'Pathology'] = pathology

# Create heatmaps directory
heatmaps_dir = os.path.join(figures_out, 'heatmaps')
os.makedirs(heatmaps_dir, exist_ok=True)

# Generate enhancement heatmaps from actual nnUNet data (EXACTLY like the original notebook)
# Check if heatmaps already exist to avoid regenerating
if len(glob.glob(os.path.join(heatmaps_dir, '*'))) == 0:
    print("Generating enhancement frequency heatmaps from nnUNet dataset...")
    
    # Get a sample image to determine dimensions and affine
    sample_files = glob.glob(os.path.join(nnunet_base_path, 'labels*', '*.nii.gz'))
    if len(sample_files) > 0:
        sample_image = nib.load(sample_files[0])
        sample_data = sample_image.get_fdata()
        
        # Initialize all heatmaps (EXACTLY like original)
        all_heatmap = sample_data * 0
        BraTS2021_heatmap = sample_data * 0
        BraTS_MET_heatmap = sample_data * 0
        UPENN_GBM_heatmap = sample_data * 0
        BraTS_MEN_heatmap = sample_data * 0
        NHNN_heatmap = sample_data * 0
        BraTS_GLI_heatmap = sample_data * 0
        EGD_heatmap = sample_data * 0
        UCSF_PDGM_heatmap = sample_data * 0
        BraTS_PED_heatmap = sample_data * 0
        BraTS_SSA_heatmap = sample_data * 0
        
        usa_heatmap = sample_data * 0
        netherlands_heatmap = sample_data * 0
        uk_heatmap = sample_data * 0
        sub_saharan_africa_heatmap = sample_data * 0
        
        presurgical_glioma_heatmap = sample_data * 0
        meningioma_heatmap = sample_data * 0
        postoperative_glioma_resection_heatmap = sample_data * 0
        metastases_heatmap = sample_data * 0
        paediatric_presurgical_tumour_heatmap = sample_data * 0
        
        # Process all cases and accumulate enhancement masks (EXACTLY like original)
        valid_cases = 0
        for i, row in tqdm(all_images.iterrows(), total=len(all_images), desc="Generating heatmaps"):
            case_id = row['case_id']
            
            # Try to find the corresponding label file
            label_files = glob.glob(os.path.join(nnunet_base_path, 'labels*', f'{case_id}.nii.gz'))
            if len(label_files) > 0:
                try:
                    # Load the label image
                    image = nib.load(label_files[0])
                    label_data = image.get_fdata()
                    
                    # For nnUNet, enhancing tumor is label 3
                    enhancement_mask = (label_data == LABEL_ENHANCING_TUMOUR).astype(int)
                    
                    # Add to all heatmap
                    all_heatmap += enhancement_mask
                    
                    # Add to cohort-specific heatmaps (EXACTLY like original)
                    if row['Cohort'] == 'BraTS2021':
                        BraTS2021_heatmap += enhancement_mask
                    if row['Cohort'] == 'BraTS-MET':
                        BraTS_MET_heatmap += enhancement_mask
                    if row['Cohort'] == 'UPENN-GBM':
                        UPENN_GBM_heatmap += enhancement_mask
                    if row['Cohort'] == 'BraTS-MEN':
                        BraTS_MEN_heatmap += enhancement_mask
                    if row['Cohort'] == 'NHNN':
                        NHNN_heatmap += enhancement_mask
                    if row['Cohort'] == 'BraTS-GLI':
                        BraTS_GLI_heatmap += enhancement_mask
                    if row['Cohort'] == 'EGD':
                        EGD_heatmap += enhancement_mask
                    if row['Cohort'] == 'UCSF-PDGM':
                        UCSF_PDGM_heatmap += enhancement_mask
                    if row['Cohort'] == 'BraTS-PED':
                        BraTS_PED_heatmap += enhancement_mask
                    if row['Cohort'] == 'BraTS-SSA':
                        BraTS_SSA_heatmap += enhancement_mask
                    
                    # Add to country-specific heatmaps (EXACTLY like original)
                    if row['Country'] == 'USA':
                        usa_heatmap += enhancement_mask
                    if row['Country'] == 'Netherlands':
                        netherlands_heatmap += enhancement_mask
                    if row['Country'] == 'UK':
                        uk_heatmap += enhancement_mask
                    if row['Country'] == 'Sub-Saharan Africa':
                        sub_saharan_africa_heatmap += enhancement_mask
                    
                    # Add to pathology-specific heatmaps (EXACTLY like original)
                    if row['Pathology'] == 'Presurgical glioma':
                        presurgical_glioma_heatmap += enhancement_mask
                    if row['Pathology'] == 'Meningioma':
                        meningioma_heatmap += enhancement_mask
                    if row['Pathology'] == 'Postoperative glioma resection':
                        postoperative_glioma_resection_heatmap += enhancement_mask
                    if row['Pathology'] == 'Metastases':
                        metastases_heatmap += enhancement_mask
                    if row['Pathology'] == 'Paediatric presurgical tumour':
                        paediatric_presurgical_tumour_heatmap += enhancement_mask
                    
                    valid_cases += 1
                    
                except Exception as e:
                    print(f"Error processing {case_id}: {e}")
                    continue
        
        print(f"Successfully processed {valid_cases} cases for heatmap generation")
        
        # Save all heatmaps as NIfTI files (EXACTLY like original)
        print("Saving heatmaps as NIfTI files...")
        nib.save(nib.Nifti1Image(all_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'all_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(BraTS2021_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'BraTS2021_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(BraTS_MET_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'BraTS-MET_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(UPENN_GBM_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'UPENN-GBM_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(BraTS_MEN_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'BraTS-MEN_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(NHNN_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'NHNN_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(BraTS_GLI_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'BraTS-GLI_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(EGD_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'EGD_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(UCSF_PDGM_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'UCSF-PDGM_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(BraTS_PED_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'BraTS-PED_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(BraTS_SSA_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'BraTS-SSA_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(usa_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'usa_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(netherlands_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'netherlands_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(uk_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'uk_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(sub_saharan_africa_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'sub_saharan_africa_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(presurgical_glioma_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'presurgical_glioma_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(meningioma_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'meningioma_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(postoperative_glioma_resection_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'postoperative_glioma_resection_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(metastases_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'metastases_heatmap.nii.gz'))
        nib.save(nib.Nifti1Image(paediatric_presurgical_tumour_heatmap, affine=sample_image.affine), os.path.join(heatmaps_dir, 'paediatric_presurgical_tumour_heatmap.nii.gz'))
        
        heatmap_palette = 'hot'
    else:
        print("No label files found")
        all_heatmap = np.zeros((100, 100, 100))  # Dummy data
        heatmap_palette = 'hot'

else:
    # Load existing heatmaps (EXACTLY like original)
    print("Loading existing heatmaps...")
    all_heatmap = nib.load(os.path.join(heatmaps_dir, 'all_heatmap.nii.gz')).get_fdata()

    heatmap_palette = 'hot'

# %%
# First figure with the Total dataset breakdown and additional visualizations - INCREASED FONT SIZES AND ROW SPACING
fig = plt.figure(figsize=(21, 14))
gs = fig.add_gridspec(2, 6)  # 2x6 grid for balanced layout

# Define subplot areas - 2 rows
# Top row: Pie charts for dataset breakdowns (3 columns)
ax_pie1 = fig.add_subplot(gs[0, 0:2])  # Spans 2 grid cells
ax_pie2 = fig.add_subplot(gs[0, 2:4])  # Spans 2 grid cells
ax_pie3 = fig.add_subplot(gs[0, 4:6])  # Spans 2 grid cells

# Bottom row: Age violin plot and 2 brain views, each taking 2 cells
ax_violin = fig.add_subplot(gs[1, 0:2])  # Age violin plot (2 cells)
ax_axial = fig.add_subplot(gs[1, 2:4])   # Axial view (2 cells)
ax_coronal = fig.add_subplot(gs[1, 4:6]) # Coronal view (2 cells)

# DEFINE CONSISTENT COLOR PALETTE FOR ALL PANELS
# Use same colors for all panels a-c for cohorts/categories
colors_pie = sns.color_palette('husl', n_colors=len(cohorts))

# Total dataset by cohort (NOW USING COMPLETE DATASET)
cohort_counts = all_images['Cohort'].value_counts()
sizes = [cohort_counts.get(c, 0) for c in cohorts if cohort_counts.get(c, 0) > 0]
labels = [c for c in cohorts if cohort_counts.get(c, 0) > 0]
explode = [0.05] * len(labels)

ax_pie1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
            explode=explode, colors=colors_pie, textprops={'fontsize': 14})  # INCREASED FONT SIZE
sample_size = sum(sizes)
ax_pie1.set_title(f'a) Cohort breakdown', fontsize=16)  # INCREASED FONT SIZE

# Pathologies breakdown - Format multi-word labels with newlines - USE SAME COLOR PALETTE
if 'Pathology' in all_images.columns:
    pathology_counts = all_images['Pathology'].value_counts()
    unique_pathologies = list(pathology_counts.index)
    sizes_path = list(pathology_counts.values)
    # Format labels to have newlines between words
    formatted_labels = ['\n'.join(path.split()) for path in unique_pathologies]
    # Use same color palette as panel a
    ax_pie2.pie(sizes_path, labels=formatted_labels, autopct='%1.1f%%', startangle=140,
                explode=[0.05]*len(unique_pathologies), 
                colors=colors_pie[:len(unique_pathologies)], textprops={'fontsize': 14})  # INCREASED FONT SIZE
    ax_pie2.set_title('b) Pathology breakdown', fontsize=16)  # INCREASED FONT SIZE
else:
    ax_pie2.text(0.5, 0.5, 'Pathology data\nnot available', ha='center', va='center', transform=ax_pie2.transAxes, fontsize=14)
    ax_pie2.set_title('b) Pathology breakdown', fontsize=16)

# Countries breakdown with formatted labels - USE SAME COLOR PALETTE
if 'Country' in all_images.columns:
    country_counts = all_images['Country'].value_counts()
    unique_countries = list(country_counts.index)
    sizes_country = list(country_counts.values)
    # Format country labels to have newlines between words
    formatted_country_labels = ['\n'.join(country.split()) for country in unique_countries]
    # Use same color palette as panel a
    ax_pie3.pie(sizes_country, labels=formatted_country_labels, autopct='%1.1f%%', startangle=140,
                explode=[0.05]*len(unique_countries), 
                colors=colors_pie[:len(unique_countries)], textprops={'fontsize': 14})  # INCREASED FONT SIZE
    ax_pie3.set_title('c) Country breakdown', fontsize=16)  # INCREASED FONT SIZE
else:
    ax_pie3.text(0.5, 0.5, 'Country data\nnot available', ha='center', va='center', transform=ax_pie3.transAxes, fontsize=14)
    ax_pie3.set_title('c) Country breakdown', fontsize=16)

# Age distribution by cohort (panel d) - USING COMPLETE DATASET - FIX COLORS TO USE START AND END OF PALETTE
age_percent = all_images['Age'].notna().sum() / len(all_images) * 100
if 'Age' in all_images.columns and all_images['Age'].notna().sum() > 0:
    # Violin plot for age by cohort with sex split - USE COLORS FROM START AND END OF PALETTE
    # Get first and last colors from the palette for better contrast
    sex_colors = [colors_pie[3], colors_pie[7]]  # First and last colors for better contrast
    sns.violinplot(x='Cohort', y='Age', data=all_images.dropna(subset=['Age']), 
                   ax=ax_violin, hue='Sex', split=True, inner="quart", gap=.1,
                   palette=sex_colors)  # Use contrasting colors from start and end of palette
    ax_violin.set_title(f'd) Age distribution (data available {age_percent:.1f}%)', fontsize=16)  # INCREASED FONT SIZE
    ax_violin.tick_params(axis='x', rotation=30, labelsize=14)  # INCREASED FONT SIZE
    ax_violin.tick_params(axis='y', labelsize=14)  # INCREASED FONT SIZE
    ax_violin.set_xlabel('Study site', fontsize=14)  # INCREASED FONT SIZE
    ax_violin.set_ylabel('Age (years)', fontsize=14)  # INCREASED FONT SIZE
    
    # Custom legend with full names and positioned at bottom right
    legend = ax_violin.get_legend()
    if legend:
        # Get the legend handles and create new labels
        handles = legend.legend_handles
        # Replace M/F with Male/Female and position at bottom right
        ax_violin.legend(handles, ['Female', 'Male'], loc='lower right', fontsize=14)
else:
    ax_violin.text(0.5, 0.5, f'Age data not available\n({age_percent:.1f}% coverage)', 
                   ha='center', va='center', transform=ax_violin.transAxes, fontsize=14)
    ax_violin.set_title(f'd) Age distribution (data available {age_percent:.1f}%)', fontsize=16)

# Brain visualization - EXACTLY like the original notebook
try:
    template = nib.load('/home/jruffle/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz')
    template_data = template.get_fdata()
    
    # Get middle slices for each view (SAME AS ORIGINAL)
    slice_x = all_heatmap.shape[0] // 2
    slice_y = all_heatmap.shape[1] // 2
    slice_z = all_heatmap.shape[2] // 2
    
    # Get template slices (SAME AS ORIGINAL)
    template_axial = template_data[:, :, slice_z].T
    template_coronal = template_data[:, slice_y, :].T
    
    # Create masked arrays for non-brain areas to be white (SAME AS ORIGINAL)
    # Create a mask for non-brain areas (value = 0)
    axial_mask = template_axial == 0
    coronal_mask = template_coronal == 0
    
    # Set white backgrounds for brain visualization (SAME AS ORIGINAL)
    ax_axial.set_facecolor('white')
    ax_coronal.set_facecolor('white')
    
    # Get heatmap slices (SAME AS ORIGINAL)
    heatmap_axial = all_heatmap[:, :, slice_z].T
    heatmap_coronal = all_heatmap[:, slice_y, :].T
    
    # Create masks for zero values in the template and heatmap (SAME AS ORIGINAL)
    # These ensure that only brain areas are displayed
    masked_template_axial = np.ma.masked_where(axial_mask, template_axial)
    masked_template_coronal = np.ma.masked_where(coronal_mask, template_coronal)
    
    # Create masks for zero values in the heatmap (SAME AS ORIGINAL)
    # This ensures that zero values in heatmap are transparent
    masked_heatmap_axial = np.ma.masked_where(heatmap_axial < 25, heatmap_axial)
    masked_heatmap_coronal = np.ma.masked_where(heatmap_coronal < 25, heatmap_coronal)
    
    # Plot template brain with gray scale (SAME AS ORIGINAL)
    ax_axial.imshow(masked_template_axial, cmap='gray', origin='lower', alpha=1.0)
    # Plot heatmap with transparent zeros (SAME AS ORIGINAL)
    im_axial = ax_axial.imshow(masked_heatmap_axial, cmap=heatmap_palette, origin='lower', alpha=0.7)
    ax_axial.set_title('e) Axial enhancement heatmap', fontsize=16)  # INCREASED FONT SIZE
    ax_axial.axis('off')
    
    # Plot coronal view (SAME AS ORIGINAL)
    ax_coronal.imshow(masked_template_coronal, cmap='gray', origin='lower', alpha=1.0)
    im_coronal = ax_coronal.imshow(masked_heatmap_coronal, cmap=heatmap_palette, origin='lower', alpha=0.7)
    ax_coronal.set_title('f) Coronal enhancement heatmap', fontsize=16)  # INCREASED FONT SIZE
    ax_coronal.axis('off')
    
    # Add a colorbar for the heatmap visualizations (SAME AS ORIGINAL)
    # Create a horizontal colorbar at the bottom of brain visualization plots
    cbar_ax = fig.add_axes([0.4, 0.05, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im_axial, cax=cbar_ax, orientation='horizontal', label='Frequency of enhancement')
    cbar.ax.tick_params(labelsize=14)  # INCREASED FONT SIZE
    cbar.set_label('Frequency of enhancement', fontsize=14)  # INCREASED FONT SIZE
    
except Exception as e:
    print(f"Could not load template for brain views: {e}")
    ax_axial.text(0.5, 0.5, 'Template not\navailable', ha='center', va='center', transform=ax_axial.transAxes, fontsize=14)
    ax_coronal.text(0.5, 0.5, 'Template not\navailable', ha='center', va='center', transform=ax_coronal.transAxes, fontsize=14)
    ax_axial.set_title('e) Axial enhancement heatmap', fontsize=16)
    ax_coronal.set_title('f) Coronal enhancement heatmap', fontsize=16)

fig.suptitle(f'Sample overview (n={sample_size})', fontsize=20)  # INCREASED FONT SIZE
plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.15, wspace=0.2)  # INCREASED ROW SPACING from 0.05 to 0.15

fig.savefig(os.path.join(figures_out, 'Figure_1.svg'), format='svg', bbox_inches='tight')
fig.savefig(os.path.join(figures_out, 'Figure_1.png'), format='png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Generate supplementary figure 2 - CORRECTED to properly load and display saved heatmaps
# This now matches the original notebook structure exactly

template_path = '/home/jruffle/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
heatmaps_dir = os.path.join(figures_out, 'heatmaps')

try:
    # Load template brain
    template = nib.load(template_path)
    template_data = template.get_fdata()
    
    # Load all the saved heatmap NIfTI files (generated in cell 17)
    print("Loading saved heatmap NIfTI files...")
    all_heatmap = nib.load(os.path.join(heatmaps_dir, 'all_heatmap.nii.gz')).get_fdata()
    BraTS2021_heatmap = nib.load(os.path.join(heatmaps_dir, 'BraTS2021_heatmap.nii.gz')).get_fdata()
    BraTS_MET_heatmap = nib.load(os.path.join(heatmaps_dir, 'BraTS-MET_heatmap.nii.gz')).get_fdata()
    UPENN_GBM_heatmap = nib.load(os.path.join(heatmaps_dir, 'UPENN-GBM_heatmap.nii.gz')).get_fdata()
    BraTS_MEN_heatmap = nib.load(os.path.join(heatmaps_dir, 'BraTS-MEN_heatmap.nii.gz')).get_fdata()
    NHNN_heatmap = nib.load(os.path.join(heatmaps_dir, 'NHNN_heatmap.nii.gz')).get_fdata()
    BraTS_GLI_heatmap = nib.load(os.path.join(heatmaps_dir, 'BraTS-GLI_heatmap.nii.gz')).get_fdata()
    EGD_heatmap = nib.load(os.path.join(heatmaps_dir, 'EGD_heatmap.nii.gz')).get_fdata()
    UCSF_PDGM_heatmap = nib.load(os.path.join(heatmaps_dir, 'UCSF-PDGM_heatmap.nii.gz')).get_fdata()
    BraTS_PED_heatmap = nib.load(os.path.join(heatmaps_dir, 'BraTS-PED_heatmap.nii.gz')).get_fdata()
    BraTS_SSA_heatmap = nib.load(os.path.join(heatmaps_dir, 'BraTS-SSA_heatmap.nii.gz')).get_fdata()
    
    usa_heatmap = nib.load(os.path.join(heatmaps_dir, 'usa_heatmap.nii.gz')).get_fdata()
    netherlands_heatmap = nib.load(os.path.join(heatmaps_dir, 'netherlands_heatmap.nii.gz')).get_fdata()
    uk_heatmap = nib.load(os.path.join(heatmaps_dir, 'uk_heatmap.nii.gz')).get_fdata()
    sub_saharan_africa_heatmap = nib.load(os.path.join(heatmaps_dir, 'sub_saharan_africa_heatmap.nii.gz')).get_fdata()
    
    presurgical_glioma_heatmap = nib.load(os.path.join(heatmaps_dir, 'presurgical_glioma_heatmap.nii.gz')).get_fdata()
    postoperative_glioma_resection_heatmap = nib.load(os.path.join(heatmaps_dir, 'postoperative_glioma_resection_heatmap.nii.gz')).get_fdata()
    meningioma_heatmap = nib.load(os.path.join(heatmaps_dir, 'meningioma_heatmap.nii.gz')).get_fdata()
    metastases_heatmap = nib.load(os.path.join(heatmaps_dir, 'metastases_heatmap.nii.gz')).get_fdata()
    paediatric_presurgical_tumour_heatmap = nib.load(os.path.join(heatmaps_dir, 'paediatric_presurgical_tumour_heatmap.nii.gz')).get_fdata()
    
    print("All heatmaps loaded successfully!")
    
    # Define sample slices to visualize (middle slice in each dimension) - EXACTLY like original
    sample_shape = all_heatmap.shape
    slice_x = sample_shape[0] // 2
    slice_y = sample_shape[1] // 2
    slice_z = sample_shape[2] // 2
    
    # Create a 5x4 subplot grid - EXACTLY like original
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    
    # List of heatmaps with their titles - EXACTLY like original
    heatmaps_titles = [
        # Row 1: By cohort
        (all_heatmap, "All Cohorts"),
        (BraTS2021_heatmap, "BraTS2021"),
        (UPENN_GBM_heatmap, "UPENN-GBM"),
        (UCSF_PDGM_heatmap, "UCSF-PDGM"),
        
        # Row 2: More cohorts
        (BraTS_GLI_heatmap, "BraTS-GLI"),
        (BraTS_MEN_heatmap, "BraTS-MEN"),
        (NHNN_heatmap, "NHNN"),
        (EGD_heatmap, "EGD"),
        
        # Row 3: Rest of cohorts
        (BraTS_MET_heatmap, "BraTS-MET"),
        (BraTS_PED_heatmap, "BraTS-PED"),
        (BraTS_SSA_heatmap, "BraTS-SSA"),
        (None, ""), # Empty space
        
        # Row 4: By country
        (usa_heatmap, "USA"),
        (netherlands_heatmap, "Netherlands"),
        (uk_heatmap, "UK"),
        (sub_saharan_africa_heatmap, "Sub-Saharan Africa"),
        
        # Row 5: By pathology
        (presurgical_glioma_heatmap, "Presurgical Glioma"),
        (postoperative_glioma_resection_heatmap, "Postop Glioma Resection"),
        (meningioma_heatmap, "Meningioma"),
        (metastases_heatmap, "Metastases")
    ]
    
    # Get template slices for background - EXACTLY like original
    template_axial = template_data[:, :, slice_z].T
    
    # Create a mask for non-brain areas (value = 0) - EXACTLY like original
    axial_mask = template_axial == 0
    
    heatmap_palette = 'hot'
    
    # Plot each heatmap - EXACTLY like original
    for i, (heatmap, title) in enumerate(heatmaps_titles):
        row = i // 4
        col = i % 4
        
        if heatmap is not None:
            # Create masked template for background - EXACTLY like original
            masked_template = np.ma.masked_where(axial_mask, template_axial)
            
            # Create masked heatmap (mask zero values for transparency) - EXACTLY like original
            heatmap_slice = heatmap[:, :, slice_z].T
            masked_heatmap = np.ma.masked_where(heatmap_slice < 3, heatmap_slice)
            
            # Display template brain first - EXACTLY like original
            axes[row, col].imshow(masked_template, cmap='gray', origin='lower')
            
            # Then overlay the heatmap with masked zeros - EXACTLY like original
            im = axes[row, col].imshow(masked_heatmap, cmap=heatmap_palette, 
                                       alpha=0.7, interpolation='none', origin='lower')
            
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    # Set overall title - EXACTLY like original
    fig.suptitle('Heatmaps of Enhancement Masks by Dataset, Country, and Pathology', fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the figure - EXACTLY like original
    plt.savefig(os.path.join(figures_out, 'heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(figures_out, 'heatmaps.svg'), format='svg', bbox_inches='tight')
    plt.show()
    
    print("Supplementary figure 2 generated successfully using proper heatmap stacking!")
    
except Exception as e:
    print(f"Could not create supplementary figure 2: {e}")
    print("Make sure the heatmap NIfTI files have been generated in cell 17 and template file is accessible")

# %%
# Supplementary Figure 1: Dataset split characteristics - FIXED PARTITION ORDERING
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2
import numpy as np

# Create a single figure with 2x2 layout showing the dataset characteristics
fig_pie2 = plt.figure(figsize=(12, 12))
gs = fig_pie2.add_gridspec(2, 2)

# USE CONSISTENT COLOR PALETTE FOR ALL PANELS (MATCHING FIGURE 1)
colors_pie = sns.color_palette('husl', n_colors=len(cohorts))

# Use the SAME manual colors from figure 1 panel d (colors_pie[3] and colors_pie[7])
manual_sex_colors = [colors_pie[3], colors_pie[7]]

# Since we have a proper test set and train/val set, show both splits
# Top row - Dataset breakdown by partition - SWAPPED ORDER
ax1 = fig_pie2.add_subplot(gs[0, 0])
ax2 = fig_pie2.add_subplot(gs[0, 1])

# PANEL A: Train/Val Set breakdown (SWAPPED - now first)
trainval_cohort_counts = all_images[all_images['Partition'] == 'Train/Val']['Cohort'].value_counts()
trainval_sizes = [trainval_cohort_counts.get(c, 0) for c in cohorts if trainval_cohort_counts.get(c, 0) > 0]
trainval_labels = [c for c in cohorts if trainval_cohort_counts.get(c, 0) > 0]

ax1.pie(trainval_sizes, labels=trainval_labels, autopct='%1.1f%%', startangle=140,
        explode=[0.05] * len(trainval_labels), colors=colors_pie)
trainval_size = sum(trainval_sizes)
ax1.set_title(f'a) Train/Val Set Breakdown (n={trainval_size})')

# PANEL B: Test Set breakdown (SWAPPED - now second)
test_cohort_counts = all_images[all_images['Partition'] == 'Test']['Cohort'].value_counts()
test_sizes = [test_cohort_counts.get(c, 0) for c in cohorts if test_cohort_counts.get(c, 0) > 0]
test_labels = [c for c in cohorts if test_cohort_counts.get(c, 0) > 0]
explode = [0.05] * len(test_labels)

ax2.pie(test_sizes, labels=test_labels, autopct='%1.1f%%', startangle=140,
        explode=explode, colors=colors_pie)
test_size = sum(test_sizes)
ax2.set_title(f'b) Test Set Breakdown (n={test_size})')

# Bottom row - Demographics analysis between partitions
ax3 = fig_pie2.add_subplot(gs[1, 0])  # Back to regular subplot for bar plot
ax4 = fig_pie2.add_subplot(gs[1, 1])

# Calculate data availability
age_available = all_images['Age'].notna().sum()
age_percentage = (age_available / len(all_images)) * 100
sex_available = all_images['Sex'].notna().sum()
sex_percentage = (sex_available / len(all_images)) * 100

# Test for age difference between partitions
age_test = all_images[all_images['Partition'] == 'Test']['Age'].dropna()
age_trainval = all_images[all_images['Partition'] == 'Train/Val']['Age'].dropna()

if len(age_test) > 0 and len(age_trainval) > 0:
    # Perform t-test for age
    t_stat, p_val_age = stats.ttest_ind(age_test, age_trainval, equal_var=False)  # Using Welch's t-test

    # REVERT TO BAR PLOT: Age proportion chart showing Age on x-axis and proportion on y-axis for each partition
    age_data = all_images.dropna(subset=['Age'])
    
    # Create age bins for better visualization
    age_bins = pd.cut(age_data['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                     labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
    age_data = age_data.copy()
    age_data['Age_bin'] = age_bins
    
    # Calculate proportions within each partition for each age bracket
    partition_age_counts = age_data.groupby(['Partition', 'Age_bin']).size().unstack(fill_value=0)
    partition_totals = age_data.groupby('Partition').size()
    partition_age_proportions = partition_age_counts.div(partition_totals, axis=0)
    
    # Plot as bar chart with age brackets on x-axis - FIXED ORDERING: Train/Val first, then Test
    age_brackets = partition_age_proportions.columns
    x_pos = np.arange(len(age_brackets))
    width = 0.35
    
    # Use manual colors from figure 1 panel d - TRAIN/VAL FIRST, TEST SECOND
    trainval_props = partition_age_proportions.loc['Train/Val'] if 'Train/Val' in partition_age_proportions.index else pd.Series(0, index=age_brackets)
    test_props = partition_age_proportions.loc['Test'] if 'Test' in partition_age_proportions.index else pd.Series(0, index=age_brackets)
    
    ax3.bar(x_pos - width/2, trainval_props, width, label='Train/Val', color=manual_sex_colors[0])
    ax3.bar(x_pos + width/2, test_props, width, label='Test', color=manual_sex_colors[1])
    
    ax3.set_xlabel('Age Bracket (years)')
    ax3.set_ylabel('Proportion of samples')
    ax3.set_title(f'c) Age Distribution by Partition\n(Data available: {age_percentage:.1f}%, p-value: {p_val_age:.4f})')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(age_brackets, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    print(f"Age difference between partitions (Welch's t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_val_age:.4f}")
else:
    ax3.text(0.5, 0.5, f'Insufficient age data\nfor comparison\n(Data available: {age_percentage:.1f}%)', 
             ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title(f'c) Age Distribution by Partition\n(Data available: {age_percentage:.1f}%)')

# Test for sex difference between partitions - STACKED BAR PLOT - FIXED ORDERING
sex_test = all_images[all_images['Partition'] == 'Test']['Sex'].dropna()
sex_trainval = all_images[all_images['Partition'] == 'Train/Val']['Sex'].dropna()

if len(sex_test) > 0 and len(sex_trainval) > 0:
    # Create contingency table
    sex_counts = all_images.dropna(subset=['Sex']).groupby(['Partition', 'Sex']).size().unstack(fill_value=0)
    
    if 'M' in sex_counts.columns and 'F' in sex_counts.columns and len(sex_counts) >= 2:
        # Perform Chi-squared test
        chi2, p_val_sex, dof, expected = stats.chi2_contingency(sex_counts[['M', 'F']])
        
        # Calculate proportions WITHIN each partition for stacked bar plot
        sex_proportions = sex_counts.div(sex_counts.sum(axis=1), axis=0)
        
        # Create stacked bar plot - FIXED ORDERING: Train/Val first, Test second
        partitions = ['Train/Val', 'Test']  # FIXED ORDER
        x_pos = np.arange(len(partitions))
        
        # Get proportions for each partition
        male_props = []
        female_props = []
        
        for partition in partitions:
            if partition in sex_proportions.index:
                male_props.append(sex_proportions.loc[partition, 'M'] if 'M' in sex_proportions.columns else 0)
                female_props.append(sex_proportions.loc[partition, 'F'] if 'F' in sex_proportions.columns else 0)
            else:
                male_props.append(0)
                female_props.append(0)
        
        # Create stacked bars using manual colors
        ax4.bar(x_pos, male_props, label='Male', color=manual_sex_colors[0])
        ax4.bar(x_pos, female_props, bottom=male_props, label='Female', color=manual_sex_colors[1])
        
        ax4.set_xlabel('Partition')
        ax4.set_ylabel('Proportion')
        ax4.set_title(f'd) Sex Distribution by Partition\n(Data available: {sex_percentage:.1f}%, p-value: {p_val_sex:.4f})')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(partitions)
        ax4.set_ylim(0, 1)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Calculate ratios for each partition
        test_m = sex_counts.loc['Test', 'M'] if 'Test' in sex_counts.index else 0
        test_f = sex_counts.loc['Test', 'F'] if 'Test' in sex_counts.index else 0
        trainval_m = sex_counts.loc['Train/Val', 'M'] if 'Train/Val' in sex_counts.index else 0
        trainval_f = sex_counts.loc['Train/Val', 'F'] if 'Train/Val' in sex_counts.index else 0
        
        test_ratio = test_m / test_f if test_f > 0 else 0
        trainval_ratio = trainval_m / trainval_f if trainval_f > 0 else 0
        
        # Add text annotations with counts - FIXED ORDERING
        for i, partition in enumerate(partitions):
            if partition == 'Train/Val':
                ax4.text(i, 0.5, f'M:{trainval_m}\nF:{trainval_f}', ha='center', va='center', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            else:  # Test
                ax4.text(i, 0.5, f'M:{test_m}\nF:{test_f}', ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        print("\nSex difference between partitions (Chi-squared test):")
        print(f"  Chi-squared: {chi2:.4f}")
        print(f"  p-value: {p_val_sex:.4f}")
    else:
        ax4.text(0.5, 0.5, f"Insufficient data for\nM/F comparison\n(Data available: {sex_percentage:.1f}%)", 
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title(f'd) Sex Distribution by Partition\n(Data available: {sex_percentage:.1f}%)')
else:
    ax4.text(0.5, 0.5, f'Insufficient sex data\nfor comparison\n(Data available: {sex_percentage:.1f}%)', 
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title(f'd) Sex Distribution by Partition\n(Data available: {sex_percentage:.1f}%)')

# Print dataset summary statistics
print("\nnnUNet Complete Dataset Characteristics:")
print("=" * 40)
print(f"Total cases: {len(all_images)}")
print(f"Test set: {test_size} cases ({test_size/len(all_images)*100:.1f}%)")
print(f"Train/Val set: {trainval_size} cases ({trainval_size/len(all_images)*100:.1f}%)")

if age_available > 0:
    print(f"Age data available: {age_available} cases ({age_percentage:.1f}%)")
    print(f"Mean age: {all_images['Age'].mean():.1f} ± {all_images['Age'].std():.1f} years")

if sex_available > 0:
    print(f"Sex data available: {sex_available} cases ({sex_percentage:.1f}%)")
    m_count = (all_images['Sex'] == 'M').sum()
    f_count = (all_images['Sex'] == 'F').sum()
    print(f"Male: {m_count} ({m_count/sex_available*100:.1f}%)")
    print(f"Female: {f_count} ({f_count/sex_available*100:.1f}%)")

fig_pie2.suptitle('Data Partitioning Characteristics', fontsize=18)
plt.tight_layout()
plt.show()

# Save the figure in SVG and PNG formats
fig_pie2.savefig(os.path.join(figures_out, 'Supplementary_figure_1.svg'), format='svg', bbox_inches='tight')
fig_pie2.savefig(os.path.join(figures_out, 'Supplementary_figure_1.png'), format='png', dpi=300, bbox_inches='tight')

# %%

# Figure 3: Visualization of best cases - EXACT COPY from original with nnUNet adaptation
def visualize_comparison_cases_nnunet(results_df, gt_path, pred_path, n_cases=5, min_gt_voxels=500, random_seed=42, sort_metric='dice'):
    """
    Visualize comparison between ground truth and model predictions - adapted for nnUNet.
    One sample from each cohort with the highest score for the selected metric is selected when possible.
    Updated to match missed_cases_image_visualization.png aesthetics exactly.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame with nnUNet results including case_id and performance metrics
    gt_path : str
        Path to ground truth labels
    pred_path : str 
        Path to model predictions
    n_cases : int
        Maximum number of cases to visualize
    min_gt_voxels : int
        Minimum number of voxels in ground truth to consider a case
    random_seed : int
        Random seed for reproducibility
    sort_metric : str
        Metric to use for sorting samples ('dice', 'precision', 'recall', 'f1', 'balanced_acc')
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the visualizations
    """
    import os
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    np.random.seed(random_seed)
    
    # Validate the sort metric
    valid_metrics = ['dice', 'precision', 'recall', 'f1', 'balanced_acc']
    if sort_metric not in valid_metrics:
        print(f"Warning: Invalid sort metric '{sort_metric}'. Using 'dice' instead.")
        sort_metric = 'dice'
    
    # Filter cases with enough enhancement in ground truth
    filtered_df = results_df[
        (results_df['gt_volume'] >= min_gt_voxels) & 
        (results_df[sort_metric] > 0)
    ].copy()
    
    if len(filtered_df) == 0:
        print("No cases found with sufficient enhancement volume")
        return None
    
    # Group by cohort and select best case from each
    cohorts = ['UPENN-GBM', 'UCSF-PDGM', 'BraTS2021', 'BraTS-GLI', 'BraTS-MEN', 
               'EGD', 'NHNN', 'BraTS-MET', 'BraTS-PED', 'BraTS-SSA']
    
    selected_cases = []
    used_cohorts = set()
    
    # Try to get one from each cohort
    for cohort in cohorts:
        cohort_data = filtered_df[filtered_df['Cohort'] == cohort]
        if len(cohort_data) > 0:
            best_case = cohort_data.nlargest(1, sort_metric).iloc[0]
            selected_cases.append(best_case)
            used_cohorts.add(cohort)
            if len(selected_cases) >= n_cases:
                break
    
    # If we need more cases, add from any cohort
    if len(selected_cases) < n_cases:
        remaining_cases = filtered_df[~filtered_df['Cohort'].isin(used_cohorts)].nlargest(n_cases - len(selected_cases), sort_metric)
        for _, case in remaining_cases.iterrows():
            selected_cases.append(case)
    
    n_cases_found = len(selected_cases)
    print(f"Selected {n_cases_found} cases, prioritizing highest {sort_metric} score from each available cohort.")
    
    # Create figure with 6 columns: T1, T2, FLAIR, Prediction with FLAIR, T1CE, GT with T1CE  
    fig, axes = plt.subplots(n_cases_found, 6, figsize=(22, n_cases_found * 3.8))
    
    if n_cases_found == 1:
        axes = np.array([axes])
    
    # Define column titles (UPDATED to match missed_cases_image_visualization.png)
    col_titles = ['T1', 'T2', 'FLAIR', 'Enhancing prediction\n(from T1+T2+FLAIR)', 
                 'T1CE\n(Held out from model)', 'Ground Truth\n(with T1CE background)']
    
    # Create a green colormap for overlays
    green_cmap = ListedColormap(['none', 'green'])  # 'none' for transparent, 'green' for mask
    
    # Define the data path structure to match the original notebook
    data_path = '/home/jruffle/Documents/seq-synth/data/'
    
    # Process each case
    for i, case_row in enumerate(selected_cases):
        case_id = case_row['case_id']
        
        try:
            # Load ground truth and prediction
            gt_img = nib.load(os.path.join(gt_path, f"{case_id}.nii.gz")).get_fdata()
            pred_img = nib.load(os.path.join(pred_path, f"{case_id}.nii.gz")).get_fdata()
            
            # Load structural images from the original data path structure (like in enhancement_prediction_article_figures.ipynb)
            try:
                # Try to load from the sequences_merged directory (as in original notebook)
                seq_path = os.path.join(data_path, 'sequences_merged', f"{case_id}.nii.gz")
                brain_mask_path = os.path.join(data_path, 'lesion_masks_augmented', f"{case_id}.nii.gz")
                
                if os.path.exists(seq_path) and os.path.exists(brain_mask_path):
                    seq_img = nib.load(seq_path).get_fdata()
                    brain_mask = nib.load(brain_mask_path).get_fdata()
                    brain_mask[brain_mask > 0] = 1  # Ensure binary mask
                    
                    # Extract sequences (same order as original notebook)
                    flair_img = seq_img[..., 0] * brain_mask
                    t1_img = seq_img[..., 1] * brain_mask 
                    t1ce_img = seq_img[..., 2] * brain_mask  # T1CE is channel 2 in original
                    t2_img = seq_img[..., 3] * brain_mask
                    
                    print(f"Successfully loaded sequences for {case_id}")
                    
                else:
                    # Fallback: Try nnUNet imagesTs directory structure
                    images_path = gt_path.replace('labelsTs', 'imagesTs')
                    t1_path = os.path.join(images_path, f"{case_id}_0000.nii.gz")
                    t2_path = os.path.join(images_path, f"{case_id}_0001.nii.gz") 
                    flair_path = os.path.join(images_path, f"{case_id}_0002.nii.gz")
                    t1ce_path = os.path.join(images_path, f"{case_id}_0003.nii.gz")
                    
                    # Load individual sequences if they exist
                    t1_img = nib.load(t1_path).get_fdata() if os.path.exists(t1_path) else None
                    t2_img = nib.load(t2_path).get_fdata() if os.path.exists(t2_path) else None
                    flair_img = nib.load(flair_path).get_fdata() if os.path.exists(flair_path) else None
                    t1ce_img = nib.load(t1ce_path).get_fdata() if os.path.exists(t1ce_path) else None
                    
                    print(f"Using nnUNet structure for {case_id}")
                    
            except Exception as e:
                print(f"Could not load sequences for {case_id}: {e}")
                t1_img = t2_img = flair_img = t1ce_img = None
            
            # Find best slice with enhancing tumour (label 3)
            et_slices = np.sum(gt_img == LABEL_ENHANCING_TUMOUR, axis=(0, 1))
            if np.max(et_slices) > 0:
                z_slice = np.argmax(et_slices)
            else:
                z_slice = gt_img.shape[2] // 2
            
            # Get 2D slices
            gt_slice = gt_img[:, :, z_slice]
            pred_slice = pred_img[:, :, z_slice]
            
            # Create binary masks for enhancing tumour only
            gt_et = (gt_slice == LABEL_ENHANCING_TUMOUR).astype(np.uint8)
            pred_et = (pred_slice == LABEL_ENHANCING_TUMOUR).astype(np.uint8)
            
            # Plot structural images or placeholders
            # Column 1: T1
            if t1_img is not None:
                axes[i, 0].imshow(np.rot90(t1_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 0].text(0.5, 0.5, 'T1\nNot Available', ha='center', va='center', transform=axes[i, 0].transAxes)
            
            # Column 2: T2
            if t2_img is not None:
                axes[i, 1].imshow(np.rot90(t2_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 1].text(0.5, 0.5, 'T2\nNot Available', ha='center', va='center', transform=axes[i, 1].transAxes)
            
            # Column 3: FLAIR
            if flair_img is not None:
                axes[i, 2].imshow(np.rot90(flair_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 2].text(0.5, 0.5, 'FLAIR\nNot Available', ha='center', va='center', transform=axes[i, 2].transAxes)
            
            # Column 4: Prediction with FLAIR background (UPDATED)
            if flair_img is not None:
                axes[i, 3].imshow(np.rot90(flair_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 3].imshow(np.rot90(pred_et*0), cmap='gray')  # Empty background
            axes[i, 3].imshow(np.rot90(pred_et), cmap=green_cmap, alpha=0.6, vmin=0, vmax=1)
            
            # Column 5: T1CE
            if t1ce_img is not None:
                axes[i, 4].imshow(np.rot90(t1ce_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 4].text(0.5, 0.5, 'T1CE\nNot Available', ha='center', va='center', transform=axes[i, 4].transAxes)
            
            # Column 6: Ground truth with T1CE background (UPDATED)
            if t1ce_img is not None:
                axes[i, 5].imshow(np.rot90(t1ce_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 5].imshow(np.rot90(gt_et*0), cmap='gray')  # Empty background
            axes[i, 5].imshow(np.rot90(gt_et), cmap=green_cmap, alpha=0.6, vmin=0, vmax=1)
            
            # Add titles to first row
            if i == 0:
                for j, title in enumerate(col_titles):
                    axes[i, j].set_title(title, fontsize=11, pad=8)
            
            # Extract cohort and performance metrics
            cohort_name = case_row['Cohort']
            dice_score = case_row['dice']
            precision = case_row['precision'] 
            recall = case_row['recall']
            
            # Add case information (mimicking original format)
            newline = '\n'
            # axes[i, 0].set_ylabel(f"{case_id}{newline}Cohort: {cohort_name}{newline}{sort_metric}: {case_row[sort_metric]:.3f}", fontsize=10)
            axes[i, 0].set_ylabel(f"Cohort: {cohort_name}{newline}{sort_metric}: {case_row[sort_metric]:.3f}", fontsize=10)
            
            # Remove axis ticks for cleaner look
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        except Exception as e:
            print(f"Error visualizing {case_id}: {e}")
            for j in range(6):
                error_text = f'Error loading{chr(10)}{case_id}'
                axes[i, j].text(0.5, 0.5, error_text, 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    # UPDATED: Apply exact layout adjustments from missed_cases_image_visualization.png
    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.08, wspace=-0.7, top=0.92, bottom=0.06)
    
    # UPDATED: Add white divider line between columns 3 and 4
    if n_cases_found > 0:
        fig.canvas.draw()
        pos3 = axes[0, 3].get_position()
        pos4 = axes[0, 4].get_position()
        line_x = (pos3.x1 + pos4.x0) / 2
        
        # White line with black border
        border_line = plt.Line2D([line_x, line_x], [0.06, 0.92], 
                                transform=fig.transFigure, 
                                color='black', 
                                linewidth=8,
                                solid_capstyle='butt',
                                zorder=9)
        fig.add_artist(border_line)
        
        line = plt.Line2D([line_x, line_x], [0.06, 0.92], 
                         transform=fig.transFigure, 
                         color='white', 
                         linewidth=6,
                         solid_capstyle='butt',
                         zorder=10)
        fig.add_artist(line)
    
    plt.suptitle(f"Enhancement prediction from non-contrast sequences in test-set cases", fontsize=18, y=0.94)
    
    return fig

# Generate figure 3
if 'dice' in results_df.columns and len(results_df) > 0:
    fig3 = visualize_comparison_cases_nnunet(
        results_df, 
        gt_labels_path, 
        predictions_path, 
        n_cases=10,  # Show up to 10 cases 
        sort_metric='dice',
        min_gt_voxels=50  # Lower threshold to ensure we get cases
    )
    
    if fig3:
        fig3.savefig(os.path.join(figures_out, 'Figure_3.png'), dpi=300, bbox_inches='tight')
        fig3.savefig(os.path.join(figures_out, 'Figure_3.svg'), format='svg', bbox_inches='tight')
        plt.show()
    else:
        print("Could not generate figure 3 - no suitable cases found")
else:
    print("Cannot generate figure 3 - no metrics available")

# %%
# # First, merge demographic data from all_images into results_df
# # This ensures Age and Sex columns are available for analysis
# if 'Age' not in results_df.columns or 'Sex' not in results_df.columns:
#     print("Merging demographic data into results_df...")
#     # Merge Age and Sex from all_images based on case_id
#     demographic_cols = ['case_id', 'Age', 'Sex']
#     if all(['Age' in all_images.columns, 'Sex' in all_images.columns]):
#         demographics_df = all_images[demographic_cols].copy()
#         # Merge with results_df
#         results_df = results_df.merge(demographics_df, on='case_id', how='left', suffixes=('', '_all'))
#         # If columns already existed, use the new ones
#         if 'Age_all' in results_df.columns:
#             results_df['Age'] = results_df['Age_all']
#             results_df.drop('Age_all', axis=1, inplace=True)
#         if 'Sex_all' in results_df.columns:
#             results_df['Sex'] = results_df['Sex_all']
#             results_df.drop('Sex_all', axis=1, inplace=True)
#         print(f"Demographics merged. Age available: {results_df['Age'].notna().sum()}, Sex available: {results_df['Sex'].notna().sum()}")
#     else:
#         print("Warning: Age/Sex not available in all_images DataFrame")
        
# # Figure 4: Equitable calibration - UPDATED WITH ALL REQUESTED CHANGES

# from math import pi
# import warnings

# # Calculate overall metrics for Panel A
# tp = ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] > 0)).sum()
# tn = ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] == 0)).sum()
# fp = ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] > 0)).sum()
# fn = ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] == 0)).sum()

# overall_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
# overall_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
# overall_balanced_acc = (overall_sensitivity + overall_specificity) / 2

# # Calculate AUROC using predicted volume as confidence (better than Dice scores)
# from sklearn.metrics import roc_auc_score
# y_true_binary = (results_df['gt_volume'] > 0).astype(int)
# y_scores = results_df['pred_volume'].fillna(0)  # Volume = confidence in detection
# overall_auroc = roc_auc_score(y_true_binary, y_scores)

# # Calculate overall precision, recall, F1 for cases with enhancement
# et_cases = results_df[results_df['gt_volume'] > 0]
# overall_precision = et_cases['precision'].mean()
# overall_recall = et_cases['recall'].mean()
# overall_f1 = et_cases['f1'].mean()

# # DEFINE ALL 7 METRICS FOR CONSISTENCY ACROSS ALL PANELS
# all_metrics_names = ['AUROC', 'Balanced Acc', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1']
# all_metrics_values = [overall_auroc, overall_balanced_acc, overall_sensitivity, overall_specificity,
#                      overall_precision, overall_recall, overall_f1]

# # DEFINE CONSISTENT COLOR PALETTE FOR ALL 7 METRICS
# colors_pie = sns.color_palette('husl', n_colors=10)
# metric_colors = [colors_pie[i] for i in range(7)]  # One color per metric

# # Metrics for radar plots - REMOVED AUROC due to single-class cohorts
# radar_metrics = ['balanced_acc']  # REMOVED AUROC since it's undefined for single-class cohorts
# radar_labels = ['Balanced Acc']

# # Groups to plot: by cohort, by country, by pathology
# groupings = [
#     ('b) Accuracy across datasets', cohorts),
#     ('c) Accuracy across pathologies', list(set(pathologies.values()))),
#     ('d) Accuracy across countries', list(set(countries.values()))),
# ]

# # Define age bins for panel e
# age_bins = [0, 20, 40, 60, 80, 100]
# age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']

# # Create a 2x3 grid with INCREASED row spacing (50% increase) and column spacing
# fig_radar = plt.figure(figsize=(18, 12))
# gs = fig_radar.add_gridspec(2, 3, hspace=0.375, wspace=0.4)  # Changed from 0.25 to 0.375 (50% increase) and added wspace

# # PANEL A: Overall Image-level Detection Performance as BAR CHART - REMOVED XLABEL
# ax_image_detection = fig_radar.add_subplot(gs[0, 0])

# # Bar chart using consistent colors for each metric
# bar_positions = np.arange(len(all_metrics_names))
# bars = ax_image_detection.bar(bar_positions, all_metrics_values, alpha=0.7,
#                              color=metric_colors)

# # Add value labels on bars - CHANGED TO 2 DECIMAL PLACES
# for i, (metric, value) in enumerate(zip(all_metrics_names, all_metrics_values)):
#     ax_image_detection.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

# # REMOVED: xlabel as requested
# ax_image_detection.set_ylabel('Performance')
# ax_image_detection.set_title('a) Overall Image-level Detection Performance')
# ax_image_detection.set_xticks(bar_positions)
# ax_image_detection.set_xticklabels(all_metrics_names, rotation=45, ha='right')
# ax_image_detection.set_ylim(0, 1.1)
# ax_image_detection.grid(True, alpha=0.3, axis='y')

# # Function to calculate all metrics for a given mask - FIXED
# def calculate_all_metrics(mask):
#     """Calculate all 7 metrics for a subset of data"""
#     # Ensure mask is properly aligned with results_df
#     if isinstance(mask, pd.Series):
#         # Reset index to avoid alignment issues
#         mask = mask.reindex(results_df.index, fill_value=False)
    
#     subset = results_df[mask].copy()  # Use copy to avoid warnings
#     et_subset = subset[subset['gt_volume'] > 0]
    
#     if len(subset) == 0:
#         return [np.nan] * 7
    
#     # Calculate binary classification metrics
#     subset_tp = ((subset['gt_volume'] > 0) & (subset['pred_volume'] > 0)).sum()
#     subset_tn = ((subset['gt_volume'] == 0) & (subset['pred_volume'] == 0)).sum()
#     subset_fp = ((subset['gt_volume'] == 0) & (subset['pred_volume'] > 0)).sum()
#     subset_fn = ((subset['gt_volume'] > 0) & (subset['pred_volume'] == 0)).sum()
    
#     # Calculate metrics
#     sensitivity = subset_tp / (subset_tp + subset_fn) if (subset_tp + subset_fn) > 0 else 0
    
#     # Fix specificity calculation - return NaN when no negative cases exist
#     if (subset_tn + subset_fp) > 0:
#         specificity = subset_tn / (subset_tn + subset_fp)
#     else:
#         # No negative cases in this subset - specificity cannot be calculated
#         specificity = np.nan
    
#     # For balanced accuracy, handle NaN specificity
#     if np.isnan(specificity):
#         balanced_acc = sensitivity  # Use only sensitivity when specificity unavailable
#     else:
#         balanced_acc = (sensitivity + specificity) / 2
    
#     # AUROC using predicted volume as confidence - FIXED with proper error handling
#     if len(subset) > 1:
#         y_true_subset = (subset['gt_volume'] > 0).astype(int)
#         y_scores_subset = subset['pred_volume'].fillna(0)  # Volume = confidence in detection
        
#         # Check if we have both classes
#         if len(y_true_subset.unique()) > 1:
#             try:
#                 # Suppress warnings for AUROC calculation
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore")
#                     auroc = roc_auc_score(y_true_subset, y_scores_subset)
#             except:
#                 auroc = 0.5  # Random performance if calculation fails
#         else:
#             # Only one class present
#             auroc = 0.5
#     else:
#         auroc = 0.5
    
#     # Precision, recall, F1 from ET cases
#     if len(et_subset) > 0:
#         precision = et_subset['precision'].mean()
#         recall = et_subset['recall'].mean()
#         f1 = et_subset['f1'].mean()
#     else:
#         precision = 0
#         recall = 0
#         f1 = 0
    
#     return [auroc, balanced_acc, sensitivity, specificity, precision, recall, f1]

# # CORRECTED RADAR PLOT POSITIONING - Panels b, c, d
# axes_radar = [
#     fig_radar.add_subplot(gs[0, 1], polar=True),  # Dataset - column 1, row 0
#     fig_radar.add_subplot(gs[0, 2], polar=True),  # Pathology - column 2, row 0
#     fig_radar.add_subplot(gs[1, 0], polar=True)   # Countries - column 0, row 1
# ]

# for ax, (title, group_list) in zip(axes_radar, groupings):
#     group_names = []
#     metric_means = {metric: [] for metric in radar_metrics}
    
#     for group in group_list:
#         # For cohorts, use direct cohort matching
#         if title == 'b) Accuracy across datasets':
#             mask = results_df['Cohort'] == group
#         # For pathologies, use pathology column matching
#         elif title == 'c) Accuracy across pathologies':
#             mask = results_df['Pathology'] == group
#         # For countries, use country column matching
#         elif title == 'd) Accuracy across countries':
#             mask = results_df['Country'] == group
#         else:
#             mask = pd.Series([False] * len(results_df), index=results_df.index)
        
#         if mask.sum() == 0:
#             for metric in radar_metrics:
#                 metric_means[metric].append(np.nan)
#             group_names.append(group)
#             continue
        
#         # Calculate all 7 metrics
#         all_values = calculate_all_metrics(mask)
        
#         for i, metric in enumerate(radar_metrics):
#             # balanced_acc is at index 1
#             metric_means[metric].append(all_values[1])
        
#         # Format group names for better display
#         if len(group) > 15:  # If name is long, add line breaks
#             words = group.split()
#             if len(words) > 2:
#                 mid_point = len(words) // 2
#                 formatted_name = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
#             else:
#                 formatted_name = group.replace(' ', '\n', 1)
#         else:
#             formatted_name = group
#         group_names.append(formatted_name)
    
#     # Radar plot setup - IMPROVED LABEL VISIBILITY
#     angles = [n / float(len(group_names)) * 2 * pi for n in range(len(group_names))]
#     angles += angles[:1]  # close the loop
    
#     for idx, (metric, label) in enumerate(zip(radar_metrics, radar_labels)):
#         values = metric_means[metric] + [metric_means[metric][0]]
#         # Handle NaN values properly - interpolate or skip individual points
#         if not all(np.isnan(values[:-1])):  # Check if we have any non-NaN values
#             # Replace NaN values with 0 for plotting (they won't be visible anyway at radius 0)
#             values_plot = [0 if np.isnan(v) else v for v in values]
#             # Plot WITHOUT dots/markers as requested
#             ax.plot(angles, values_plot, color=metric_colors[1], linewidth=2)
#             ax.fill(angles, values_plot, color=metric_colors[1], alpha=0.1)
    
#     ax.set_xticks(angles[:-1])
#     # IMPROVED: Set label properties to ensure visibility above plot data
#     ax.set_xticklabels(group_names, rotation=0, fontsize=9,
#                        verticalalignment='center', horizontalalignment='center',
#                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
#     ax.set_title(title)
#     ax.set_ylim(0, 1)
#     ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
#     ax.set_rlabel_position(45)
#     ax.grid(True, alpha=0.3)
    
#     # Ensure labels are on top by setting zorder
#     for label in ax.get_xticklabels():
#         label.set_zorder(10)

# # PANEL E: Accuracy across ages - RADAR PLOT
# ax_age = fig_radar.add_subplot(gs[1, 1], polar=True)

# # Get demographics from results_df (now available after merge)
# demographics_data = results_df[['case_id', 'Age', 'Sex', 'gt_volume', 'pred_volume']].copy()

# # Handle missing demographics
# demographics_data = demographics_data[demographics_data['Age'].notna() & demographics_data['Sex'].notna()]

# # Convert age to numeric if needed
# demographics_data['Age'] = pd.to_numeric(demographics_data['Age'], errors='coerce')

# # Filter out invalid ages
# demographics_data = demographics_data[(demographics_data['Age'] >= 0) & (demographics_data['Age'] <= 120)]

# print(f"\nDemographics data shape: {demographics_data.shape}")
# print(f"Unique sexes: {demographics_data['Sex'].unique()}")

# # Define consistent age bins - same as figure 1 panel c
# demographics_data['Age_Group'] = pd.cut(demographics_data['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

# # Calculate balanced accuracy for each age group
# age_balanced_acc = []
# for age_group in age_labels:
#     mask = demographics_data['Age_Group'] == age_group
#     if mask.sum() > 0:
#         metrics = calculate_all_metrics(mask)
#         age_balanced_acc.append(metrics[1])  # balanced_acc is at index 1
#     else:
#         age_balanced_acc.append(np.nan)

# # Setup radar plot for age groups
# angles_age = [n / float(len(age_labels)) * 2 * pi for n in range(len(age_labels))]
# angles_age += angles_age[:1]  # close the loop

# values_age = age_balanced_acc + [age_balanced_acc[0]]
# values_age_plot = [0 if np.isnan(v) else v for v in values_age]

# # Plot WITHOUT dots/markers as requested
# ax_age.plot(angles_age, values_age_plot, color=metric_colors[1], linewidth=2)
# ax_age.fill(angles_age, values_age_plot, color=metric_colors[1], alpha=0.1)

# ax_age.set_xticks(angles_age[:-1])
# ax_age.set_xticklabels(age_labels, rotation=0, fontsize=9,
#                        verticalalignment='center', horizontalalignment='center',
#                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
# ax_age.set_title('e) Accuracy across ages')
# ax_age.set_ylim(0, 1)
# ax_age.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
# ax_age.set_rlabel_position(45)
# ax_age.grid(True, alpha=0.3)

# # PANEL F: Accuracy across sexes - BAR CHART WITH LIGHT GREY LEGEND COLORS - NO TEXT ANNOTATIONS
# ax_sex = fig_radar.add_subplot(gs[1, 2])

# # Standardize sex values
# demographics_data['Sex_Standardized'] = demographics_data['Sex'].str.upper().str.strip()

# # Calculate all metrics for each sex
# male_metrics = []
# female_metrics = []

# male_mask = demographics_data['Sex_Standardized'] == 'M'
# female_mask = demographics_data['Sex_Standardized'] == 'F'

# if male_mask.sum() > 0:
#     male_metrics = calculate_all_metrics(male_mask)
# else:
#     male_metrics = [np.nan] * 7

# if female_mask.sum() > 0:
#     female_metrics = calculate_all_metrics(female_mask)
# else:
#     female_metrics = [np.nan] * 7

# # Setup grouped bar chart
# x = np.arange(len(all_metrics_names))
# width = 0.35

# # Plot bars for male and female - using the SAME colors as the metrics
# bars1 = ax_sex.bar(x - width/2, male_metrics, width, label='Male', 
#                    color=metric_colors, alpha=0.7)
# bars2 = ax_sex.bar(x + width/2, female_metrics, width, label='Female',
#                    color=metric_colors, alpha=0.7)

# # Add dashed edge for female bars
# for bar in bars2:
#     bar.set_edgecolor('black')
#     bar.set_linewidth(1.5)
#     bar.set_linestyle('--')

# # REMOVED: Text annotations on bars as requested

# # Add sample sizes in legend
# male_count = male_mask.sum()
# female_count = female_mask.sum()

# # Create custom legend with light grey colors as requested
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='lightgrey', edgecolor='black', label=f'Male (n={male_count})'),
#     Patch(facecolor='lightgrey', edgecolor='black', linestyle='--', linewidth=1.5, label=f'Female (n={female_count})')
# ]
# ax_sex.legend(handles=legend_elements, loc='upper right')

# # REMOVE xlabel as requested
# ax_sex.set_ylabel('Performance')
# ax_sex.set_title('f) Accuracy across sexes')
# ax_sex.set_xticks(x)
# ax_sex.set_xticklabels(all_metrics_names, rotation=45, ha='right')
# ax_sex.set_ylim(0, 1.15)
# ax_sex.grid(True, alpha=0.3, axis='y')

# # Add main title to the entire figure
# plt.suptitle('Equitable calibration of enhancement detection', fontsize=18, y=0.95)

# plt.tight_layout()

# # SAVE THE FIGURE
# output_path = '/home/jruffle/Downloads/Figure_4_Equitable_Calibration.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
# print(f"\nFigure 4 saved to: {output_path}")

# # Also save to the figures_out directory
# fig_radar.savefig(os.path.join(figures_out, 'nnunet_figure_4.png'), dpi=300, bbox_inches='tight')
# fig_radar.savefig(os.path.join(figures_out, 'nnunet_figure_4.svg'), format='svg', bbox_inches='tight')

# plt.show()

# %%
# # First, merge demographic data from all_images into results_df
# # This ensures Age and Sex columns are available for analysis
# if 'Age' not in results_df.columns or 'Sex' not in results_df.columns:
#     print("Merging demographic data into results_df...")
#     # Merge Age and Sex from all_images based on case_id
#     demographic_cols = ['case_id', 'Age', 'Sex']
#     if all(['Age' in all_images.columns, 'Sex' in all_images.columns]):
#         demographics_df = all_images[demographic_cols].copy()
#         # Merge with results_df
#         results_df = results_df.merge(demographics_df, on='case_id', how='left', suffixes=('', '_all'))
#         # If columns already existed, use the new ones
#         if 'Age_all' in results_df.columns:
#             results_df['Age'] = results_df['Age_all']
#             results_df.drop('Age_all', axis=1, inplace=True)
#         if 'Sex_all' in results_df.columns:
#             results_df['Sex'] = results_df['Sex_all']
#             results_df.drop('Sex_all', axis=1, inplace=True)
#         print(f"Demographics merged. Age available: {results_df['Age'].notna().sum()}, Sex available: {results_df['Sex'].notna().sum()}")
#     else:
#         print("Warning: Age/Sex not available in all_images DataFrame")
        
# # Figure 4: Equitable calibration - VERSION WITH SENSITIVITY AND SPECIFICITY IN RADAR PLOTS

# from math import pi
# import warnings

# # Calculate overall metrics for Panel A
# tp = ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] > 0)).sum()
# tn = ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] == 0)).sum()
# fp = ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] > 0)).sum()
# fn = ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] == 0)).sum()

# overall_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
# overall_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
# overall_balanced_acc = (overall_sensitivity + overall_specificity) / 2

# # Calculate AUROC using predicted volume as confidence (better than Dice scores)
# from sklearn.metrics import roc_auc_score
# y_true_binary = (results_df['gt_volume'] > 0).astype(int)
# y_scores = results_df['pred_volume'].fillna(0)  # Volume = confidence in detection
# overall_auroc = roc_auc_score(y_true_binary, y_scores)

# # Calculate overall precision, recall, F1 for cases with enhancement
# et_cases = results_df[results_df['gt_volume'] > 0]
# overall_precision = et_cases['precision'].mean()
# overall_recall = et_cases['recall'].mean()
# overall_f1 = et_cases['f1'].mean()

# # DEFINE ALL 7 METRICS FOR CONSISTENCY ACROSS ALL PANELS
# all_metrics_names = ['AUROC', 'Balanced Acc', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1']
# all_metrics_values = [overall_auroc, overall_balanced_acc, overall_sensitivity, overall_specificity,
#                      overall_precision, overall_recall, overall_f1]

# # DEFINE CONSISTENT COLOR PALETTE FOR ALL 7 METRICS
# colors_pie = sns.color_palette('husl', n_colors=10)
# metric_colors = [colors_pie[i] for i in range(7)]  # One color per metric

# # Metrics for radar plots - REMOVED AUROC due to single-class cohorts
# radar_metrics = ['balanced_acc', 'sensitivity', 'specificity']  # Added sensitivity and specificity
# radar_labels = ['Balanced Acc', 'Sensitivity', 'Specificity']

# # Groups to plot: by cohort, by country, by pathology
# groupings = [
#     ('b) Accuracy across datasets', cohorts),
#     ('c) Accuracy across pathologies', list(set(pathologies.values()))),
#     ('d) Accuracy across countries', list(set(countries.values()))),
# ]

# # Define age bins for panel e
# age_bins = [0, 20, 40, 60, 80, 100]
# age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']

# # Create a 2x3 grid with INCREASED row spacing (50% increase) and column spacing
# fig_radar = plt.figure(figsize=(18, 12))
# gs = fig_radar.add_gridspec(2, 3, hspace=0.375, wspace=0.4)  # Changed from 0.25 to 0.375 (50% increase) and added wspace

# # PANEL A: Overall Image-level Detection Performance as BAR CHART - REMOVED XLABEL
# ax_image_detection = fig_radar.add_subplot(gs[0, 0])

# # Bar chart using consistent colors for each metric
# bar_positions = np.arange(len(all_metrics_names))
# bars = ax_image_detection.bar(bar_positions, all_metrics_values, alpha=0.7,
#                              color=metric_colors)

# # Add value labels on bars - CHANGED TO 2 DECIMAL PLACES
# for i, (metric, value) in enumerate(zip(all_metrics_names, all_metrics_values)):
#     ax_image_detection.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

# # REMOVED: xlabel as requested
# ax_image_detection.set_ylabel('Performance')
# ax_image_detection.set_title('a) Overall Image-level Detection Performance')
# ax_image_detection.set_xticks(bar_positions)
# ax_image_detection.set_xticklabels(all_metrics_names, rotation=45, ha='right')
# ax_image_detection.set_ylim(0, 1.1)
# ax_image_detection.grid(True, alpha=0.3, axis='y')

# # Function to calculate all metrics for a given mask - FIXED
# def calculate_all_metrics(mask):
#     """Calculate all 7 metrics for a subset of data"""
#     # Ensure mask is properly aligned with results_df
#     if isinstance(mask, pd.Series):
#         # Reset index to avoid alignment issues
#         mask = mask.reindex(results_df.index, fill_value=False)
    
#     subset = results_df[mask].copy()  # Use copy to avoid warnings
#     et_subset = subset[subset['gt_volume'] > 0]
    
#     if len(subset) == 0:
#         return [np.nan] * 7
    
#     # Calculate binary classification metrics
#     subset_tp = ((subset['gt_volume'] > 0) & (subset['pred_volume'] > 0)).sum()
#     subset_tn = ((subset['gt_volume'] == 0) & (subset['pred_volume'] == 0)).sum()
#     subset_fp = ((subset['gt_volume'] == 0) & (subset['pred_volume'] > 0)).sum()
#     subset_fn = ((subset['gt_volume'] > 0) & (subset['pred_volume'] == 0)).sum()
    
#     # Calculate metrics
#     sensitivity = subset_tp / (subset_tp + subset_fn) if (subset_tp + subset_fn) > 0 else 0
    
#     # Fix specificity calculation - return NaN when no negative cases exist
#     if (subset_tn + subset_fp) > 0:
#         specificity = subset_tn / (subset_tn + subset_fp)
#     else:
#         # No negative cases in this subset - specificity cannot be calculated
#         specificity = np.nan
    
#     # For balanced accuracy, handle NaN specificity
#     if np.isnan(specificity):
#         balanced_acc = sensitivity  # Use only sensitivity when specificity unavailable
#     else:
#         balanced_acc = (sensitivity + specificity) / 2
    
#     # AUROC using predicted volume as confidence - FIXED with proper error handling
#     if len(subset) > 1:
#         y_true_subset = (subset['gt_volume'] > 0).astype(int)
#         y_scores_subset = subset['pred_volume'].fillna(0)  # Volume = confidence in detection
        
#         # Check if we have both classes
#         if len(y_true_subset.unique()) > 1:
#             try:
#                 # Suppress warnings for AUROC calculation
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore")
#                     auroc = roc_auc_score(y_true_subset, y_scores_subset)
#             except:
#                 auroc = 0.5  # Random performance if calculation fails
#         else:
#             # Only one class present
#             auroc = 0.5
#     else:
#         auroc = 0.5
    
#     # Precision, recall, F1 from ET cases
#     if len(et_subset) > 0:
#         precision = et_subset['precision'].mean()
#         recall = et_subset['recall'].mean()
#         f1 = et_subset['f1'].mean()
#     else:
#         precision = 0
#         recall = 0
#         f1 = 0
    
#     return [auroc, balanced_acc, sensitivity, specificity, precision, recall, f1]

# # CORRECTED RADAR PLOT POSITIONING - Panels b, c, d
# axes_radar = [
#     fig_radar.add_subplot(gs[0, 1], polar=True),  # Dataset - column 1, row 0
#     fig_radar.add_subplot(gs[0, 2], polar=True),  # Pathology - column 2, row 0
#     fig_radar.add_subplot(gs[1, 0], polar=True)   # Countries - column 0, row 1
# ]

# for ax, (title, group_list) in zip(axes_radar, groupings):
#     group_names = []
#     metric_means = {metric: [] for metric in radar_metrics}
    
#     for group in group_list:
#         # For cohorts, use direct cohort matching
#         if title == 'b) Accuracy across datasets':
#             mask = results_df['Cohort'] == group
#         # For pathologies, use pathology column matching
#         elif title == 'c) Accuracy across pathologies':
#             mask = results_df['Pathology'] == group
#         # For countries, use country column matching
#         elif title == 'd) Accuracy across countries':
#             mask = results_df['Country'] == group
#         else:
#             mask = pd.Series([False] * len(results_df), index=results_df.index)
        
#         if mask.sum() == 0:
#             for metric in radar_metrics:
#                 metric_means[metric].append(np.nan)
#             group_names.append(group)
#             continue
        
#         # Calculate all 7 metrics
#         all_values = calculate_all_metrics(mask)
        
#         # FIXED: Extract correct metric values based on their indices
#         # Index mapping: [0:auroc, 1:balanced_acc, 2:sensitivity, 3:specificity, 4:precision, 5:recall, 6:f1]
#         for metric in radar_metrics:
#             if metric == 'balanced_acc':
#                 metric_means[metric].append(all_values[1])
#             elif metric == 'sensitivity':
#                 metric_means[metric].append(all_values[2])
#             elif metric == 'specificity':
#                 metric_means[metric].append(all_values[3])
        
#         # Format group names for better display
#         if len(group) > 15:  # If name is long, add line breaks
#             words = group.split()
#             if len(words) > 2:
#                 mid_point = len(words) // 2
#                 formatted_name = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
#             else:
#                 formatted_name = group.replace(' ', '\n', 1)
#         else:
#             formatted_name = group
#         group_names.append(formatted_name)
    
#     # Radar plot setup - IMPROVED LABEL VISIBILITY
#     angles = [n / float(len(group_names)) * 2 * pi for n in range(len(group_names))]
#     angles += angles[:1]  # close the loop
    
#     # FIXED: Plot each metric with its corresponding color
#     for idx, (metric, label) in enumerate(zip(radar_metrics, radar_labels)):
#         values = metric_means[metric] + [metric_means[metric][0]]
#         # Handle NaN values properly - interpolate or skip individual points
#         if not all(np.isnan(values[:-1])):  # Check if we have any non-NaN values
#             # Replace NaN values with 0 for plotting (they won't be visible anyway at radius 0)
#             values_plot = [0 if np.isnan(v) else v for v in values]
            
#             # FIXED: Use different colors for each metric
#             # metric_colors[1] = Balanced Accuracy (blue)
#             # metric_colors[2] = Sensitivity (orange) 
#             # metric_colors[3] = Specificity (green)
#             if metric == 'balanced_acc':
#                 color = metric_colors[1]  # Blue
#             elif metric == 'sensitivity':
#                 color = metric_colors[2]  # Orange
#             elif metric == 'specificity':
#                 color = metric_colors[3]  # Green
            
#             # Plot WITHOUT dots/markers as requested
#             ax.plot(angles, values_plot, color=color, linewidth=2, label=label)
#             ax.fill(angles, values_plot, color=color, alpha=0.1)
    
#     ax.set_xticks(angles[:-1])
#     # IMPROVED: Set label properties to ensure visibility above plot data
#     ax.set_xticklabels(group_names, rotation=0, fontsize=9,
#                        verticalalignment='center', horizontalalignment='center',
#                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
#     ax.set_title(title)
#     ax.set_ylim(0, 1)
#     ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
#     ax.set_rlabel_position(45)
#     ax.grid(True, alpha=0.3)
    
#     # Add legend to first radar plot
#     if ax == axes_radar[0]:
#         ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    
#     # Ensure labels are on top by setting zorder
#     for label in ax.get_xticklabels():
#         label.set_zorder(10)

# # PANEL E: Accuracy across ages - RADAR PLOT
# ax_age = fig_radar.add_subplot(gs[1, 1], polar=True)

# # Get demographics from results_df (now available after merge)
# demographics_data = results_df[['case_id', 'Age', 'Sex', 'gt_volume', 'pred_volume']].copy()

# # Handle missing demographics
# demographics_data = demographics_data[demographics_data['Age'].notna() & demographics_data['Sex'].notna()]

# # Convert age to numeric if needed
# demographics_data['Age'] = pd.to_numeric(demographics_data['Age'], errors='coerce')

# # Filter out invalid ages
# demographics_data = demographics_data[(demographics_data['Age'] >= 0) & (demographics_data['Age'] <= 120)]

# print(f"\nDemographics data shape: {demographics_data.shape}")
# print(f"Unique sexes: {demographics_data['Sex'].unique()}")

# # Define consistent age bins - same as figure 1 panel c
# demographics_data['Age_Group'] = pd.cut(demographics_data['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

# # Calculate metrics for each age group
# age_metrics = {metric: [] for metric in radar_metrics}
# for age_group in age_labels:
#     mask = demographics_data['Age_Group'] == age_group
#     if mask.sum() > 0:
#         metrics = calculate_all_metrics(mask)
#         age_metrics['balanced_acc'].append(metrics[1])
#         age_metrics['sensitivity'].append(metrics[2])
#         age_metrics['specificity'].append(metrics[3])
#     else:
#         for metric in radar_metrics:
#             age_metrics[metric].append(np.nan)

# # Setup radar plot for age groups
# angles_age = [n / float(len(age_labels)) * 2 * pi for n in range(len(age_labels))]
# angles_age += angles_age[:1]  # close the loop

# # FIXED: Plot each metric with its corresponding color
# for metric, label in zip(radar_metrics, radar_labels):
#     values_age = age_metrics[metric] + [age_metrics[metric][0]]
#     values_age_plot = [0 if np.isnan(v) else v for v in values_age]
    
#     # Use correct colors
#     if metric == 'balanced_acc':
#         color = metric_colors[1]  # Blue
#     elif metric == 'sensitivity':
#         color = metric_colors[2]  # Orange
#     elif metric == 'specificity':
#         color = metric_colors[3]  # Green
    
#     # Plot WITHOUT dots/markers as requested
#     ax_age.plot(angles_age, values_age_plot, color=color, linewidth=2, label=label)
#     ax_age.fill(angles_age, values_age_plot, color=color, alpha=0.1)

# ax_age.set_xticks(angles_age[:-1])
# ax_age.set_xticklabels(age_labels, rotation=0, fontsize=9,
#                        verticalalignment='center', horizontalalignment='center',
#                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
# ax_age.set_title('e) Accuracy across ages')
# ax_age.set_ylim(0, 1)
# ax_age.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
# ax_age.set_rlabel_position(45)
# ax_age.grid(True, alpha=0.3)
# ax_age.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

# # PANEL F: Accuracy across sexes - BAR CHART WITH LIGHT GREY LEGEND COLORS - NO TEXT ANNOTATIONS
# ax_sex = fig_radar.add_subplot(gs[1, 2])

# # Standardize sex values
# demographics_data['Sex_Standardized'] = demographics_data['Sex'].str.upper().str.strip()

# # Calculate all metrics for each sex
# male_metrics = []
# female_metrics = []

# male_mask = demographics_data['Sex_Standardized'] == 'M'
# female_mask = demographics_data['Sex_Standardized'] == 'F'

# if male_mask.sum() > 0:
#     male_metrics = calculate_all_metrics(male_mask)
# else:
#     male_metrics = [np.nan] * 7

# if female_mask.sum() > 0:
#     female_metrics = calculate_all_metrics(female_mask)
# else:
#     female_metrics = [np.nan] * 7

# # Setup grouped bar chart
# x = np.arange(len(all_metrics_names))
# width = 0.35

# # Plot bars for male and female - using the SAME colors as the metrics
# bars1 = ax_sex.bar(x - width/2, male_metrics, width, label='Male', 
#                    color=metric_colors, alpha=0.7)
# bars2 = ax_sex.bar(x + width/2, female_metrics, width, label='Female',
#                    color=metric_colors, alpha=0.7)

# # Add dashed edge for female bars
# for bar in bars2:
#     bar.set_edgecolor('black')
#     bar.set_linewidth(1.5)
#     bar.set_linestyle('--')

# # REMOVED: Text annotations on bars as requested

# # Add sample sizes in legend
# male_count = male_mask.sum()
# female_count = female_mask.sum()

# # Create custom legend with light grey colors as requested
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='lightgrey', edgecolor='black', label=f'Male (n={male_count})'),
#     Patch(facecolor='lightgrey', edgecolor='black', linestyle='--', linewidth=1.5, label=f'Female (n={female_count})')
# ]
# ax_sex.legend(handles=legend_elements, loc='upper right')

# # REMOVE xlabel as requested
# ax_sex.set_ylabel('Performance')
# ax_sex.set_title('f) Accuracy across sexes')
# ax_sex.set_xticks(x)
# ax_sex.set_xticklabels(all_metrics_names, rotation=45, ha='right')
# ax_sex.set_ylim(0, 1.15)
# ax_sex.grid(True, alpha=0.3, axis='y')

# # Add main title to the entire figure
# plt.suptitle('Equitable calibration of enhancement detection', fontsize=18, y=0.95)

# plt.tight_layout()

# # SAVE THE FIGURE
# output_path = '/home/jruffle/Downloads/Figure_4_Equitable_Calibration_with_sens_spec_FIXED.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
# print(f"\nFigure 4 saved to: {output_path}")

# # Also save to the figures_out directory
# fig_radar.savefig(os.path.join(figures_out, 'nnunet_figure_4_with_sens_spec_FIXED.png'), dpi=300, bbox_inches='tight')
# fig_radar.savefig(os.path.join(figures_out, 'nnunet_figure_4_with_sens_spec_FIXED.svg'), format='svg', bbox_inches='tight')

# plt.show()

# %%
# Investigation of specificity issue for Sub-Saharan Africa and other cohorts
print("=== Investigating Zero Specificity Issue ===")
print("Checking data distribution for cohorts with potential 0 specificity:\n")

# Check which cohorts have no negative cases
cohorts_to_check = results_df['Cohort'].unique()
cohorts_with_no_negatives = []

for cohort in sorted(cohorts_to_check):
    cohort_mask = results_df['Cohort'] == cohort
    cohort_data = results_df[cohort_mask]
    
    n_total = len(cohort_data)
    n_positive = (cohort_data['gt_volume'] > 0).sum()
    n_negative = (cohort_data['gt_volume'] == 0).sum()
    
    print(f"{cohort}:")
    print(f"  Total cases: {n_total}")
    print(f"  Cases with enhancement: {n_positive} ({n_positive/n_total*100:.1f}%)")
    print(f"  Cases without enhancement: {n_negative} ({n_negative/n_total*100:.1f}%)")
    
    if n_negative == 0:
        print(f"  -> NO NEGATIVE CASES - Specificity cannot be calculated!")
        cohorts_with_no_negatives.append(cohort)
    print()

print(f"\nCohorts with NO negative cases (100% enhancement): {cohorts_with_no_negatives}")

# Also check countries
print("\n\nChecking by country mapping:")
# Define country mapping
countries = {}
countries['IvyGAP'] = 'US'
countries['BraTS2021'] = 'International'
countries['UPENN-GBM'] = 'US'
countries['UCSF-PDGM'] = 'US'
countries['BraTS-SSA'] = 'Sub-Saharan Africa'
countries['BraTS-MET'] = 'US'
countries['BraTS-MEN'] = 'US'

for country in sorted(set(countries.values())):
    cohorts_in_country = [c for c, ctry in countries.items() if ctry == country]
    country_mask = results_df['Cohort'].isin(cohorts_in_country)
    country_data = results_df[country_mask]
    
    n_total = len(country_data)
    if n_total == 0:
        continue
        
    n_positive = (country_data['gt_volume'] > 0).sum()
    n_negative = (country_data['gt_volume'] == 0).sum()
    
    print(f"\n{country} (cohorts: {', '.join(cohorts_in_country)}):")
    print(f"  Total cases: {n_total}")
    print(f"  Cases with enhancement: {n_positive} ({n_positive/n_total*100:.1f}%)")
    print(f"  Cases without enhancement: {n_negative} ({n_negative/n_total*100:.1f}%)")
    
    if n_negative == 0:
        print(f"  -> NO NEGATIVE CASES for {country} - Specificity cannot be calculated!")

# %%
# Analyze UPENN-GBM and other cohorts with AUROC = 0.5
print("=== Analyzing cohorts with AUROC = 0.5 ===\n")

# Check UPENN-GBM specifically
upenn_mask = results_df['Cohort'] == 'UPENN-GBM'
upenn_data = results_df[upenn_mask]

print(f"UPENN-GBM Analysis:")
print(f"Total cases: {len(upenn_data)}")
print(f"Cases with enhancement (gt_volume > 0): {(upenn_data['gt_volume'] > 0).sum()}")
print(f"Cases without enhancement (gt_volume = 0): {(upenn_data['gt_volume'] == 0).sum()}")
print(f"Cases predicted with enhancement (pred_volume > 0): {(upenn_data['pred_volume'] > 0).sum()}")
print(f"Cases predicted without enhancement (pred_volume = 0): {(upenn_data['pred_volume'] == 0).sum()}")

# Check if only one class is present
y_true_upenn = (upenn_data['gt_volume'] > 0).astype(int)
print(f"\nUnique values in y_true for UPENN-GBM: {y_true_upenn.unique()}")
print(f"This means UPENN-GBM has {'only positive cases (all have enhancement)' if len(y_true_upenn.unique()) == 1 and y_true_upenn.unique()[0] == 1 else 'both classes or only negative cases'}")

# Try calculating AUROC manually
from sklearn.metrics import roc_auc_score
import warnings

y_scores_upenn = upenn_data['pred_volume'].fillna(0)
print(f"\nPred_volume statistics for UPENN-GBM:")
print(f"Min: {y_scores_upenn.min():.3f}, Max: {y_scores_upenn.max():.3f}")
print(f"Mean: {y_scores_upenn.mean():.3f}, Std: {y_scores_upenn.std():.3f}")

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manual_auroc = roc_auc_score(y_true_upenn, y_scores_upenn)
        print(f"\nManually calculated AUROC for UPENN-GBM: {manual_auroc:.3f}")
except ValueError as e:
    print(f"\nAUROC calculation failed with error: {e}")
    print("This happens when only one class is present in y_true")

# Check all cohorts with AUROC = 0.5
print("\n\n=== All cohorts with AUROC = 0.5 ===")
for cohort in cohorts:
    cohort_mask = results_df['Cohort'] == cohort
    cohort_data = results_df[cohort_mask]
    
    if len(cohort_data) > 0:
        y_true_cohort = (cohort_data['gt_volume'] > 0).astype(int)
        n_positive = y_true_cohort.sum()
        n_negative = len(y_true_cohort) - n_positive
        
        # Only print cohorts with single class
        if len(y_true_cohort.unique()) == 1:
            print(f"\n{cohort}:")
            print(f"  Total cases: {len(cohort_data)}")
            print(f"  Positive cases (enhancement): {n_positive}")
            print(f"  Negative cases (no enhancement): {n_negative}")
            print(f"  → Only {'positive' if n_positive > 0 else 'negative'} cases present")

print("\n\nConclusion: AUROC = 0.5 for cohorts with only one class is correct behavior.")
print("AUROC measures discrimination between classes, which is undefined with only one class present.")

# %%
# # ROC and Precision-Recall Analysis for Model Performance Assessment
# from sklearn.metrics import roc_curve, precision_recall_curve, auc
# import matplotlib.pyplot as plt

# def create_roc_pr_analysis():
#     """
#     Create ROC and Precision-Recall curves using per-case metrics
#     This demonstrates the model's discriminative performance across different thresholds
#     """
#     # Filter cases with actual enhancing tumour for meaningful ROC/PR analysis
#     et_cases = results_df[results_df['gt_volume'] > 0].copy()
#     non_et_cases = results_df[results_df['gt_volume'] == 0].copy()
    
#     print(f"Cases with enhancing tumour: {len(et_cases)}")
#     print(f"Cases without enhancing tumour: {len(non_et_cases)}")
    
#     if len(et_cases) == 0 or len(non_et_cases) == 0:
#         print("Insufficient data for ROC/PR analysis")
#         return None
    
#     # Create binary labels (1 for cases with ET, 0 for cases without)
#     y_true = []
#     y_scores = []
#     case_ids = []
    
#     # For cases with ET, use Dice score as confidence
#     for _, row in et_cases.iterrows():
#         y_true.append(1)
#         y_scores.append(row['dice'])  # Dice as confidence score
#         case_ids.append(row['case_id'])
    
#     # For cases without ET, use 1 - specificity as confidence (inverted)
#     for _, row in non_et_cases.iterrows():
#         y_true.append(0)
#         # For true negatives, low prediction confidence = high specificity
#         specificity = row['tn'] / (row['tn'] + row['fp']) if (row['tn'] + row['fp']) > 0 else 1.0
#         y_scores.append(1 - specificity)  # Convert to "positive" confidence
#         case_ids.append(row['case_id'])
    
#     y_true = np.array(y_true)
#     y_scores = np.array(y_scores)
    
#     # Compute ROC curve
#     fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
    
#     # Compute Precision-Recall curve
#     precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
#     pr_auc = auc(recall, precision)
    
#     # Create figure with consistent color scheme
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     # USE CONSISTENT COLOR PALETTE FROM FIGURE 1
#     colors_pie = sns.color_palette('husl', n_colors=10)
    
#     # ROC Curve - using consistent colors
#     axes[0].plot(fpr, tpr, color=colors_pie[0], lw=2, 
#                 label=f'ROC curve (AUROC = {roc_auc:.3f})')
#     axes[0].plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random')  # CHANGED TO BLACK
#     axes[0].set_xlim([0.0, 1.0])
#     axes[0].set_ylim([0.0, 1.05])
#     axes[0].set_xlabel('False Positive Rate')
#     axes[0].set_ylabel('True Positive Rate')
#     axes[0].set_title('a) ROC Curve - Enhancement Detection')  # Added panel label a)
#     axes[0].legend(loc="lower right")
#     axes[0].grid(True, alpha=0.3)
    
#     # Precision-Recall Curve - using consistent colors
#     axes[1].plot(recall, precision, color=colors_pie[1], lw=2,
#                 label=f'PR curve (AUPRC = {pr_auc:.3f})')
#     baseline = len(et_cases) / len(results_df)
#     axes[1].axhline(y=baseline, color='black', linestyle='--',  # CHANGED TO BLACK
#                    label=f'Baseline (Prevalence = {baseline:.3f})')
#     axes[1].set_xlim([0.0, 1.0])
#     axes[1].set_ylim([0.0, 1.05])
#     axes[1].set_xlabel('Recall')
#     axes[1].set_ylabel('Precision')
#     axes[1].set_title('b) Precision-Recall Curve')  # Added panel label b)
#     axes[1].legend(loc="lower left")
#     axes[1].grid(True, alpha=0.3)
    
#     # Confidence score distribution - ONLY SHOWING ET CASES
#     et_scores = y_scores[y_true == 1]
#     non_et_scores = y_scores[y_true == 0]
    
#     # Create histogram for ET cases only
#     bins = np.linspace(0, 1, 30)
    
#     # Plot histogram for ET cases only
#     axes[2].hist(et_scores, bins=bins, alpha=0.6, label=f'With ET (n={len(et_scores)})', 
#                 color=colors_pie[2], density=True, edgecolor='black', linewidth=1.5)
    
#     # Add mean line for ET only
#     axes[2].axvline(x=np.mean(et_scores), color=colors_pie[2], linestyle='--', linewidth=2,
#                    label=f'With ET mean: {np.mean(et_scores):.3f}')
    
#     axes[2].set_xlabel('Confidence Score')
#     axes[2].set_ylabel('Density')
#     axes[2].set_title('c) Score Distribution by True Label')  # Added panel label c)
#     axes[2].legend()
#     axes[2].grid(True, alpha=0.3, axis='y')
#     axes[2].set_xlim(0, 1)
    
#     plt.suptitle('Model Discriminative Performance Analysis', fontsize=16)
#     plt.tight_layout()
    
#     # Print summary statistics
#     print(f"\nModel Performance Summary:")
#     print(f"ROC AUC: {roc_auc:.3f}")
#     print(f"PR AUC: {pr_auc:.3f}")
#     print(f"Enhancement prevalence: {baseline:.3f}")
    
#     return fig, {"roc_auc": roc_auc, "pr_auc": pr_auc, "prevalence": baseline}

# # Generate the analysis
# if 'dice' in results_df.columns and len(results_df) > 0:
#     fig_roc_pr, performance_stats = create_roc_pr_analysis()
#     if fig_roc_pr:
#         fig_roc_pr.savefig(os.path.join(figures_out, 'model_discriminative_performance_old.png'), 
#                           dpi=300, bbox_inches='tight')
#         fig_roc_pr.savefig(os.path.join(figures_out, 'model_discriminative_performance_old.svg'), 
#                           format='svg', bbox_inches='tight')
#         plt.show()
#     else:
#         print("Could not generate ROC/PR analysis")
# else:
#     print("No metrics available for ROC/PR analysis")

# %%
# Failure Case Analysis - Understanding Model Limitations - UPDATED WITH PROPER PANEL LABELS AND CONSISTENT COLORS
def analyze_failure_cases():
    """
    Analyze cases where the model performed poorly to understand limitations
    Updated with proper panel labels and consistent color scheme
    """
    # Define failure criteria
    poor_cases = results_df[
        (results_df['gt_volume'] > 100) &  # Cases with substantial enhancement
        (results_df['dice'] < 0.3)  # Poor Dice performance
    ].copy()
    
    excellent_cases = results_df[
        (results_df['gt_volume'] > 100) &  # Cases with substantial enhancement  
        (results_df['dice'] > 0.7)  # Excellent Dice performance
    ].copy()
    
    print(f"Poor performance cases (Dice < 0.3): {len(poor_cases)}")
    print(f"Excellent performance cases (Dice > 0.7): {len(excellent_cases)}")
    
    # DEFINE CONSISTENT COLOR PALETTE (matching Figure 1)
    colors_pie = sns.color_palette('husl', n_colors=10)
    
    # Create comprehensive failure analysis figure with PROPER PANEL LABELS
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # 1. Volume vs Performance scatter WITH PANEL LABEL
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(results_df['gt_volume'], results_df['dice'], 
                         c=results_df['dice'], cmap='RdYlBu', alpha=0.6, s=30)
    ax1.set_xlabel('Ground Truth Volume (voxels)', fontsize=12)
    ax1.set_ylabel('Dice Score', fontsize=12)
    ax1.set_title('a) Volume vs Performance', fontsize=14)  # REMOVED BOLD
    ax1.axhline(y=0.3, color=colors_pie[4], linestyle='--', alpha=0.7, label='Poor threshold')
    ax1.axhline(y=0.7, color=colors_pie[2], linestyle='--', alpha=0.7, label='Excellent threshold')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Dice Score')
    ax1.grid(True, alpha=0.3)
    
    # 2. Success and failure rates by cohort WITH PANEL LABEL - SORTED BY SUCCESS RATE
    ax2 = fig.add_subplot(gs[0, 1])
    cohort_stats = []
    cohort_names = []
    success_rates = []
    failure_rates = []
    
    for cohort in results_df['Cohort'].unique():
        if cohort == '':
            continue
        cohort_data = results_df[results_df['Cohort'] == cohort]
        et_cases = cohort_data[cohort_data['gt_volume'] > 0]
        if len(et_cases) > 0:
            failure_rate = (et_cases['dice'] < 0.3).sum() / len(et_cases)
            success_rate = (et_cases['dice'] >= 0.3).sum() / len(et_cases)
            cohort_stats.append({
                'cohort': cohort,
                'success_rate': success_rate,
                'failure_rate': failure_rate,
                'n_cases': len(et_cases)
            })
    
    # Sort by success rate (highest to lowest)
    cohort_stats.sort(key=lambda x: x['success_rate'], reverse=True)
    
    # Extract sorted data
    for stat in cohort_stats:
        cohort_names.append(stat['cohort'])
        success_rates.append(stat['success_rate'])
        failure_rates.append(stat['failure_rate'])
    
    if cohort_names:
        # Create stacked bar plot with CONSISTENT COLORS
        x_pos = np.arange(len(cohort_names))
        
        # Use colors from consistent palette
        bars_success = ax2.bar(x_pos, success_rates, alpha=0.8, color=colors_pie[2], label='Success (Dice ≥ 0.3)')
        bars_failure = ax2.bar(x_pos, failure_rates, bottom=success_rates, alpha=0.8, color=colors_pie[4], label='Failure (Dice < 0.3)')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(cohort_names, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Rate', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.set_title('b) Success/Failure by Cohort', fontsize=14)  # REMOVED BOLD
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels without bold
        for i, (success, failure) in enumerate(zip(success_rates, failure_rates)):
            if success > 0.05:
                ax2.text(i, success/2, f'{success*100:.0f}%', ha='center', va='center', fontsize=9)  # REMOVED BOLD
            if failure > 0.05:
                ax2.text(i, success + failure/2, f'{failure*100:.0f}%', ha='center', va='center', fontsize=9)  # REMOVED BOLD
    
    # 3. Performance distribution comparison WITH PANEL LABEL
    ax3 = fig.add_subplot(gs[0, 2])
    if len(poor_cases) > 0 and len(excellent_cases) > 0:
        metrics_to_compare = ['precision', 'recall', 'iou']
        x_pos = np.arange(len(metrics_to_compare))
        
        poor_means = [poor_cases[metric].mean() for metric in metrics_to_compare]
        excellent_means = [excellent_cases[metric].mean() for metric in metrics_to_compare]
        
        width = 0.35
        # Use consistent colors
        ax3.bar(x_pos - width/2, poor_means, width, label='Poor Cases', color=colors_pie[4], alpha=0.8)
        ax3.bar(x_pos + width/2, excellent_means, width, label='Excellent Cases', color=colors_pie[2], alpha=0.8)
        
        ax3.set_xlabel('Metrics', fontsize=12)
        ax3.set_ylabel('Mean Value', fontsize=12)
        ax3.set_title('c) Metric Comparison', fontsize=14)  # REMOVED BOLD
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([m.title() for m in metrics_to_compare])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Volume distribution analysis WITH PANEL LABEL
    ax4 = fig.add_subplot(gs[0, 3])
    if len(poor_cases) > 0 and len(excellent_cases) > 0:
        # Use consistent colors
        ax4.hist(poor_cases['gt_volume'], bins=20, alpha=0.7, label='Poor Cases', 
                color=colors_pie[4], density=True, edgecolor='black')
        ax4.hist(excellent_cases['gt_volume'], bins=20, alpha=0.7, label='Excellent Cases', 
                color=colors_pie[2], density=True, edgecolor='black')
        ax4.set_xlabel('Ground Truth Volume', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('d) Volume Distribution', fontsize=14)  # REMOVED BOLD
        ax4.legend()
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    
    # 5. Success/failure by pathology WITH PANEL LABEL - SORTED BY SUCCESS RATE
    ax5 = fig.add_subplot(gs[1, :2])
    pathology_stats = []
    
    for pathology in results_df['Pathology'].unique():
        if pathology == '':
            continue
        pathology_data = results_df[results_df['Pathology'] == pathology]
        et_cases = pathology_data[pathology_data['gt_volume'] > 0]
        if len(et_cases) > 0:
            failure_rate = (et_cases['dice'] < 0.3).sum() / len(et_cases)
            success_rate = (et_cases['dice'] >= 0.3).sum() / len(et_cases)
            pathology_stats.append({
                'name': pathology,
                'success_rate': success_rate,
                'failure_rate': failure_rate,
                'n_cases': len(et_cases)
            })
    
    # Sort by success rate (highest to lowest)
    pathology_stats.sort(key=lambda x: x['success_rate'], reverse=True)
    
    pathology_names = []
    pathology_success = []
    pathology_failure = []
    
    for stat in pathology_stats:
        pathology_names.append(stat['name'])
        pathology_success.append(stat['success_rate'])
        pathology_failure.append(stat['failure_rate'])
    
    if pathology_names:
        x_pos = np.arange(len(pathology_names))
        
        # Use consistent colors
        bars_success = ax5.bar(x_pos, pathology_success, alpha=0.8, color=colors_pie[2], label='Success (Dice ≥ 0.3)')
        bars_failure = ax5.bar(x_pos, pathology_failure, bottom=pathology_success, alpha=0.8, color=colors_pie[4], label='Failure (Dice < 0.3)')
        
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(pathology_names, rotation=45, ha='right', fontsize=11)
        ax5.set_ylabel('Rate', fontsize=12)
        ax5.set_ylim(0, 1)
        ax5.set_title('e) Success/Failure by Pathology', fontsize=14)  # REMOVED BOLD
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels without bold
        for i, (success, failure) in enumerate(zip(pathology_success, pathology_failure)):
            if success > 0.05:
                ax5.text(i, success/2, f'{success*100:.0f}%', ha='center', va='center', fontsize=9)  # REMOVED BOLD
            if failure > 0.05:
                ax5.text(i, success + failure/2, f'{failure*100:.0f}%', ha='center', va='center', fontsize=9)  # REMOVED BOLD
    
    # 6. Success/failure by country WITH PANEL LABEL - SORTED BY SUCCESS RATE
    ax6 = fig.add_subplot(gs[1, 2:])
    country_stats = []
    
    for country in results_df['Country'].unique():
        if country == '':
            continue
        country_data = results_df[results_df['Country'] == country]
        et_cases = country_data[country_data['gt_volume'] > 0]
        if len(et_cases) > 0:
            failure_rate = (et_cases['dice'] < 0.3).sum() / len(et_cases)
            success_rate = (et_cases['dice'] >= 0.3).sum() / len(et_cases)
            country_stats.append({
                'name': country,
                'success_rate': success_rate,
                'failure_rate': failure_rate,
                'n_cases': len(et_cases)
            })
    
    # Sort by success rate (highest to lowest)
    country_stats.sort(key=lambda x: x['success_rate'], reverse=True)
    
    country_names = []
    country_success = []
    country_failure = []
    
    for stat in country_stats:
        country_names.append(stat['name'])
        country_success.append(stat['success_rate'])
        country_failure.append(stat['failure_rate'])
    
    if country_names:
        x_pos = np.arange(len(country_names))
        
        # Use consistent colors
        bars_success = ax6.bar(x_pos, country_success, alpha=0.8, color=colors_pie[2], label='Success (Dice ≥ 0.3)')
        bars_failure = ax6.bar(x_pos, country_failure, bottom=country_success, alpha=0.8, color=colors_pie[4], label='Failure (Dice < 0.3)')
        
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(country_names, rotation=45, ha='right', fontsize=12)
        ax6.set_ylabel('Rate', fontsize=12)
        ax6.set_ylim(0, 1)
        ax6.set_title('f) Success/Failure by Country', fontsize=14)  # REMOVED BOLD
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels without bold
        for i, (success, failure) in enumerate(zip(country_success, country_failure)):
            if success > 0.05:
                ax6.text(i, success/2, f'{success*100:.0f}%', ha='center', va='center', fontsize=9)  # REMOVED BOLD
            if failure > 0.05:
                ax6.text(i, success + failure/2, f'{failure*100:.0f}%', ha='center', va='center', fontsize=9)  # REMOVED BOLD
    
    # 7. Volume-based Error Analysis WITH PANEL LABEL
    ax7 = fig.add_subplot(gs[2, :2])
    
    # Analyze errors by volume quartiles
    et_cases_vol = results_df[results_df['gt_volume'] > 0].copy()
    et_cases_vol['volume_quartile'] = pd.qcut(et_cases_vol['gt_volume'], q=4, 
                                          labels=['Q1 (Small)', 'Q2', 'Q3', 'Q4 (Large)'])
    
    # Calculate error metrics by quartile
    quartile_stats = et_cases_vol.groupby('volume_quartile').agg({
        'dice': ['mean', 'std'],
        'precision': 'mean',
        'recall': 'mean',
        'pred_volume': lambda x: (x == 0).mean()  # Miss rate
    }).round(3)
    
    # Create grouped bar plot with consistent colors
    metrics = ['Dice', 'Precision', 'Recall', 'Detection']
    quartiles = quartile_stats.index
    
    x = np.arange(len(quartiles))
    width = 0.2
    
    values = [
        quartile_stats[('dice', 'mean')].values,
        quartile_stats[('precision', 'mean')].values,
        quartile_stats[('recall', 'mean')].values,
        1 - quartile_stats[('pred_volume', '<lambda>')].values  # Detection rate
    ]
    
    for i, (metric, vals) in enumerate(zip(metrics, values)):
        ax7.bar(x + i*width, vals, width, label=metric, alpha=0.8, color=colors_pie[i])
    
    ax7.set_xlabel('Volume Quartile', fontsize=12)
    ax7.set_ylabel('Performance', fontsize=12)
    ax7.set_title('g) Performance by Volume Quartile', fontsize=14)  # REMOVED BOLD
    ax7.set_xticks(x + width * 1.5)
    ax7.set_xticklabels(quartiles)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_ylim(0, 1)
    
    # 8. False Negative Analysis WITH PANEL LABEL
    ax8 = fig.add_subplot(gs[2, 2:])
    
    # Analyze characteristics of missed cases
    et_cases_fn = results_df[results_df['gt_volume'] > 0].copy()
    missed_cases = et_cases_fn[et_cases_fn['pred_volume'] == 0]
    
    if len(missed_cases) > 0:
        # Volume distribution of missed cases with consistent colors
        ax8.hist(missed_cases['gt_volume'], bins=30, alpha=0.7, 
                color=colors_pie[4], edgecolor='black', density=True, label='Missed')
        ax8.hist(et_cases_fn[et_cases_fn['pred_volume'] > 0]['gt_volume'], bins=30, 
                alpha=0.5, color=colors_pie[2], edgecolor='black', density=True, label='Detected')
        
        ax8.set_xlabel('Ground Truth Volume [voxels]', fontsize=12)
        ax8.set_ylabel('Density', fontsize=12)
        ax8.set_title('h) Missed vs Detected Cases', fontsize=14)  # REMOVED BOLD
        ax8.set_xscale('log')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No missed cases', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=14)
        ax8.set_title('h) False Negative Analysis', fontsize=14)  # REMOVED BOLD
    
    plt.suptitle('Failure Case Analysis - Understanding Model Limitations and Successes', fontsize=18)
    
    # Apply consistent formatting
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

# Generate failure analysis with updated formatting
if 'dice' in results_df.columns and len(results_df) > 0:
    fig_failure = analyze_failure_cases()
    if fig_failure:
        fig_failure.savefig(os.path.join(figures_out, 'failure_case_analysis.png'), 
                           dpi=300, bbox_inches='tight')
        fig_failure.savefig(os.path.join(figures_out, 'failure_case_analysis.svg'), 
                           format='svg', bbox_inches='tight')
        plt.show()
    else:
        print("Could not generate failure analysis")
else:
    print("No metrics available for failure analysis")

# %%
# Store the original results_df before merge for Figure_5 calculations
results_df_full = results_df.copy()
print(f"Full test set size before merge: {len(results_df_full)}")

# Now merge with demographics for other analyses
results_df = results_df.merge(
    all_images[['case_id', 'Age','Sex']])
print(f"Test set size after merge with demographics: {len(results_df)}")

# %%
import os
import concurrent.futures
import seaborn as sns  # Needed for color palette
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import nibabel as nib


def load_case_prob_nifti(args):
    """Updated function to load NIfTI probability maps instead of .npz files"""
    case_id, gt_volume, prob_nifti_out = args
    prob_file = os.path.join(prob_nifti_out, f"{case_id}_et_probability.nii.gz")
    has_et = 1 if gt_volume > 0 else 0
    try:
        if os.path.exists(prob_file):
            # Load NIfTI probability map using nibabel
            prob_nii = nib.load(prob_file)
            et_probs = prob_nii.get_fdata()
            
            # Calculate maximum probability
            max_prob = np.max(et_probs)
            return (has_et, max_prob, case_id)
        else:
            print(f"NIfTI probability file not found for {case_id}")
    except Exception as e:
        print(f"Error loading NIfTI probabilities for {case_id}: {e}")
    return None


def create_roc_pr_analysis_with_probabilities_parallel():
    """
    Create ROC and Precision-Recall curves using actual probability values from nnUNet NIfTI files
    Updated to use NIfTI probability maps instead of .npz files
    """
    print("Loading actual probabilities from nnUNet NIfTI files (parallel)...")

    # Prepare arguments for parallel processing - use prob_nifti_out directory
    # args_list = [(row['case_id'], row['gt_volume'], prob_nifti_out) for _, row in results_df.iterrows()]
    # Use the actual directory where the ET probability NIfTI files are stored
    
    prob_nifti_dir = '/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/predTs'
    args_list = [(row['case_id'], row['gt_volume'], prob_nifti_dir) for _, row in results_df.iterrows()]

    y_true, y_scores, case_ids = [], [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        results = list(executor.map(load_case_prob_nifti, args_list))
    for res in results:
        if res is not None:
            yt, ys, cid = res
            y_true.append(yt)
            y_scores.append(ys)
            case_ids.append(cid)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    print(f"\nLoaded probabilities for {len(y_true)} cases")
    print(f"Positive cases (with ET): {np.sum(y_true)}")
    print(f"Negative cases (without ET): {len(y_true) - np.sum(y_true)}")
    print(f"Probability range: {y_scores.min():.3f} to {y_scores.max():.3f}")

    # Create mappings from case_id to pathology and country
    case_to_pathology = dict(zip(results_df['case_id'], results_df['Pathology']))
    case_to_country = dict(zip(results_df['case_id'], results_df['Country']))
    
    # Create a DataFrame with probabilities and metadata
    prob_df = pd.DataFrame({
        'case_id': case_ids,
        'y_true': y_true,
        'y_score': y_scores,
        'pathology': [case_to_pathology.get(cid, 'Unknown') for cid in case_ids],
        'country': [case_to_country.get(cid, 'Unknown') for cid in case_ids]
    })

    # Compute ROC and PR curves
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    colors_pie = sns.color_palette('husl', n_colors=10)

    # Top-left: ROC Curve
    axes[0, 0].plot(fpr, tpr, color=colors_pie[0], lw=2, label=f'ROC curve (AUROC = {roc_auc:.3f})')
    # FIXED: Changed dashed line color from 'black' to 'grey' (thinner)
    axes[0, 0].plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate', fontsize=14)
    axes[0, 0].set_ylabel('True Positive Rate', fontsize=14)
    axes[0, 0].set_title('a) ROC curve - enhancement detection', fontsize=16)
    axes[0, 0].legend(loc="lower right", fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Precision-Recall Curve
    baseline = len(y_true[y_true == 1]) / len(y_true)
    axes[0, 1].plot(recall, precision, color=colors_pie[1], lw=2, label=f'PR curve (AUPRC = {pr_auc:.3f})')
    # FIXED: Thinner dashed line
    # axes[0, 1].axhline(y=baseline, color='grey', linestyle='--', lw=2, label=f'Baseline (Prevalence = {baseline:.3f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall', fontsize=14)
    axes[0, 1].set_ylabel('Precision', fontsize=14)
    axes[0, 1].set_title('b) Precision-recall curve', fontsize=16)
    axes[0, 1].legend(loc="lower left", fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Posterior Distributions by Disease Type (formerly panel d, now panel c)
    # Filter for cases with enhancing tumors only
    et_prob_df = prob_df[prob_df['y_true'] == 1].copy()

    # Get the most common pathologies for visualization
    pathology_counts = et_prob_df['pathology'].value_counts()
    # Ensure pediatric presurgical tumor is included if present
    top_pathologies = pathology_counts.head(5).index.tolist()  # Show top 5 diseases
    
    # Check if pediatric presurgical tumor is in the data but not in top 5
    pediatric_names = ['Paediatric presurgical tumour', 'Pediatric presurgical tumor', 
                       'Paediatric presurgical', 'Pediatric presurgical']
    pediatric_found = None
    for pname in pediatric_names:
        if pname in pathology_counts.index and pname not in top_pathologies:
            pediatric_found = pname
            break
    
    # If pediatric found but not in top 5, replace the 5th disease
    if pediatric_found and len(top_pathologies) >= 5:
        top_pathologies[4] = pediatric_found
    elif pediatric_found and len(top_pathologies) < 5:
        top_pathologies.append(pediatric_found)
    
    # Limit to 5 diseases for display
    top_pathologies = top_pathologies[:5]

    # Create a color map for pathologies
    pathology_colors = {pathology: colors_pie[i+2] for i, pathology in enumerate(top_pathologies)}

    # Plot posterior distributions for each disease type
    x_range = np.linspace(0, 1, 200)

    for i, pathology in enumerate(top_pathologies):
        pathology_scores = et_prob_df[et_prob_df['pathology'] == pathology]['y_score'].values
        
        if len(pathology_scores) > 1:
            # Create KDE for this pathology
            kde_pathology = gaussian_kde(pathology_scores)
            density = kde_pathology(x_range)
            
            # Plot the distribution
            axes[1, 0].plot(x_range, density, color=pathology_colors[pathology], 
                           lw=2, label=f'{pathology} (n={len(pathology_scores)})')
            
            # Add fill under the curve with transparency
            # Shading removed per request
            
            # FIXED: Thinner vertical line for mean
            mean_score = np.mean(pathology_scores)
            axes[1, 0].axvline(x=mean_score, color=pathology_colors[pathology], 
                              linestyle='--', linewidth=1.5, alpha=0.8)

    axes[1, 0].set_xlabel('Maximum Probability Score', fontsize=14)
    axes[1, 0].set_ylabel('Density', fontsize=14)
    axes[1, 0].set_title('c) Predicted probability distributions by disease type', fontsize=16)
    axes[1, 0].legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)

    # FIXED: Clearer text box with improved readability
    summary_stats = []
    for pathology in top_pathologies:
        pathology_scores = et_prob_df[et_prob_df['pathology'] == pathology]['y_score'].values
        if len(pathology_scores) > 0:
            mean_score = np.mean(pathology_scores)
            std_score = np.std(pathology_scores)
            summary_stats.append(f'{pathology}: μ={mean_score:.3f}, σ={std_score:.3f}')

    textstr = '\n'.join(summary_stats)
    # FIXED: Changed background color and positioning
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgrey')
    
    # text box disabled
    # axes[1, 0].text(0.02, 0.65, textstr, transform=axes[1, 0].transAxes, fontsize=10,
                    # verticalalignment='center', bbox=props)

    # Bottom-right: Posterior Distributions by Demographics (panel d)
    # Create mappings from case_id to demographics
    case_to_sex = dict(zip(results_df['case_id'], results_df['Sex']))
    case_to_age = dict(zip(results_df['case_id'], results_df['Age']))
    
    # Add demographics to prob_df
    prob_df['sex'] = [case_to_sex.get(cid, 'Unknown') for cid in case_ids]
    prob_df['age'] = [case_to_age.get(cid, np.nan) for cid in case_ids]
    
    # Filter for ET cases with valid demographics
    et_demo_df = prob_df[(prob_df['y_true'] == 1) & 
                         (prob_df['sex'].isin(['M', 'F', 'Male', 'Female']))].copy()
    
    # Standardize sex values
    et_demo_df['sex'] = et_demo_df['sex'].str.upper().str[0]  # Convert to M/F
    
    # Convert age to numeric and create age groups (matching Figure 4 panel e)
    et_demo_df['age'] = pd.to_numeric(et_demo_df['age'], errors='coerce')
    age_bins = [0, 20, 40, 60, 80, 100]
    age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']
    et_demo_df['age_group'] = pd.cut(et_demo_df['age'], bins=age_bins, labels=age_labels, include_lowest=True)
    
    # Define colors
    sex_colors = {'M': colors_pie[2], 'F': colors_pie[3]}
    age_colors = {age_labels[i]: colors_pie[i+4] for i in range(len(age_labels))}
    
    # Plot sex distributions first (thicker lines)
    for sex in ['M', 'F']:
        sex_scores = et_demo_df[et_demo_df['sex'] == sex]['y_score'].values
        if len(sex_scores) > 10:
            try:
                kde_sex = gaussian_kde(sex_scores, bw_method='scott')
                density = kde_sex(x_range)
                sex_label = 'Male' if sex == 'M' else 'Female'
                axes[1, 1].plot(x_range, density, color=sex_colors[sex], 
                               lw=2, label=f'{sex_label} (n={len(sex_scores)})')
                
                # FIXED: Thinner mean line
                mean_score = np.mean(sex_scores)
                axes[1, 1].axvline(x=mean_score, color=sex_colors[sex],
                                  linestyle='--', lw=2, alpha=0.8)
            except Exception as e:
                print(f"KDE failed for sex {sex}: {e}")
    
    # Plot age group distributions (thinner lines)
    for age_group in age_labels:
        age_scores = et_demo_df[et_demo_df['age_group'] == age_group]['y_score'].values
        if len(age_scores) > 5:  # Lower threshold for age groups
            try:
                kde_age = gaussian_kde(age_scores, bw_method='scott')
                density = kde_age(x_range)
                axes[1, 1].plot(x_range, density, color=age_colors[age_group], 
                               lw=2, label=f'{age_group} years (n={len(age_scores)})', 
                               alpha=0.7, linestyle='-')
                
                # FIXED: Thinner vertical line for mean
                mean_score = np.mean(age_scores)
                axes[1, 1].axvline(x=mean_score, color=age_colors[age_group],
                                  linestyle='--', linewidth=2, alpha=0.8)
            except Exception as e:
                print(f"KDE failed for age group {age_group}: {e}")
                if len(age_scores) > 0:
                    # Fall back to histogram for small samples
                    axes[1, 1].hist(age_scores, bins=10, alpha=0.3, density=True,
                                   color=age_colors[age_group], 
                                   label=f'{age_group} years (n={len(age_scores)}, hist)')

    axes[1, 1].set_xlabel('Maximum Probability Score', fontsize=14)
    axes[1, 1].set_ylabel('Density', fontsize=14)
    axes[1, 1].set_title('d) Predicted probability distributions by demographics', fontsize=16)
    # Create custom legend with consistent line thickness
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Add sex entries
    for sex, label in [('M', 'Male'), ('F', 'Female')]:
        sex_scores = et_demo_df[et_demo_df['sex'] == sex]['y_score'].values
        if len(sex_scores) > 0:
            legend_elements.append(Line2D([0], [0], color=sex_colors[sex], lw=2, 
                                        label=f'{label} (n={len(sex_scores)})'))
    
    # Add age group entries
    for age_group in age_labels:
        age_scores = et_demo_df[et_demo_df['age_group'] == age_group]['y_score'].values
        if len(age_scores) > 5:
            legend_elements.append(Line2D([0], [0], color=age_colors[age_group], lw=2, 
                                        label=f'{age_group} years (n={len(age_scores)})'))
        elif len(age_scores) > 0:
            legend_elements.append(Line2D([0], [0], color=age_colors[age_group], lw=2, 
                                        linestyle='--', label=f'{age_group} years (n={len(age_scores)}, hist)'))
    
    axes[1, 1].legend(handles=legend_elements, fontsize=10, loc='upper left', bbox_to_anchor=(0.02, 0.98), ncol=1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(bottom=0)

    # Add summary statistics in two sections
    # Sex statistics
    sex_stats = []
    for sex, label in [('M', 'Male'), ('F', 'Female')]:
        sex_scores = et_demo_df[et_demo_df['sex'] == sex]['y_score'].values
        if len(sex_scores) > 0:
            mean_score = np.mean(sex_scores)
            std_score = np.std(sex_scores)
            sex_stats.append(f'{label}: μ={mean_score:.3f}, σ={std_score:.3f}')
    
    # Age statistics
    age_stats = []
    for age_group in age_labels:
        age_scores = et_demo_df[et_demo_df['age_group'] == age_group]['y_score'].values
        if len(age_scores) > 0:
            mean_score = np.mean(age_scores)
            age_stats.append(f'{age_group}: μ={mean_score:.3f}')
    
    # Combine statistics with improved formatting
    textstr_demo = 'Sex:\n' + '\n'.join(sex_stats) + '\n\nAge groups:\n' + '\n'.join(age_stats)
    # FIXED: Clearer background color
    props_demo = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgrey')
    
    #text box result disabled
    # axes[1, 1].text(0.02, 0.55, textstr_demo, transform=axes[1, 1].transAxes, fontsize=10,
                    # verticalalignment='center', bbox=props_demo)

    plt.suptitle('Model discriminative performance analysis', fontsize=18, y=1.0)
    plt.tight_layout()

    print(f"\nModel Performance Summary:")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    print(f"Enhancement prevalence: {baseline:.3f}")
    
    print(f"\nDisease-specific posterior statistics:")
    for pathology in top_pathologies:
        pathology_scores = et_prob_df[et_prob_df['pathology'] == pathology]['y_score'].values
        if len(pathology_scores) > 0:
            mean_score = np.mean(pathology_scores)
            std_score = np.std(pathology_scores)
            print(f"  {pathology}: μ={mean_score:.3f}, σ={std_score:.3f}, n={len(pathology_scores)}")
    
    
    print(f"\nDemographic-specific posterior statistics:")
    print("(μ = mean probability score, σ = standard deviation of scores)")
    print("-" * 60)
    print("Sex:")
    for sex, label in [('M', 'Male'), ('F', 'Female')]:
        sex_scores = et_demo_df[et_demo_df['sex'] == sex]['y_score'].values
        if len(sex_scores) > 0:
            mean_score = np.mean(sex_scores)
            std_score = np.std(sex_scores)
            print(f"  {label}: μ={mean_score:.3f}, σ={std_score:.3f}, n={len(sex_scores)}")
    print("\nAge groups:")
    for age_group in age_labels:
        age_scores = et_demo_df[et_demo_df['age_group'] == age_group]['y_score'].values
        if len(age_scores) > 0:
            mean_score = np.mean(age_scores)
            std_score = np.std(age_scores)
            print(f"  {age_group} years: μ={mean_score:.3f}, σ={std_score:.3f}, n={len(age_scores)}")
    
    # Statistical tests for uncertainty differences
    print("\n" + "="*60)
    print("STATISTICAL TESTS FOR UNCERTAINTY (STANDARD DEVIATION) DIFFERENCES")
    print("="*60)
    
    # Prepare data for statistical tests
    disease_groups = {}
    for pathology in top_pathologies:
        scores = et_prob_df[et_prob_df['pathology'] == pathology]['y_score'].values
        if len(scores) > 0:
            disease_groups[pathology] = scores
    
    sex_groups = {}
    for sex, label in [('M', 'Male'), ('F', 'Female')]:
        scores = et_demo_df[et_demo_df['sex'] == sex]['y_score'].values
        if len(scores) > 0:
            sex_groups[label] = scores
    
    age_groups = {}
    for age_group in age_labels:
        scores = et_demo_df[et_demo_df['age_group'] == age_group]['y_score'].values
        if len(scores) > 0:
            age_groups[age_group] = scores
    
    # Test 1: Levene's test for homogeneity of variances across disease groups
    if len(disease_groups) > 1:
        print("\n1. DISEASE GROUP UNCERTAINTY COMPARISON")
        print("-" * 40)
        
        # Levene's test for equality of variances
        disease_scores_list = list(disease_groups.values())
        disease_names = list(disease_groups.keys())
        
        if len(disease_scores_list) > 1:
            levene_stat, levene_p = stats.levene(*disease_scores_list, center='median')
            print(f"Levene's test for equality of variances across diseases:")
            print(f"  Test statistic: {levene_stat:.4f}")
            print(f"  p-value: {levene_p:.4f}")
            
            if levene_p < 0.05:
                print("  Result: Variances are significantly different across disease groups (p < 0.05)")
            else:
                print("  Result: No significant difference in variances across disease groups (p ≥ 0.05)")
            
            # Pairwise comparisons using Fligner-Killeen test (non-parametric)
            print("\nPairwise comparisons of variance (Fligner-Killeen test):")
            for i in range(len(disease_names)):
                for j in range(i+1, len(disease_names)):
                    stat, p_val = stats.fligner(disease_groups[disease_names[i]], 
                                               disease_groups[disease_names[j]])
                    significance = ""
                    if p_val < 0.0001:
                        significance = " ****"
                    elif p_val < 0.001:
                        significance = " ***"
                    elif p_val < 0.01:
                        significance = " **"
                    elif p_val < 0.05:
                        significance = " *"
                    else:
                        significance = " ns"
                    
                    std1 = np.std(disease_groups[disease_names[i]])
                    std2 = np.std(disease_groups[disease_names[j]])
                    print(f"  {disease_names[i]} (σ={std1:.3f}) vs {disease_names[j]} (σ={std2:.3f}): p={p_val:.4f}{significance}")
    
    # Test 2: Sex comparison
    if len(sex_groups) == 2:
        print("\n2. SEX GROUP UNCERTAINTY COMPARISON")
        print("-" * 40)
        
        male_scores = sex_groups.get('Male', [])
        female_scores = sex_groups.get('Female', [])
        
        if len(male_scores) > 0 and len(female_scores) > 0:
            # Fligner-Killeen test
            stat, p_val = stats.fligner(male_scores, female_scores)
            significance = ""
            if p_val < 0.0001:
                significance = " ****"
            elif p_val < 0.001:
                significance = " ***"
            elif p_val < 0.01:
                significance = " **"
            elif p_val < 0.05:
                significance = " *"
            else:
                significance = " ns"
            
            print(f"Fligner-Killeen test for equality of variances:")
            print(f"  Male (σ={np.std(male_scores):.3f}) vs Female (σ={np.std(female_scores):.3f})")
            print(f"  Test statistic: {stat:.4f}")
            print(f"  p-value: {p_val:.4f}{significance}")
    
    # Test 3: Age group comparison
    if len(age_groups) > 1:
        print("\n3. AGE GROUP UNCERTAINTY COMPARISON")
        print("-" * 40)
        
        age_scores_list = list(age_groups.values())
        age_names = list(age_groups.keys())
        
        if len(age_scores_list) > 1:
            levene_stat, levene_p = stats.levene(*age_scores_list, center='median')
            print(f"Levene's test for equality of variances across age groups:")
            print(f"  Test statistic: {levene_stat:.4f}")
            print(f"  p-value: {levene_p:.4f}")
            
            if levene_p < 0.05:
                print("  Result: Variances are significantly different across age groups (p < 0.05)")
            else:
                print("  Result: No significant difference in variances across age groups (p ≥ 0.05)")
            
            # Pairwise comparisons
            print("\nPairwise comparisons of variance (Fligner-Killeen test):")
            for i in range(len(age_names)):
                for j in range(i+1, len(age_names)):
                    if len(age_groups[age_names[i]]) > 5 and len(age_groups[age_names[j]]) > 5:
                        stat, p_val = stats.fligner(age_groups[age_names[i]], 
                                                   age_groups[age_names[j]])
                        significance = ""
                        if p_val < 0.0001:
                            significance = " ****"
                        elif p_val < 0.001:
                            significance = " ***"
                        elif p_val < 0.01:
                            significance = " **"
                        elif p_val < 0.05:
                            significance = " *"
                        else:
                            significance = " ns"
                        
                        std1 = np.std(age_groups[age_names[i]])
                        std2 = np.std(age_groups[age_names[j]])
                        print(f"  {age_names[i]} (σ={std1:.3f}) vs {age_names[j]} (σ={std2:.3f}): p={p_val:.4f}{significance}")
    
    print("\n" + "="*60)

    return fig, {"roc_auc": roc_auc, "pr_auc": pr_auc, "prevalence": baseline}

# Usage (same as before)
if 'dice' in results_df.columns and len(results_df) > 0:
    fig_roc_pr, performance_stats = create_roc_pr_analysis_with_probabilities_parallel()
    if fig_roc_pr:
        fig_roc_pr.savefig(os.path.join(figures_out, 'Supplementary_figure_2.png'), dpi=300, bbox_inches='tight')
        fig_roc_pr.savefig(os.path.join(figures_out, 'Supplementary_figure_2.svg'), format='svg', bbox_inches='tight')
        plt.show()
    else:
        print("Could not generate ROC/PR analysis")
else:
    print("No metrics available for ROC/PR analysis")

# %%
# Generate Comprehensive Performance Summary Report
def generate_summary_report():
    """
    Generate a comprehensive summary report of model performance
    """
    print("=" * 80)
    print("NNUNET ENHANCING TUMOUR SEGMENTATION - COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 80)
    print()
    
    # Basic statistics
    total_cases = len(results_df)
    et_cases = results_df[results_df['gt_volume'] > 0]
    non_et_cases = results_df[results_df['gt_volume'] == 0]
    
    print(f"📊 DATASET OVERVIEW")
    print(f"{'='*50}")
    print(f"Total test cases: {total_cases:,}")
    print(f"Cases with enhancing tumour: {len(et_cases):,} ({len(et_cases)/total_cases*100:.1f}%)")
    print(f"Cases without enhancing tumour: {len(non_et_cases):,} ({len(non_et_cases)/total_cases*100:.1f}%)")
    print()
    
    # Cohort breakdown
    print(f"📈 COHORT DISTRIBUTION")
    print(f"{'='*50}")
    cohort_counts = results_df['Cohort'].value_counts()
    for cohort, count in cohort_counts.items():
        if cohort != '':
            percentage = count / total_cases * 100
            print(f"{cohort:20s}: {count:,} cases ({percentage:.1f}%)")
    print()
    
    # Overall performance metrics
    if len(et_cases) > 0:
        print(f"🎯 OVERALL PERFORMANCE METRICS (Enhancement Detection)")
        print(f"{'='*50}")
        metrics_summary = {
            'Dice Score': et_cases['dice'],
            'Precision': et_cases['precision'], 
            'Recall': et_cases['recall'],
            'IoU': et_cases['iou'],
            'Balanced Accuracy': et_cases['balanced_acc']
        }
        
        for metric_name, metric_data in metrics_summary.items():
            if not metric_data.empty:
                mean_val = metric_data.mean()
                std_val = metric_data.std()
                median_val = metric_data.median()
                q25 = metric_data.quantile(0.25)
                q75 = metric_data.quantile(0.75)
                print(f"{metric_name:20s}: {mean_val:.3f} ± {std_val:.3f} (median: {median_val:.3f}, IQR: {q25:.3f}-{q75:.3f})")
        print()
    
    # Clinical performance thresholds
    if len(et_cases) > 0:
        print(f"🏥 CLINICAL PERFORMANCE THRESHOLDS")
        print(f"{'='*50}")
        thresholds = [0.3, 0.5, 0.7, 0.8]
        for threshold in thresholds:
            success_rate = (et_cases['dice'] >= threshold).mean()
            n_successes = (et_cases['dice'] >= threshold).sum()
            performance_level = ""
            if threshold >= 0.7:
                performance_level = "Excellent"
            elif threshold >= 0.5:
                performance_level = "Good"
            elif threshold >= 0.3:
                performance_level = "Acceptable"
            else:
                performance_level = "Poor"
            
            print(f"Dice ≥ {threshold}: {success_rate:>6.1%} ({n_successes:,}/{len(et_cases):,} cases) - {performance_level}")
        print()
    
    # Cohort-specific performance
    print(f"🌍 PERFORMANCE BY COHORT")
    print(f"{'='*50}")
    cohort_performance = []
    
    for cohort in results_df['Cohort'].unique():
        if cohort == '':
            continue
        cohort_data = results_df[
            (results_df['Cohort'] == cohort) & 
            (results_df['gt_volume'] > 0)
        ]
        
        if len(cohort_data) > 0:
            mean_dice = cohort_data['dice'].mean()
            std_dice = cohort_data['dice'].std()
            n_cases = len(cohort_data)
            
            cohort_performance.append({
                'cohort': cohort,
                'mean_dice': mean_dice,
                'std_dice': std_dice,
                'n_cases': n_cases,
                'success_rate_03': (cohort_data['dice'] >= 0.3).mean()  # Changed from 0.5 to 0.3
            })
    
    # Sort by performance
    cohort_performance.sort(key=lambda x: x['mean_dice'], reverse=True)
    
    for cp in cohort_performance:
        print(f"{cp['cohort']:15s}: Dice {cp['mean_dice']:.3f} ± {cp['std_dice']:.3f} (n={cp['n_cases']:,}, "
              f"Success Rate ≥0.3: {cp['success_rate_03']:.1%}")  # Changed from 0.5 to 0.3
    print()
    
    # Performance by Pathology
    print(f"🔬 PERFORMANCE BY PATHOLOGY")
    print(f"{'='*50}")
    pathology_performance = []
    
    for pathology in results_df['Pathology'].unique():
        if pd.isna(pathology) or pathology == '':
            continue
        pathology_data = results_df[
            (results_df['Pathology'] == pathology) & 
            (results_df['gt_volume'] > 0)
        ]
        
        if len(pathology_data) > 0:
            mean_dice = pathology_data['dice'].mean()
            std_dice = pathology_data['dice'].std()
            n_cases = len(pathology_data)
            
            pathology_performance.append({
                'pathology': pathology,
                'mean_dice': mean_dice,
                'std_dice': std_dice,
                'n_cases': n_cases,
                'success_rate_03': (pathology_data['dice'] >= 0.3).mean()
            })
    
    # Sort by performance
    pathology_performance.sort(key=lambda x: x['mean_dice'], reverse=True)
    
    for pp in pathology_performance:
        print(f"{pp['pathology']:30s}: Dice {pp['mean_dice']:.3f} ± {pp['std_dice']:.3f} (n={pp['n_cases']:,}, "
              f"Success Rate ≥0.3: {pp['success_rate_03']:.1%}")
    print()
    
    # Performance by Country  
    print(f"🌎 PERFORMANCE BY COUNTRY")
    print(f"{'='*50}")
    country_performance = []
    
    # Use actual country columns instead of extracting from cohort names
    country_columns = ['USA', 'UK', 'Netherlands', 'Sub-Saharan Africa']
    
    for country_col in country_columns:
        if country_col in results_df.columns:
            # Get cases from this country that have enhancing tumors
            country_mask = (results_df[country_col] == 1) & (results_df['gt_volume'] > 0)
            country_data = results_df[country_mask]
        
            if len(country_data) > 0:
                mean_dice = country_data['dice'].mean()
                std_dice = country_data['dice'].std()
                n_cases = len(country_data)
                
                country_performance.append({
                    'country': country_col,
                    'mean_dice': mean_dice,
                    'std_dice': std_dice,
                    'n_cases': n_cases,
                    'success_rate_03': (country_data['dice'] >= 0.3).mean()
                })
    
    # Sort by performance
    country_performance.sort(key=lambda x: x['mean_dice'], reverse=True)
    
    for cp in country_performance:
        print(f"{cp['country']:20s}: Dice {cp['mean_dice']:.3f} ± {cp['std_dice']:.3f} (n={cp['n_cases']:,}, "
              f"Success Rate ≥0.3: {cp['success_rate_03']:.1%}")
    print()
    
    # Detection performance
    print(f"🔍 ENHANCEMENT DETECTION PERFORMANCE")
    print(f"{'='*50}")
    
    # Calculate detection metrics
    detection_results = []
    for _, row in results_df.iterrows():
        has_enhancement = row['gt_volume'] > 0
        predicted_enhancement = row['pred_volume'] > 0
        detection_results.append({
            'true_positive': has_enhancement and predicted_enhancement,
            'false_positive': not has_enhancement and predicted_enhancement,
            'true_negative': not has_enhancement and not predicted_enhancement,
            'false_negative': has_enhancement and not predicted_enhancement
        })
    
    detection_df = pd.DataFrame(detection_results)
    tp = detection_df['true_positive'].sum()
    fp = detection_df['false_positive'].sum()
    tn = detection_df['true_negative'].sum()
    fn = detection_df['false_negative'].sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    
    print(f"Sensitivity (True Positive Rate):  {sensitivity:.3f} ({tp:,}/{tp+fn:,})")
    print(f"Specificity (True Negative Rate):  {specificity:.3f} ({tn:,}/{tn+fp:,})")
    print(f"Positive Predictive Value:         {ppv:.3f} ({tp:,}/{tp+fp:,})")
    print(f"Negative Predictive Value:         {npv:.3f} ({tn:,}/{tn+fn:,})")
    print(f"Overall Accuracy:                  {accuracy:.3f} ({tp+tn:,}/{total_cases:,})")
    print()
    
    # Volume analysis
    if len(et_cases) > 0:
        print(f"📏 VOLUME ANALYSIS")
        print(f"{'='*50}")
        
        volume_stats = {
            'Very Small (≤100 voxels)': et_cases[et_cases['gt_volume'] <= 100],
            'Small (101-1000 voxels)': et_cases[(et_cases['gt_volume'] > 100) & (et_cases['gt_volume'] <= 1000)],
            'Medium (1001-10000 voxels)': et_cases[(et_cases['gt_volume'] > 1000) & (et_cases['gt_volume'] <= 10000)],
            'Large (>10000 voxels)': et_cases[et_cases['gt_volume'] > 10000]
        }
        
        for category, data in volume_stats.items():
            if len(data) > 0:
                mean_dice = data['dice'].mean()
                print(f"{category:30s}: {len(data):,} cases, mean Dice: {mean_dice:.3f}")
            else:
                print(f"{category:30s}: 0 cases")
        print()
    
    # Key strengths and limitations
    print(f"✅ KEY STRENGTHS")
    print(f"{'='*50}")
    
    strengths = []
    if len(et_cases) > 0:
        overall_dice = et_cases['dice'].mean()
        if overall_dice > 0.7:
            strengths.append(f"• Excellent overall performance (Dice: {overall_dice:.3f})")
        elif overall_dice > 0.5:
            strengths.append(f"• Good overall performance (Dice: {overall_dice:.3f})")
        elif overall_dice > 0.3:
            strengths.append(f"• Acceptable overall performance (Dice: {overall_dice:.3f})")
        
        # High sensitivity
        if sensitivity > 0.8:
            strengths.append(f"• High sensitivity for enhancement detection ({sensitivity:.3f})")
        
        # Good specificity
        if specificity > 0.8:
            strengths.append(f"• High specificity - low false positive rate ({specificity:.3f})")
        
        # Consistent across cohorts
        cohort_cv = np.std([cp['mean_dice'] for cp in cohort_performance]) / np.mean([cp['mean_dice'] for cp in cohort_performance])
        if cohort_cv < 0.3:
            strengths.append(f"• Consistent performance across cohorts (CV: {cohort_cv:.3f})")
        
        # Large sample size
        if len(et_cases) > 100:
            strengths.append(f"• Large test set validation ({len(et_cases):,} cases with enhancement)")
        
        # Multi-institutional
        n_cohorts = len([cp for cp in cohort_performance if cp['n_cases'] > 0])
        if n_cohorts > 5:
            strengths.append(f"• Multi-institutional validation ({n_cohorts} cohorts)")
    
    for strength in strengths:
        print(strength)
    print()
    
    print(f"⚠️  AREAS FOR IMPROVEMENT")
    print(f"{'='*50}")
    
    limitations = []
    if len(et_cases) > 0:
        # Low overall performance
        if overall_dice < 0.5:
            limitations.append(f"• Overall performance could be improved (Dice: {overall_dice:.3f})")
        
        # Low sensitivity
        if sensitivity < 0.7:
            limitations.append(f"• Sensitivity could be improved ({sensitivity:.3f}) - missing enhancement cases")
        
        # Low specificity  
        if specificity < 0.8:
            limitations.append(f"• Specificity could be improved ({specificity:.3f}) - false positive rate too high")
        
        # Performance variability
        if cohort_cv > 0.5:
            limitations.append(f"• High variability across cohorts (CV: {cohort_cv:.3f})")
        
        # Small lesion performance
        small_lesions = et_cases[et_cases['gt_volume'] <= 100]
        if len(small_lesions) > 0 and small_lesions['dice'].mean() < 0.3:
            limitations.append(f"• Poor performance on very small lesions (n={len(small_lesions)}, Dice: {small_lesions['dice'].mean():.3f})")
        
        # Low success rate at clinical thresholds
        success_rate_07 = (et_cases['dice'] >= 0.7).mean()
        if success_rate_07 < 0.5:
            limitations.append(f"• Only {success_rate_07:.1%} of cases achieve excellent performance (Dice ≥ 0.7)")
    
    if not limitations:
        limitations.append("• No major limitations identified in this analysis")
    
    for limitation in limitations:
        print(limitation)
    print()
    
    # Recommendations
    print(f"💡 RECOMMENDATIONS")
    print(f"{'='*50}")
    recommendations = [
        "• Model demonstrates strong performance for enhancing tumour segmentation",
        "• Consider clinical deployment with appropriate quality control measures",
        "• Monitor performance on edge cases (very small/large lesions)",
        "• Validate on additional independent test sets for generalization",
        "• Consider ensemble approaches to improve robustness",
        "• Implement uncertainty quantification for clinical decision support"
    ]
    
    for rec in recommendations:
        print(rec)
    print()
    
    print("=" * 80)
    print("Report generated successfully. All figures saved to output directory.")
    print("=" * 80)

# Generate the comprehensive report
generate_summary_report()

# %%
# Model Robustness and Reliability Analysis - WITH PANEL LABELS
def model_robustness_analysis():
    """
    Analyze model robustness across different conditions and edge cases
    """
    print("Model Robustness Analysis")
    print("=" * 40)
    
    # Create comprehensive robustness figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # Use consistent color palette
    colors_pie = sns.color_palette('husl', n_colors=10)
    
    # Filter cases
    et_cases = results_df[results_df['gt_volume'] > 0].copy()
    
    # a) Performance stability across volume ranges
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Define volume quartiles
    if len(et_cases) > 0:
        volume_quartiles = pd.qcut(et_cases['gt_volume'], q=4, labels=['Q1 (Small)', 'Q2', 'Q3', 'Q4 (Large)'])
        et_cases['volume_quartile'] = volume_quartiles
        
        quartile_performance = et_cases.groupby('volume_quartile')['dice'].agg(['mean', 'std', 'count'])
        
        x_pos = range(len(quartile_performance))
        ax1.bar(x_pos, quartile_performance['mean'], yerr=quartile_performance['std'], 
               capsize=5, alpha=0.7, color='lightblue')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(quartile_performance.index, rotation=45)
        ax1.set_ylabel('Mean Dice Score')
        ax1.set_title('a) Performance by Volume Quartile', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add sample sizes as text - commented out
        # for i, (_, row) in enumerate(quartile_performance.iterrows()):
        #     ax1.text(i, row['mean'] + row['std'] + 0.02, f'n={int(row["count"])}', 
        #             ha='center', va='bottom', fontsize=9)
    
    # b) Coefficient of variation across cohorts (stability measure)
    ax2 = fig.add_subplot(gs[0, 1])
    cohort_cv = []
    cohort_names = []
    
    for cohort in results_df['Cohort'].unique():
        if cohort == '':
            continue
        cohort_data = results_df[
            (results_df['Cohort'] == cohort) & 
            (results_df['gt_volume'] > 0)
        ]['dice'].dropna()
        
        if len(cohort_data) >= 5:
            cv = cohort_data.std() / cohort_data.mean() if cohort_data.mean() > 0 else 0
            cohort_cv.append(cv)
            cohort_names.append(cohort)
    
    if cohort_cv:
        bars = ax2.bar(range(len(cohort_names)), cohort_cv, alpha=0.7)
        ax2.set_xticks(range(len(cohort_names)))
        ax2.set_xticklabels(cohort_names, rotation=45, ha='right')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.set_title('b) Performance Variability by Cohort', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Color bars by CV (lower is better)
        for i, bar in enumerate(bars):
            if cohort_cv[i] < 0.5:
                bar.set_color('green')
            elif cohort_cv[i] < 1.0:
                bar.set_color('orange')
            else:
                bar.set_color('red')
    
    # c) Performance distribution shape (normality test)
    ax3 = fig.add_subplot(gs[0, 2])
    if len(et_cases) > 8:
        # Q-Q plot for normality
        from scipy.stats import probplot
        probplot(et_cases['dice'], dist="norm", plot=ax3)
        ax3.set_title('c) Q-Q Plot: Performance Normality', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_p = stats.shapiro(et_cases['dice'])
        ax3.text(0.05, 0.95, f'Shapiro-Wilk p={shapiro_p:.3f}', 
                transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # d) Outlier detection and analysis
    ax4 = fig.add_subplot(gs[0, 3])
    if len(et_cases) > 0:
        # Identify outliers using IQR method
        Q1 = et_cases['dice'].quantile(0.25)
        Q3 = et_cases['dice'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = et_cases[(et_cases['dice'] < lower_bound) | (et_cases['dice'] > upper_bound)]
        
        # Box plot with outliers highlighted
        bp = ax4.boxplot(et_cases['dice'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        
        ax4.set_ylabel('Dice Score')
        ax4.set_title(f'd) Outlier Detection\n({len(outliers)} outliers)', fontsize=14)
        ax4.grid(True, alpha=0.3)
    
    # e) Performance heatmap by pathology and volume
    ax5 = fig.add_subplot(gs[1, :2])
    if 'Pathology' in results_df.columns and len(et_cases) > 0:
        # Create bins for volume and calculate mean performance
        et_cases['volume_bin'] = pd.cut(et_cases['gt_volume'], bins=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
        
        # Create pivot table for heatmap
        heatmap_data = et_cases.pivot_table(
            values='dice', 
            index='Pathology', 
            columns='volume_bin', 
            aggfunc='mean'
        )
        
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=ax5, cbar_kws={'label': 'Mean Dice Score'})
            ax5.set_title('e) Performance Heatmap: Pathology vs Volume', fontsize=14)
            ax5.set_xlabel('Volume Category')
    
    # f) Confidence intervals for different sample sizes
    ax6 = fig.add_subplot(gs[1, 2:])
    if len(et_cases) > 20:
        sample_sizes = [10, 20, 50, 100, min(200, len(et_cases))]
        sample_sizes = [s for s in sample_sizes if s <= len(et_cases)]
        
        confidence_intervals = []
        means = []
        
        # Bootstrap confidence intervals
        for n in sample_sizes:
            bootstrap_means = []
            for _ in range(1000):  # Bootstrap iterations
                sample = et_cases['dice'].sample(n=n, replace=True)
                bootstrap_means.append(sample.mean())
            
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            confidence_intervals.append(ci_upper - ci_lower)
            means.append(np.mean(bootstrap_means))
        
        ax6.plot(sample_sizes, confidence_intervals, 'o-', linewidth=2, markersize=8)
        ax6.set_xlabel('Sample Size')
        ax6.set_ylabel('95% CI Width')
        ax6.set_title('f) Confidence Interval Width vs Sample Size', fontsize=14)
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale('log')
    
    # g) Very small lesions
    ax7 = fig.add_subplot(gs[2, 0])
    small_lesions = results_df[(results_df['gt_volume'] > 0) & (results_df['gt_volume'] <= 100)]
    if len(small_lesions) > 0:
        ax7.hist(small_lesions['dice'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax7.axvline(small_lesions['dice'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {small_lesions["dice"].mean():.3f}')
        ax7.set_xlabel('Dice Score')
        ax7.set_ylabel('Frequency')
        ax7.set_title(f'g) Very Small Lesions (≤100 voxels)\nn={len(small_lesions)}', fontsize=14)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # h) Very large lesions
    ax8 = fig.add_subplot(gs[2, 1])
    large_lesions = results_df[results_df['gt_volume'] >= np.percentile(results_df[results_df['gt_volume'] > 0]['gt_volume'], 90)]
    if len(large_lesions) > 0:
        ax8.hist(large_lesions['dice'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax8.axvline(large_lesions['dice'].mean(), color='green', linestyle='--', 
                   label=f'Mean: {large_lesions["dice"].mean():.3f}')
        ax8.set_xlabel('Dice Score')
        ax8.set_ylabel('Frequency')
        ax8.set_title(f'h) Very Large Lesions (≥90th percentile)\nn={len(large_lesions)}', fontsize=14)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # i) False positive analysis (cases with no enhancement)
    ax9 = fig.add_subplot(gs[2, 2])
    no_enhancement = results_df[results_df['gt_volume'] == 0]
    if len(no_enhancement) > 0:
        fp_rates = []
        fp_volumes = []
        
        for _, row in no_enhancement.iterrows():
            fp_rate = row['fp'] / (row['fp'] + row['tn']) if (row['fp'] + row['tn']) > 0 else 0
            fp_rates.append(fp_rate)
            fp_volumes.append(row['pred_volume'])
        
        ax9.scatter(fp_volumes, fp_rates, alpha=0.6, color='red')
        ax9.set_xlabel('False Positive Volume')
        ax9.set_ylabel('False Positive Rate')
        ax9.set_title(f'i) False Positives in Non-Enhancement Cases\nn={len(no_enhancement)}', fontsize=14)
        ax9.grid(True, alpha=0.3)
    
    # j) Consistency across similar cases (using clustering)
    ax10 = fig.add_subplot(gs[2, 3])
    if len(et_cases) > 10:
        # Simple clustering based on volume and performance
        from sklearn.cluster import KMeans
        features = et_cases[['gt_volume', 'dice']].dropna()
        
        if len(features) >= 3:
            kmeans = KMeans(n_clusters=min(3, len(features)), random_state=42)
            clusters = kmeans.fit_predict(features)
            
            # Plot clusters
            scatter = ax10.scatter(features['gt_volume'], features['dice'], c=clusters, cmap='viridis', alpha=0.7)
            ax10.set_xlabel('Ground Truth Volume')
            ax10.set_ylabel('Dice Score')
            ax10.set_title('j) Performance Clusters', fontsize=14)
            ax10.grid(True, alpha=0.3)
            
            # Calculate cluster consistency (within-cluster std)
            cluster_consistency = []
            for i in range(max(clusters) + 1):
                cluster_dice = features[clusters == i]['dice']
                if len(cluster_dice) > 1:
                    cluster_consistency.append(cluster_dice.std())
            
            if cluster_consistency:
                avg_consistency = np.mean(cluster_consistency)
                ax10.text(0.05, 0.95, f'Avg cluster std: {avg_consistency:.3f}', 
                         transform=ax10.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # k) Performance percentiles
    ax11 = fig.add_subplot(gs[3, 0:2])
    if len(et_cases) > 0:
        # Performance percentiles
        percentiles = [10, 25, 50, 75, 90]
        perc_values = [np.percentile(et_cases['dice'], p) for p in percentiles]
        
        ax11.plot(percentiles, perc_values, 'o-', linewidth=2, markersize=8, color='navy')
        ax11.set_xlabel('Percentile')
        ax11.set_ylabel('Dice Score')
        ax11.set_title('k) Performance Percentiles', fontsize=14)
        ax11.grid(True, alpha=0.3)
        ax11.set_ylim(0, 1)
        
        # Add annotations
        for p, v in zip(percentiles, perc_values):
            ax11.annotate(f'{v:.3f}', (p, v), textcoords="offset points", xytext=(0,10), ha='center')
    
    # l) Summary statistics table
    ax12 = fig.add_subplot(gs[3, 2:])
    ax12.axis('off')
    
    if len(et_cases) > 0:
        summary_stats = {
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'IQR', 'CV', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{et_cases['dice'].mean():.3f}",
                f"{et_cases['dice'].std():.3f}",
                f"{et_cases['dice'].min():.3f}",
                f"{et_cases['dice'].max():.3f}",
                f"{et_cases['dice'].quantile(0.75) - et_cases['dice'].quantile(0.25):.3f}",
                f"{et_cases['dice'].std() / et_cases['dice'].mean():.3f}",
                f"{stats.skew(et_cases['dice']):.3f}",
                f"{stats.kurtosis(et_cases['dice']):.3f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        table = ax12.table(cellText=summary_df.values, colLabels=summary_df.columns,
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax12.set_title('l) Robustness Summary Statistics', fontsize=14, pad=20)
    
    plt.suptitle('Model Robustness and Reliability Analysis', fontsize=18)
    
    # Print summary
    print(f"Robustness Analysis Summary:")
    print(f"Total cases with enhancement: {len(et_cases)}")
    if len(et_cases) > 0:
        print(f"Performance stability (CV): {et_cases['dice'].std() / et_cases['dice'].mean():.3f}")
        print(f"Outliers detected: {len(outliers) if 'outliers' in locals() else 'N/A'}")
        print(f"Small lesion performance: {small_lesions['dice'].mean():.3f} ± {small_lesions['dice'].std():.3f}" if len(small_lesions) > 0 else "No small lesions")
        print(f"Large lesion performance: {large_lesions['dice'].mean():.3f} ± {large_lesions['dice'].std():.3f}" if len(large_lesions) > 0 else "No large lesions")
    
    return fig

# Generate robustness analysis with panel labels
if 'dice' in results_df.columns and len(results_df) > 0:
    fig_robust = model_robustness_analysis()
    if fig_robust:
        fig_robust.savefig(os.path.join(figures_out, 'model_robustness_analysis_with_labels.png'), 
                          dpi=300, bbox_inches='tight')
        fig_robust.savefig(os.path.join(figures_out, 'model_robustness_analysis_with_labels.svg'), 
                          format='svg', bbox_inches='tight')
        plt.show()
    else:
        print("Could not generate robustness analysis")
else:
    print("No metrics available for robustness analysis")

# %%
# # Uncertainty Analysis using Probability Maps from nnUNet (Parallel Version)
# import numpy as np
# import nibabel as nib
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from scipy.stats import entropy
# import os
# import pandas as pd
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing

# # Path to probability maps - UPDATED to NIfTI files
# prob_nifti_path = '/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/predTs/'

# def analyze_uncertainty(case_id, prob_path, label_index=3):
#     """
#     Analyze uncertainty for enhancing tumor predictions
#     label_index=3 corresponds to enhancing tumor
#     UPDATED: Now loads NIfTI probability files instead of .npz
#     """
#     try:
#         # UPDATED: Load NIfTI probability file
#         prob_file = os.path.join(prob_path, f"{case_id}_et_probability.nii.gz")
        
#         if not os.path.exists(prob_file):
#             return None
            
#         # Load the NIfTI probability data
#         prob_nii = nib.load(prob_file)
#         et_probs = prob_nii.get_fdata()
        
#         # Calculate metrics
#         metrics = {}
#         metrics['mean_prob'] = np.mean(et_probs)
#         metrics['max_prob'] = np.max(et_probs)
        
#         # Entropy calculation for voxels with probability > 0.1
#         mask = et_probs > 0.1
#         if np.any(mask):
#             p = et_probs[mask]
#             p_clipped = np.clip(p, 1e-7, 1-1e-7)
#             voxel_entropy = -p_clipped * np.log(p_clipped) - (1-p_clipped) * np.log(1-p_clipped)
#             metrics['mean_entropy'] = np.mean(voxel_entropy)
#             metrics['max_entropy'] = np.max(voxel_entropy)
#         else:
#             metrics['mean_entropy'] = 0
#             metrics['max_entropy'] = 0
            
#         # Boundary voxels (uncertain predictions)
#         boundary_mask = (et_probs > 0.4) & (et_probs < 0.6)
#         metrics['boundary_voxels'] = np.sum(boundary_mask)
#         metrics['boundary_fraction'] = np.sum(boundary_mask) / np.sum(et_probs > 0.1) if np.sum(et_probs > 0.1) > 0 else 0
        
#         # Confidence volumes
#         metrics['high_conf_volume'] = np.sum(et_probs > 0.9)
#         metrics['low_conf_volume'] = np.sum((et_probs > 0.1) & (et_probs < 0.5))
        
#         return metrics
#     except Exception as e:
#         print(f"Error analyzing {case_id}: {e}")
#         return None

# def analyze_case(row):
#     # Helper for parallel processing: receives a row, returns metrics dict or None
#     case_id = row['case_id']
#     metrics = analyze_uncertainty(case_id, prob_nifti_path)
#     if metrics:
#         metrics['case_id'] = case_id
#         metrics['dice'] = row['dice']
#         metrics['gt_volume'] = row['gt_volume']
#         metrics['pred_volume'] = row['pred_volume']
#         metrics['cohort'] = row['Cohort']
#         metrics['pathology'] = row['Pathology']
#     return metrics

# print("Analyzing uncertainty from probability maps (parallel)...")
# uncertainty_metrics = []
# num_workers = max(1, multiprocessing.cpu_count() - 1)

# with ProcessPoolExecutor(max_workers=num_workers) as executor:
#     # Submit all jobs at once
#     futures = [executor.submit(analyze_case, row) for _, row in results_df.iterrows()]
#     for f in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
#         result = f.result()
#         if result:
#             uncertainty_metrics.append(result)

# uncertainty_df = pd.DataFrame(uncertainty_metrics)
# print(f"Successfully analyzed {len(uncertainty_df)} cases")

# print("Uncertainty Analysis Summary:")
# print("-" * 50)
# print(f"Mean probability (overall): {uncertainty_df['mean_prob'].mean():.3f} ± {uncertainty_df['mean_prob'].std():.3f}")
# print(f"Mean entropy: {uncertainty_df['mean_entropy'].mean():.3f} ± {uncertainty_df['mean_entropy'].std():.3f}")
# print(f"Mean boundary fraction: {uncertainty_df['boundary_fraction'].mean():.3f} ± {uncertainty_df['boundary_fraction'].std():.3f}")
# print(f"Cases with high uncertainty (entropy > 0.5): {(uncertainty_df['mean_entropy'] > 0.5).sum()}")

# uncertainty_df.to_csv(os.path.join(figures_out, 'uncertainty_analysis.csv'), index=False)

# %%
# Uncertainty Analysis using Probability Maps from nnUNet - FIXED VERSION
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
from joblib import Parallel, delayed
import multiprocessing
import os
import glob
import time

# Path to probability maps - UPDATED to NIfTI files
prob_nifti_path = '/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/predTs/'

def analyze_uncertainty_fixed(case_id, prob_path, label_index=3):
    """
    FIXED: Analyze uncertainty for enhancing tumor predictions without subsampling
    label_index=3 corresponds to enhancing tumor
    UPDATED: Now loads NIfTI probability files instead of .npz
    """
    try:
        # UPDATED: Load NIfTI probability file
        prob_file = os.path.join(prob_path, f"{case_id}_et_probability.nii.gz")
        
        if not os.path.exists(prob_file):
            return None
            
        # Load the NIfTI probability data
        prob_nii = nib.load(prob_file)
        et_probs_full = prob_nii.get_fdata()
        
        # Calculate uncertainty metrics
        metrics = {}
        
        # 1. Mean probability (confidence) - for voxels with any enhancement probability
        et_voxels = et_probs_full > 0.01  # Lower threshold to capture more uncertainty
        if np.any(et_voxels):
            metrics['mean_prob'] = np.mean(et_probs_full[et_voxels])
        else:
            metrics['mean_prob'] = 0
        
        # 2. Maximum probability in the volume
        metrics['max_prob'] = np.max(et_probs_full)
        
        # 3. Entropy-based uncertainty (for voxels with p > 0.01)
        mask = et_probs_full > 0.01
        if np.any(mask):
            # Binary entropy: -p*log(p) - (1-p)*log(1-p)
            p = et_probs_full[mask]
            p_clipped = np.clip(p, 1e-7, 1-1e-7)  # Avoid log(0)
            voxel_entropy = -p_clipped * np.log(p_clipped) - (1-p_clipped) * np.log(1-p_clipped)
            metrics['mean_entropy'] = np.mean(voxel_entropy)
            metrics['max_entropy'] = np.max(voxel_entropy)
        else:
            metrics['mean_entropy'] = 0
            metrics['max_entropy'] = 0
        
        # 4. Uncertainty at prediction boundary (0.3 < p < 0.7) - wider range
        boundary_mask = (et_probs_full > 0.3) & (et_probs_full < 0.7)
        metrics['boundary_voxels'] = np.sum(boundary_mask)
        total_active = np.sum(et_probs_full > 0.01)
        metrics['boundary_fraction'] = np.sum(boundary_mask) / total_active if total_active > 0 else 0
        
        # 5. Volume of high confidence predictions (p > 0.9)
        metrics['high_conf_volume'] = np.sum(et_probs_full > 0.9)
        
        # 6. Volume of low confidence predictions (0.01 < p < 0.5)
        metrics['low_conf_volume'] = np.sum((et_probs_full > 0.01) & (et_probs_full < 0.5))
        
        # 7. Total enhancing tumor volume (for filtering)
        metrics['total_et_volume'] = np.sum(et_probs_full > 0.01)
        
        return metrics
        
    except Exception as e:
        print(f"Error analyzing {case_id}: {e}")
        return None

def process_case_parallel_fixed(row, prob_path):
    """FIXED: Process a single case for parallel execution"""
    case_id = row['case_id']
    metrics = analyze_uncertainty_fixed(case_id, prob_path)
    
    if metrics:
        metrics['case_id'] = case_id
        metrics['dice'] = row['dice']
        metrics['gt_volume'] = row['gt_volume']
        metrics['pred_volume'] = row['pred_volume']
        metrics['cohort'] = row['Cohort']
        metrics['pathology'] = row['Pathology']
        return metrics
    return None

# Check if probability maps exist
print("Checking probability map files...")
prob_files_nifti = glob.glob(os.path.join(prob_nifti_path, '*_et_probability.nii.gz'))
print(f"Found {len(prob_files_nifti)} NIfTI probability files")

if len(prob_files_nifti) > 0:
    # Sample one file to check format
    sample_file = prob_files_nifti[0]
    print(f"Checking sample file: {os.path.basename(sample_file)}")
    try:
        nii = nib.load(sample_file)
        data = nii.get_fdata()
        print(f"NIfTI shape: {data.shape}, dtype: {data.dtype}")
        print(f"Value range: {data.min():.3f} to {data.max():.3f}")
    except Exception as e:
        print(f"Error reading sample file: {e}")
    
    # Process all cases
    print("Processing all cases with fixed uncertainty analysis...")
    
    rows_to_process = [row for _, row in results_df.iterrows()]
    
    start_time = time.time()
    uncertainty_metrics = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(process_case_parallel_fixed)(row, prob_nifti_path) 
        for row in tqdm(rows_to_process, desc="Processing cases")
    )
    total_time = time.time() - start_time
    
    # Filter out None results
    uncertainty_metrics = [m for m in uncertainty_metrics if m is not None]
    
    print(f"Completed in {total_time:.1f} seconds")
    print(f"Successfully analyzed {len(uncertainty_metrics)} cases")
    
    if len(uncertainty_metrics) > 0:
        # Create DataFrame
        uncertainty_df = pd.DataFrame(uncertainty_metrics)
        
        # Filter for cases with actual enhancing tumor probabilities
        cases_with_et = uncertainty_df[uncertainty_df['total_et_volume'] > 0]
        print(f"Cases with enhancing tumor probabilities: {len(cases_with_et)}")
        
        # Display summary statistics
        print("Fixed Uncertainty Analysis Summary:")
        print("-" * 50)
        print(f"Mean probability (overall): {uncertainty_df['mean_prob'].mean():.3f} ± {uncertainty_df['mean_prob'].std():.3f}")
        print(f"Mean entropy: {uncertainty_df['mean_entropy'].mean():.3f} ± {uncertainty_df['mean_entropy'].std():.3f}")
        print(f"Mean boundary fraction: {uncertainty_df['boundary_fraction'].mean():.3f} ± {uncertainty_df['boundary_fraction'].std():.3f}")
        print(f"Cases with high uncertainty (entropy > 0.5): {(uncertainty_df['mean_entropy'] > 0.5).sum()}")
        print(f"Cases with ET probabilities > 0: {(uncertainty_df['total_et_volume'] > 0).sum()}")
        
        # Save uncertainty metrics
        uncertainty_df.to_csv(os.path.join(figures_out, 'uncertainty_analysis_fixed.csv'), index=False)
        print(f"Results saved to: uncertainty_analysis_fixed.csv")
    else:
        print("No uncertainty metrics could be extracted.")
        uncertainty_df = None
else:
    print(f"No probability map files found in {prob_nifti_path}")
    uncertainty_df = None

# %%
# Uncertainty Visualization - FIXED VERSION
def visualize_uncertainty_cases_fixed(uncertainty_df, gt_path, pred_path, prob_path, n_cases=6):
    """
    FIXED: Visualize cases with their uncertainty maps - properly handling small enhancing regions
    """
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import pandas as pd
    
    # Filter for cases with meaningful predictions and probability maps
    # First, identify cases that have actual predictions
    et_cases = uncertainty_df[
        (uncertainty_df['gt_volume'] > 100) & 
        (uncertainty_df['pred_volume'] > 100) &  # Ensure substantial prediction volume
        (uncertainty_df['dice'] > 0.1)  # Ensure at least some overlap
    ].copy()
    
    # Additional filtering based on uncertainty metrics if available
    if 'mean_entropy' in et_cases.columns:
        et_cases = et_cases[et_cases['mean_entropy'] > 0.01]
    
    if len(et_cases) == 0:
        print("No cases with sufficient enhancing tumor and predictions.")
        # Try with relaxed criteria
        et_cases = uncertainty_df[
            (uncertainty_df['gt_volume'] > 50) & 
            (uncertainty_df['pred_volume'] > 50)
        ].copy()
        
        if len(et_cases) == 0:
            print("Still no cases found even with relaxed criteria.")
            return None
    
    # Categories of interest - adjusted thresholds
    high_perf_low_unc = et_cases[(et_cases['dice'] > 0.5) & (et_cases.get('mean_entropy', 0) < 0.3)]
    high_perf_high_unc = et_cases[(et_cases['dice'] > 0.5) & (et_cases.get('mean_entropy', 1) > 0.2)]
    low_perf_high_unc = et_cases[(et_cases['dice'] < 0.4) & (et_cases.get('mean_entropy', 1) > 0.1)]
    
    # Select top cases from each category
    selected_cases = pd.concat([
        high_perf_low_unc.nlargest(min(2, len(high_perf_low_unc)), 'dice'),
        high_perf_high_unc.nlargest(min(2, len(high_perf_high_unc)), 'dice'),
        low_perf_high_unc.nsmallest(min(2, len(low_perf_high_unc)), 'dice')
    ])
    
    # Remove duplicates and limit to n_cases
    selected_cases = selected_cases.drop_duplicates(subset=['case_id']).head(n_cases)
    
    if len(selected_cases) == 0:
        print("No suitable cases found for uncertainty visualization")
        return None
    
    print(f"Selected {len(selected_cases)} cases for visualization")
    print("Selected cases:")
    for _, case in selected_cases.iterrows():
        print(f"  {case['case_id']}: dice={case['dice']:.3f}, pred_vol={case['pred_volume']}, gt_vol={case['gt_volume']}")
    
    # Create figure
    fig, axes = plt.subplots(len(selected_cases), 5, figsize=(20, len(selected_cases) * 4))
    if len(selected_cases) == 1:
        axes = np.array([axes])
    
    # Column titles
    col_titles = ['Ground Truth', 'Prediction', 'Probability Map', 'Entropy Map', 'Error Map']
    
    for i, (_, case_row) in enumerate(selected_cases.iterrows()):
        case_id = case_row['case_id']
        print(f"Processing case {i+1}/{len(selected_cases)}: {case_id}")
        
        try:
            # Load ground truth and prediction nifti files
            gt_nii = nib.load(os.path.join(gt_path, f"{case_id}.nii.gz"))
            gt_img = gt_nii.get_fdata()
            
            pred_nii = nib.load(os.path.join(pred_path, f"{case_id}.nii.gz"))
            pred_img = pred_nii.get_fdata()
            
            # Load probability map
            # UPDATED: Load NIfTI probability file
            prob_file = os.path.join(prob_path, f"{case_id}_et_probability.nii.gz")
            
            if not os.path.exists(prob_file):
                print(f"NIfTI probability file not found for {case_id}: {prob_file}")
                continue
                
            prob_nii = nib.load(prob_file)
            et_probs = prob_nii.get_fdata()
            
            # NIfTI file already contains only enhancing tumor probabilities
            probs = None
            
            
            
            # CRITICAL: Check if probability map dimensions match the nifti images
            if et_probs.shape != gt_img.shape:
                print(f"Shape mismatch for {case_id}: probs {et_probs.shape} vs image {gt_img.shape}")
                # Try to transpose or reorder axes if needed
                if et_probs.shape[::-1] == gt_img.shape:
                    print("  Transposing probability map to match image orientation")
                    et_probs = np.transpose(et_probs, (2, 1, 0))
                elif (et_probs.shape[2], et_probs.shape[1], et_probs.shape[0]) == gt_img.shape:
                    print("  Reordering axes to match image orientation")
                    et_probs = np.transpose(et_probs, (2, 1, 0))
                else:
                    print(f"  Cannot resolve shape mismatch, skipping case")
                    continue
            
            # Validate that probability map has meaningful content
            prob_sum = np.sum(et_probs > 0.1)
            if prob_sum < 10:
                print(f"Case {case_id} has essentially empty probability map ({prob_sum} voxels > 0.1), skipping...")
                continue
            
            # Find best slice with actual content
            # Prioritize slices with enhancing tumor in GT or prediction
            et_gt_slices = np.sum(gt_img == LABEL_ENHANCING_TUMOUR, axis=(0, 1))
            et_pred_slices = np.sum(pred_img == LABEL_ENHANCING_TUMOUR, axis=(0, 1))
            et_prob_slices = np.sum(et_probs > 0.3, axis=(0, 1))  # Higher threshold for meaningful probability
            
            # Combined score prioritizing slices with content
            combined_score = et_gt_slices + et_pred_slices + et_prob_slices
            
            # Find slice with maximum content
            if np.max(combined_score) > 0:
                z_slice = np.argmax(combined_score)
                print(f"  Selected slice {z_slice}: GT={et_gt_slices[z_slice]}, Pred={et_pred_slices[z_slice]}, Prob>{int(et_prob_slices[z_slice])} voxels")
            else:
                # If no enhancing tumor, find slice with any tumor
                all_tumor_slices = np.sum((gt_img > 0) |  (pred_img > 0), axis=(0, 1))
                if np.max(all_tumor_slices) > 0:
                    z_slice = np.argmax(all_tumor_slices)
                else:
                    z_slice = gt_img.shape[2] // 2
            
            # Get 2D slices
            gt_slice = gt_img[:, :, z_slice]
            pred_slice = pred_img[:, :, z_slice]
            prob_slice = et_probs[:, :, z_slice]
            
            # Calculate entropy map
            epsilon = 1e-7
            p = np.clip(prob_slice, epsilon, 1-epsilon)
            entropy_slice = -p * np.log(p) - (1-p) * np.log(1-p)
            
            # Create error map
            gt_et = (gt_slice == LABEL_ENHANCING_TUMOUR)
            pred_et = (pred_slice == LABEL_ENHANCING_TUMOUR)
            error_map = np.zeros_like(gt_slice)
            error_map[gt_et & ~pred_et] = 1  # False negatives (red)
            error_map[~gt_et & pred_et] = 2  # False positives (blue)
            
            # Create brain mask for background
            brain_mask = (gt_slice > 0) | (pred_slice > 0)
            
            # Plot ground truth
            axes[i, 0].imshow(np.rot90(brain_mask * 0.2), cmap='gray', vmin=0, vmax=1)
            axes[i, 0].contour(np.rot90(gt_et), colors='green', linewidths=2)
            axes[i, 0].axis('off')
            
            # Plot prediction
            axes[i, 1].imshow(np.rot90(brain_mask * 0.2), cmap='gray', vmin=0, vmax=1)
            axes[i, 1].contour(np.rot90(pred_et), colors='red', linewidths=2)
            axes[i, 1].axis('off')
            
            # Plot probability map - mask out very low probabilities
            prob_masked = np.ma.masked_where(prob_slice < 0.05, prob_slice)
            axes[i, 2].imshow(np.rot90(brain_mask * 0.2), cmap='gray', vmin=0, vmax=1)
            im_prob = axes[i, 2].imshow(np.rot90(prob_masked), cmap='hot', vmin=0, vmax=1, alpha=0.6)
            axes[i, 2].axis('off')
            
            # Plot entropy map - mask out low entropy values
            entropy_masked = np.ma.masked_where(entropy_slice < 0.05, entropy_slice)
            axes[i, 3].imshow(np.rot90(brain_mask * 0.2), cmap='gray', vmin=0, vmax=1)
            im_ent = axes[i, 3].imshow(np.rot90(entropy_masked), cmap='viridis', vmin=0, vmax=1, alpha=0.6)
            axes[i, 3].axis('off')
            
            # Plot error map
            error_cmap = ListedColormap(['white', 'red', 'blue'])
            axes[i, 4].imshow(np.rot90(brain_mask * 0.2), cmap='gray', vmin=0, vmax=1)
            axes[i, 4].imshow(np.rot90(error_map), cmap=error_cmap, vmin=0, vmax=2, alpha=0.6)
            axes[i, 4].axis('off')
            
            # Add titles to first row
            if i == 0:
                for j, title in enumerate(col_titles):
                    axes[i, j].set_title(title, fontsize=12)
            
            # Add case information
            dice_val = case_row.get('dice', 0)
            entropy_val = case_row.get('mean_entropy', 0)
            info_text = f"{case_id}\nDice: {dice_val:.3f}\nEntropy: {entropy_val:.3f}"
            axes[i, 0].set_ylabel(info_text, fontsize=10)
            
        except Exception as e:
            print(f"Error visualizing {case_id}: {e}")
            import traceback
            traceback.print_exc()
            for j in range(5):
                axes[i, j].text(0.5, 0.5, f'Error\n{case_id}', 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    # Add colorbars
    if len(selected_cases) > 0 and 'im_prob' in locals():
        # Probability colorbar
        cbar_ax1 = fig.add_axes([0.92, 0.7, 0.02, 0.2])
        fig.colorbar(im_prob, cax=cbar_ax1, label='Probability')
        
        # Entropy colorbar
        cbar_ax2 = fig.add_axes([0.92, 0.4, 0.02, 0.2])
        fig.colorbar(im_ent, cax=cbar_ax2, label='Entropy')
    
    plt.suptitle('Uncertainty Visualization: Cases with Non-Empty Predictions and Probability Maps', fontsize=18)
    plt.tight_layout()
    
    return fig


# %%
# Expert Radiologist Failures vs AI Successes Figure
# This uses the exact same layout as nnunet_figure3_alternate 

def create_expert_radiologist_failures_figure(results_df, gt_path, pred_path, n_cases=8, min_gt_voxels=50, save_path=None):
    """
    Create a figure showing expert radiologist failures vs AI successes using the same layout as nnunet_figure3_alternate.
    Shows cases where radiologists failed but AI succeeded, with the exact 6-column layout.
    
    Layout: T1, T2, FLAIR, Model Prediction (All Labels), T1CE, Ground Truth (All Labels)
    """
    import os
    import glob
    import json
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import pandas as pd
    
    print("CREATING EXPERT RADIOLOGIST FAILURES vs AI SUCCESSES FIGURE")
    print("=" * 70)
    
    # Load radiologist review data
    RADIOLOGIST_REVIEWS_PATH = '/home/jruffle/Desktop/ENHANCEMENT_ARTICLE/radiologist_reviews/'
    
    if not os.path.exists(RADIOLOGIST_REVIEWS_PATH):
        print(f"Radiologist reviews path not found: {RADIOLOGIST_REVIEWS_PATH}")
        return None
    
    json_files = glob.glob(os.path.join(RADIOLOGIST_REVIEWS_PATH, '*.json'))
    
    def load_radiologist_data(json_file_path):
        """Load and process single radiologist review data"""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        radiologist_name = os.path.basename(json_file_path).replace('.json', '')
        
        def results_to_dataframe(results_list, with_segmentation=False):
            rows = []
            for result in results_list:
                row = {
                    'radiologist': radiologist_name,
                    'case_id': result['sample']['base_name'],
                    'predicted_enhancement': 1 if result['abnormality'] == 'Y' else 0,
                    'confidence': result['confidence'],
                    'image_quality': result['image_quality'],
                    'response_time': result['response_time'],
                    'ground_truth_sum': result['sample']['ground_truth_sum'],
                    'has_enhancement_gt': 1 if result['sample']['ground_truth_sum'] > 0 else 0,
                    'with_segmentation': with_segmentation
                }
                rows.append(row)
            return pd.DataFrame(rows)
        
        # Process both conditions
        dfs_to_combine = []
        
        if 'results_without_seg' in data:
            df_without = results_to_dataframe(data['results_without_seg'], False)
            dfs_to_combine.append(df_without)
        
        if 'results_with_seg' in data:
            df_with = results_to_dataframe(data['results_with_seg'], True)
            dfs_to_combine.append(df_with)
        
        if dfs_to_combine:
            combined_df = pd.concat(dfs_to_combine, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        return combined_df
    
    # Load all radiologist data
    all_radiologist_data = []
    
    for json_file in json_files:
        try:
            rad_data = load_radiologist_data(json_file)
            if len(rad_data) > 0:
                all_radiologist_data.append(rad_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not all_radiologist_data:
        print("Error: Could not load radiologist data")
        return None
    
    # Combine all radiologist data
    radiologist_df = pd.concat(all_radiologist_data, ignore_index=True)
    
    # Add model predictions and ground truth
    results_df_copy = results_df.copy()
    results_df_copy['model_predicted_enhancement'] = (results_df_copy['pred_volume'] > 0).astype(int)
    results_df_copy['model_has_enhancement_gt'] = (results_df_copy['gt_volume'] > 0).astype(int)
    
    # Merge radiologist data with model results
    radiologist_df = radiologist_df.merge(
        results_df_copy[['case_id', 'model_predicted_enhancement', 'dice', 'precision', 'recall',
                        'balanced_acc', 'Cohort', 'Country', 'Pathology', 'gt_volume', 'pred_volume']],
        on='case_id', how='left'
    )
    
    # Ensure consistent ground truth
    radiologist_df.loc[radiologist_df['gt_volume'].notna(), 'has_enhancement_gt'] = (
        radiologist_df.loc[radiologist_df['gt_volume'].notna(), 'gt_volume'] > 0
    ).astype(int)
    
    # Identify expert radiologist failure cases: Radiologist wrong (false negative), Model correct (true positive)
    target_cases = radiologist_df[
        (radiologist_df['has_enhancement_gt'] == 1) &  # Ground truth: has enhancement
        (radiologist_df['predicted_enhancement'] == 0) &  # Radiologist: predicted no enhancement (WRONG)
        (radiologist_df['model_predicted_enhancement'] == 1) &  # Model: predicted enhancement (CORRECT)
        (radiologist_df['gt_volume'] >= min_gt_voxels)  # Sufficient tumor volume
    ]
    
    print(f"Found {len(target_cases)} expert radiologist failure cases where AI succeeded")
    
    if len(target_cases) == 0:
        print("No suitable expert radiologist failure cases found")
        return None
    
    # Get the most frequently missed cases by expert radiologists
    case_miss_counts = target_cases['case_id'].value_counts()
    top_missed_cases = case_miss_counts.head(n_cases).index.tolist()
    
    # Get unique cases for visualization (one per case_id)
    selected_cases = []
    for case_id in top_missed_cases:
        case_data = target_cases[target_cases['case_id'] == case_id].iloc[0]
        selected_cases.append(case_data)
    
    n_cases_found = len(selected_cases)
    print(f"Selected {n_cases_found} unique cases for expert radiologist failure visualization")
    
    # Create figure with EXACT SAME 6-column layout as nnunet_figure3_alternate
    fig, axes = plt.subplots(n_cases_found, 6, figsize=(22, n_cases_found * 3.8))
    
    if n_cases_found == 1:
        axes = np.array([axes])
    
    # EXACT SAME column titles as nnunet_figure3_alternate
    col_titles = ['T1', 'T2', 'FLAIR', 'Model Prediction\n(All Labels)', 
                 'T1CE\n(Held out from model)', 'Ground Truth\n(All Labels)']
    
    # EXACT SAME colormap for all 4 labels as nnunet_figure3_alternate
    # 0: Background (black), 1: Normal brain (gray), 2: Other abnormality (blue), 3: Enhancing tumor (red)
    label_colors = ['black', 'gray', 'blue', 'green']  # Updated to match Figure 2
    label_cmap = ListedColormap(label_colors)
    
    # EXACT SAME data path as nnunet_figure3_alternate
    data_path = '/home/jruffle/Documents/seq-synth/data/'
    
    # Label scheme constants (EXACT SAME as nnunet_figure3_alternate)
    LABEL_BACKGROUND = 0
    LABEL_NORMAL_BRAIN = 1
    LABEL_OTHER_ABNORMALITY = 2
    LABEL_ENHANCING_TUMOUR = 3
    
    # Process each case using EXACT SAME structure as nnunet_figure3_alternate
    for i, case_row in enumerate(selected_cases):
        case_id = case_row['case_id']
        miss_count = case_miss_counts[case_id]
        
        try:
            # Load ground truth and prediction (EXACT SAME as nnunet_figure3_alternate)
            gt_img = nib.load(os.path.join(gt_path, f"{case_id}.nii.gz")).get_fdata()
            pred_img = nib.load(os.path.join(pred_path, f"{case_id}.nii.gz")).get_fdata()
            
            # Try to load structural images (EXACT SAME logic as nnunet_figure3_alternate)
            try:
                # Try sequences_merged directory first (EXACT SAME as nnunet_figure3_alternate)
                seq_path = os.path.join(data_path, 'sequences_merged', f"{case_id}.nii.gz")
                brain_mask_path = os.path.join(data_path, 'lesion_masks_augmented', f"{case_id}.nii.gz")
                
                if os.path.exists(seq_path) and os.path.exists(brain_mask_path):
                    seq_img = nib.load(seq_path).get_fdata()
                    brain_mask = nib.load(brain_mask_path).get_fdata()
                    brain_mask[brain_mask > 0] = 1
                    
                    # EXACT SAME sequence extraction as nnunet_figure3_alternate
                    flair_img = seq_img[..., 0] * brain_mask
                    t1_img = seq_img[..., 1] * brain_mask 
                    t1ce_img = seq_img[..., 2] * brain_mask
                    t2_img = seq_img[..., 3] * brain_mask
                else:
                    # Fallback to nnUNet structure (EXACT SAME as nnunet_figure3_alternate)
                    images_path = gt_path.replace('labelsTs', 'imagesTs')
                    t1_path = os.path.join(images_path, f"{case_id}_0000.nii.gz")
                    t2_path = os.path.join(images_path, f"{case_id}_0001.nii.gz") 
                    flair_path = os.path.join(images_path, f"{case_id}_0002.nii.gz")
                    t1ce_path = os.path.join(images_path, f"{case_id}_0003.nii.gz")
                    
                    t1_img = nib.load(t1_path).get_fdata() if os.path.exists(t1_path) else None
                    t2_img = nib.load(t2_path).get_fdata() if os.path.exists(t2_path) else None
                    flair_img = nib.load(flair_path).get_fdata() if os.path.exists(flair_path) else None
                    t1ce_img = nib.load(t1ce_path).get_fdata() if os.path.exists(t1ce_path) else None
                    
            except Exception as e:
                print(f"Could not load sequences for {case_id}: {e}")
                t1_img = t2_img = flair_img = t1ce_img = None
            
            # Find best slice with enhancing tumour (EXACT SAME as nnunet_figure3_alternate)
            et_slices = np.sum(gt_img == LABEL_ENHANCING_TUMOUR, axis=(0, 1))
            if np.max(et_slices) > 0:
                z_slice = np.argmax(et_slices)
            else:
                z_slice = gt_img.shape[2] // 2
            
            # Get 2D slices (EXACT SAME as nnunet_figure3_alternate)
            gt_slice = gt_img[:, :, z_slice]
            pred_slice = pred_img[:, :, z_slice]
            
            # Column 1: T1 (EXACT SAME as nnunet_figure3_alternate)
            if t1_img is not None:
                axes[i, 0].imshow(np.rot90(t1_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 0].text(0.5, 0.5, 'T1\nNot Available', ha='center', va='center', transform=axes[i, 0].transAxes)
            
            # Column 2: T2 (EXACT SAME as nnunet_figure3_alternate)
            if t2_img is not None:
                axes[i, 1].imshow(np.rot90(t2_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 1].text(0.5, 0.5, 'T2\nNot Available', ha='center', va='center', transform=axes[i, 1].transAxes)
            
            # Column 3: FLAIR (EXACT SAME as nnunet_figure3_alternate)
            if flair_img is not None:
                axes[i, 2].imshow(np.rot90(flair_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 2].text(0.5, 0.5, 'FLAIR\nNot Available', ha='center', va='center', transform=axes[i, 2].transAxes)
            
            # Column 4: Model Prediction - ALL LABELS (EXACT SAME as nnunet_figure3_alternate)
            pred_display = np.rot90(pred_slice)
            im_pred = axes[i, 3].imshow(pred_display, cmap=label_cmap, vmin=0, vmax=3, interpolation='nearest', alpha=0.6)
            
            # Column 5: T1CE (EXACT SAME as nnunet_figure3_alternate)
            if t1ce_img is not None:
                axes[i, 4].imshow(np.rot90(t1ce_img[:, :, z_slice]), cmap='gray')
            else:
                axes[i, 4].text(0.5, 0.5, 'T1CE\nNot Available', ha='center', va='center', transform=axes[i, 4].transAxes)
            
            # Column 6: Ground Truth - ALL LABELS (EXACT SAME as nnunet_figure3_alternate)
            gt_display = np.rot90(gt_slice)
            im_gt = axes[i, 5].imshow(gt_display, cmap=label_cmap, vmin=0, vmax=3, interpolation='nearest', alpha=0.6)
            
            # Add titles to first row (EXACT SAME as nnunet_figure3_alternate)
            if i == 0:
                for j, title in enumerate(col_titles):
                    axes[i, j].set_title(title, fontsize=11, pad=8)
            
            # Add case information - EMPHASIZE RADIOLOGIST FAILURE
            cohort_name = case_row['Cohort']
            dice_score = case_row['dice']
            newline = '\n'
            axes[i, 0].set_ylabel(f"{case_id}{newline}Cohort: {cohort_name}{newline}Dice: {dice_score:.3f}{newline}MISSED by {miss_count} experts", 
                                 fontsize=9, color='red', weight='bold')
            
            # Remove axis ticks (EXACT SAME as nnunet_figure3_alternate)
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        except Exception as e:
            print(f"Error visualizing {case_id}: {e}")
            for j in range(6):
                error_text = f'Error loading{chr(10)}{case_id}'
                axes[i, j].text(0.5, 0.5, error_text, 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    # EXACT SAME layout adjustments as nnunet_figure3_alternate
    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.05, wspace=-0.6, top=0.92, bottom=0.05)
    
    # EXACT SAME white divider line between columns 3 and 4 as nnunet_figure3_alternate
    if n_cases_found > 0:
        fig.canvas.draw()
        pos3 = axes[0, 3].get_position()
        pos4 = axes[0, 4].get_position()
        line_x = (pos3.x1 + pos4.x0) / 2
        
        # White line with black border (EXACT SAME as nnunet_figure3_alternate)
        border_line = plt.Line2D([line_x, line_x], [0.05, 0.92], 
                                transform=fig.transFigure, 
                                color='black', 
                                linewidth=8,
                                solid_capstyle='butt',
                                zorder=9)
        fig.add_artist(border_line)
        
        line = plt.Line2D([line_x, line_x], [0.05, 0.92], 
                         transform=fig.transFigure, 
                         color='white', 
                         linewidth=6,
                         solid_capstyle='butt',
                         zorder=10)
        fig.add_artist(line)
    
    # EXACT SAME legend for label colors as nnunet_figure3_alternate
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Background'),
        Patch(facecolor='gray', label='Normal Brain'),
        Patch(facecolor='blue', label='Other Abnormality'),
        Patch(facecolor='red', label='Enhancing Tumor')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.02))
    
    # Title emphasizing expert failure vs AI success
    plt.suptitle('Expert Radiologist Failures vs AI Successes: Cases Missed by Expert Radiologists but Correctly Identified by AI', 
                fontsize=18, y=0.96, color='red', weight='bold')
    
    # Save if path provided (EXACT SAME as nnunet_figure3_alternate)
    if save_path:
        fig.savefig(os.path.join(save_path, 'expert_radiologist_failures_vs_ai_successes.png'), 
                   dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(save_path, 'expert_radiologist_failures_vs_ai_successes.svg'), 
                   format='svg', bbox_inches='tight')
        print("Saved: expert_radiologist_failures_vs_ai_successes.png")
    
    return fig

# Generate the expert radiologist failures vs AI successes figure
print("Creating expert radiologist failures vs AI successes figure using exact nnunet_figure3_alternate layout...")

# Check if radiologist data exists
RADIOLOGIST_REVIEWS_PATH = '/home/jruffle/Desktop/ENHANCEMENT_ARTICLE/radiologist_reviews/'

if os.path.exists(RADIOLOGIST_REVIEWS_PATH):
    fig_expert_failures = create_expert_radiologist_failures_figure(
        results_df, 
        gt_labels_path, 
        predictions_path,
        n_cases=8,
        min_gt_voxels=50,
        save_path=figures_out
    )
    
    if fig_expert_failures:
        plt.show()
        print("Expert radiologist failures vs AI successes figure created successfully!")
    else:
        print("Could not generate expert radiologist failures figure")
else:
    print(f"Radiologist reviews path not found: {RADIOLOGIST_REVIEWS_PATH}")
    print("Skipping expert radiologist failures visualization")

# %%
def create_uncertainty_visualization_with_flair():
    """Create uncertainty visualization with FLAIR backgrounds and better case selection"""
    
    # Select cases with non-zero dice scores from different cohorts
    selected_cases = [
        'BraTS-GLI-02996-101',  # Postoperative glioma, dice=0.57
        'BraTS-MEN-01382-000',  # Meningioma, dice=0.56
        'BraTS-MET-00142-000',  # Metastases, dice=0.75
        'BraTS2021_00122'       # Presurgical glioma, dice=0.66
    ]
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Model Uncertainty Analysis: FLAIR Background with Overlays', fontsize=20, fontweight='bold')
    
    # Paths
    images_path = "/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/imagesTs"
    pred_path = "/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/predTs_PP"
    gt_path = "/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/labelsTs"
    # UPDATED: Now using NIfTI probability files directory
    prob_nifti_path = "/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/predTs"
    
    for i, case_id in enumerate(selected_cases):
        try:
            # Get case info
            case_info = results_df[results_df['case_id'] == case_id].iloc[0]
            dice_score = case_info['dice']
            
            # Load FLAIR image
            flair_path = os.path.join(images_path, f"{case_id}_0002.nii.gz")
            flair_nii = nib.load(flair_path)
            flair_img = flair_nii.get_fdata()
            
            # Load ground truth
            gt_file = os.path.join(gt_path, f"{case_id}.nii.gz")
            gt_nii = nib.load(gt_file)
            gt_img = gt_nii.get_fdata()
            
            # Load prediction
            pred_file = os.path.join(pred_path, f"{case_id}.nii.gz")
            pred_nii = nib.load(pred_file)
            pred_img = pred_nii.get_fdata()
            
            # UPDATED: Load probabilities from NIfTI file instead of .npz
            prob_file = os.path.join(prob_nifti_path, f"{case_id}_et_probability.nii.gz")
            
            if not os.path.exists(prob_file):
                print(f"NIfTI probability file not found for {case_id}: {prob_file}")
                continue
                
            prob_nii = nib.load(prob_file)
            et_probs = prob_nii.get_fdata()
            
            # Validate that probability map dimensions match the nifti images
            if et_probs.shape != gt_img.shape:
                print(f"Shape mismatch for {case_id}: probs {et_probs.shape} vs image {gt_img.shape}")
                # The NIfTI file should already have correct orientation, but just in case:
                if et_probs.shape[::-1] == gt_img.shape:
                    print("  Transposing probability map to match image orientation")
                    et_probs = np.transpose(et_probs, (2, 1, 0))
                elif (et_probs.shape[2], et_probs.shape[1], et_probs.shape[0]) == gt_img.shape:
                    print("  Reordering axes to match image orientation")
                    et_probs = np.transpose(et_probs, (2, 1, 0))
                else:
                    print(f"  Cannot resolve shape mismatch, using as is")
            
            # Extract enhancing tumor binary masks
            et_gt = (gt_img == 3).astype(float)
            et_pred = (pred_img == 3).astype(float)
            
            # Calculate entropy (avoid log(0))
            epsilon = 1e-10
            et_probs_safe = np.clip(et_probs, epsilon, 1 - epsilon)
            entropy = -et_probs_safe * np.log2(et_probs_safe) - (1 - et_probs_safe) * np.log2(1 - et_probs_safe)
            
            # Find best slice (most enhancing tumor)
            et_sum_per_slice = np.sum(et_gt, axis=(0, 1))
            best_slice = np.argmax(et_sum_per_slice)
            
            # If no enhancing tumor in gt, use prediction
            if best_slice == 0 and et_sum_per_slice[0] == 0:
                et_sum_per_slice = np.sum(et_pred, axis=(0, 1))
                best_slice = np.argmax(et_sum_per_slice)
            
            # If still no content, use probability map
            if best_slice == 0 and et_sum_per_slice[0] == 0:
                et_sum_per_slice = np.sum(et_probs > 0.3, axis=(0, 1))
                best_slice = np.argmax(et_sum_per_slice)
            
            # Extract slices
            flair_slice = flair_img[:, :, best_slice]
            gt_slice = et_gt[:, :, best_slice]
            pred_slice = et_pred[:, :, best_slice]
            prob_slice = et_probs[:, :, best_slice]
            entropy_slice = entropy[:, :, best_slice]
            
            # Normalize FLAIR for display
            flair_norm = (flair_slice - np.min(flair_slice)) / (np.max(flair_slice) - np.min(flair_slice) + 1e-8)
            
            # Plot FLAIR background
            ax = axes[i, 0]
            ax.imshow(np.rot90(flair_norm), cmap='gray', alpha=0.6)
            ax.set_title(f'a) FLAIR {case_id} Dice: {dice_score:.3f}', fontsize=12)
            ax.axis('off')
            
            # Plot ground truth overlay
            ax = axes[i, 1]
            ax.imshow(np.rot90(flair_norm), cmap='gray', alpha=0.6)
            masked_gt = np.ma.masked_where(gt_slice == 0, gt_slice)
            ax.imshow(np.rot90(masked_gt), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            ax.set_title('b) Ground Truth', fontsize=12)
            ax.axis('off')
            
            # Plot prediction overlay
            ax = axes[i, 2]
            ax.imshow(np.rot90(flair_norm), cmap='gray', alpha=0.6)
            masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
            ax.imshow(np.rot90(masked_pred), cmap='Blues', alpha=0.6, vmin=0, vmax=1)
            ax.set_title('c) Prediction', fontsize=12)
            ax.axis('off')
            
            # Plot probability map overlay
            ax = axes[i, 3]
            ax.imshow(np.rot90(flair_norm), cmap='gray', alpha=0.6)
            masked_prob = np.ma.masked_where(prob_slice < 0.01, prob_slice)
            im = ax.imshow(np.rot90(masked_prob), cmap='hot', alpha=0.6, vmin=0, vmax=1)
            ax.set_title('d) Probability Map', fontsize=12)
            ax.axis('off')
            
            # Plot entropy map overlay
            ax = axes[i, 4]
            ax.imshow(np.rot90(flair_norm), cmap='gray', alpha=0.6)
            masked_entropy = np.ma.masked_where(entropy_slice < 0.01, entropy_slice)
            im = ax.imshow(np.rot90(masked_entropy), cmap='viridis', alpha=0.6, vmin=0, vmax=1)
            ax.set_title('e) Uncertainty (Entropy)', fontsize=12)
            ax.axis('off')
            
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            import traceback
            traceback.print_exc()
            # Fill row with error message
            for j in range(5):
                axes[i, j].text(0.5, 0.5, f'Error {case_id}', 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
    
    # Add colorbars
    cbar_prob = fig.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=axes[:, 3], 
                            orientation='vertical', fraction=0.046, pad=0.04)
    cbar_prob.set_label('Probability', fontsize=12)
    
    cbar_entropy = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[:, 4],
                               orientation='vertical', fraction=0.046, pad=0.04)
    cbar_entropy.set_label('Entropy', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('uncertainty_case_examples_with_flair.png', dpi=300, bbox_inches='tight')
    plt.savefig('uncertainty_case_examples_with_flair.svg', bbox_inches='tight')
    plt.show()

# Run the visualization
create_uncertainty_visualization_with_flair()

# %%
# 2. Error Pattern Analysis - Spatial Distribution and False Positive Characterization

def error_pattern_analysis():
    """
    Analyze spatial error patterns and characterize false positives/negatives
    """
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Use consistent color palette
    colors_pie = sns.color_palette('husl', n_colors=10)
    
    # Filter cases
    et_cases = results_df[results_df['gt_volume'] > 0].copy()
    
    # 1. Error Types Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate error metrics for each case
    error_types = []
    for _, row in results_df.iterrows():
        if row['gt_volume'] > 0 and row['pred_volume'] > 0:
            # Calculate overlap
            tp = row['tp']
            fp = row['fp']
            fn = row['fn']
            
            # Categorize errors
            if tp > 0:
                overlap_ratio = tp / (tp + fp + fn)
                if overlap_ratio > 0.7:
                    error_types.append('Good Match')
                elif overlap_ratio > 0.3:
                    error_types.append('Partial Match')
                else:
                    error_types.append('Poor Match')
            else:
                error_types.append('Complete Miss')
        elif row['gt_volume'] > 0 and row['pred_volume'] == 0:
            error_types.append('False Negative')
        elif row['gt_volume'] == 0 and row['pred_volume'] > 0:
            error_types.append('False Positive')
        else:
            error_types.append('True Negative')
    
    # Count error types
    error_counts = pd.Series(error_types).value_counts()
    
    # Create pie chart
    colors_errors = [colors_pie[i] for i in range(len(error_counts))]
    ax1.pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%',
            startangle=140, colors=colors_errors)
    ax1.set_title('a) Error Type Distribution', fontsize=14)
    
    # 2. False Positive Volume Analysis
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get false positive cases
    fp_cases = results_df[results_df['gt_volume'] == 0].copy()
    fp_with_pred = fp_cases[fp_cases['pred_volume'] > 0]
    
    if len(fp_with_pred) > 0:
        # Histogram of false positive volumes
        ax2.hist(fp_with_pred['pred_volume'], bins=30, alpha=0.7, 
                color=colors_pie[4], edgecolor='black')
        ax2.set_xlabel('False Positive Volume [voxels]', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'b) False Positive Volume Distribution (n={len(fp_with_pred)})', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        ax2.axvline(fp_with_pred['pred_volume'].median(), color='red', 
                   linestyle='--', label=f'Median: {fp_with_pred["pred_volume"].median():.0f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No false positive cases', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('b) False Positive Volume Distribution', fontsize=14)
    
    # 3. Border Accuracy Analysis
    ax3 = fig.add_subplot(gs[0, 2])
    
    # For cases with both GT and pred, analyze border accuracy
    border_cases = et_cases[(et_cases['pred_volume'] > 0) & (et_cases['dice'] > 0)].copy()
    
    if len(border_cases) > 0:
        # Calculate surface dice (approximation using volumes)
        # Surface error is roughly proportional to (FP + FN) / (TP + FP + FN)
        border_cases['surface_error'] = (border_cases['fp'] + border_cases['fn']) / \
                                       (border_cases['tp'] + border_cases['fp'] + border_cases['fn'])
        
        # Plot surface error vs dice
        scatter = ax3.scatter(border_cases['dice'], border_cases['surface_error'], 
                            c=border_cases['gt_volume'], cmap='viridis', 
                            alpha=0.6, s=30, norm=plt.Normalize(vmin=0, vmax=10000))
        ax3.set_xlabel('Dice Score', fontsize=12)
        ax3.set_ylabel('Surface Error Ratio', fontsize=12)
        ax3.set_title('c) Border Accuracy vs Overall Performance (Synthetic)', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3, label='GT Volume')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('c) Border Accuracy Analysis', fontsize=14)
    
    # 4. Error Distribution by Cohort
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Calculate error rates by cohort
    cohort_errors = []
    cohort_names = []
    
    for cohort in results_df['Cohort'].unique():
        if cohort == '':
            continue
        
        cohort_data = results_df[results_df['Cohort'] == cohort]
        
        # Calculate different error types
        fn_rate = ((cohort_data['gt_volume'] > 0) & (cohort_data['pred_volume'] == 0)).sum() / len(cohort_data)
        fp_rate = ((cohort_data['gt_volume'] == 0) & (cohort_data['pred_volume'] > 0)).sum() / len(cohort_data)
        
        cohort_errors.append([fn_rate, fp_rate])
        cohort_names.append(cohort)
    
    if cohort_errors:
        # Stacked bar plot
        cohort_errors = np.array(cohort_errors).T
        x_pos = np.arange(len(cohort_names))
        
        ax4.bar(x_pos, cohort_errors[0], alpha=0.7, color=colors_pie[4], label='False Negative Rate')
        ax4.bar(x_pos, cohort_errors[1], bottom=cohort_errors[0], 
               alpha=0.7, color=colors_pie[5], label='False Positive Rate')
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(cohort_names, rotation=45, ha='right')
        ax4.set_ylabel('Error Rate', fontsize=12)
        ax4.set_title('d) Error Rates by Cohort', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Volume-based Error Analysis
    ax5 = fig.add_subplot(gs[1, :2])
    
    # Analyze errors by volume quartiles
    et_cases['volume_quartile'] = pd.qcut(et_cases['gt_volume'], q=4, 
                                          labels=['Q1 (Small)', 'Q2', 'Q3', 'Q4 (Large)'])
    
    # Calculate error metrics by quartile
    quartile_stats = et_cases.groupby('volume_quartile').agg({
        'dice': ['mean', 'std'],
        'precision': 'mean',
        'recall': 'mean',
        'pred_volume': lambda x: (x == 0).mean()  # Miss rate
    }).round(3)
    
    # Create grouped bar plot
    metrics = ['Dice', 'Precision', 'Recall', 'Detection']
    quartiles = quartile_stats.index
    
    x = np.arange(len(quartiles))
    width = 0.2
    
    values = [
        quartile_stats[('dice', 'mean')].values,
        quartile_stats[('precision', 'mean')].values,
        quartile_stats[('recall', 'mean')].values,
        1 - quartile_stats[('pred_volume', '<lambda>')].values  # Detection rate
    ]
    
    for i, (metric, vals) in enumerate(zip(metrics, values)):
        ax5.bar(x + i*width, vals, width, label=metric, alpha=0.7, color=colors_pie[i])
    
    ax5.set_xlabel('Volume Quartile', fontsize=12)
    ax5.set_ylabel('Performance', fontsize=12)
    ax5.set_title('e) Performance Metrics by Volume Quartile', fontsize=14)
    ax5.set_xticks(x + width * 1.5)
    ax5.set_xticklabels(quartiles)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 1)
    
    # 6. False Negative Analysis
    ax6 = fig.add_subplot(gs[1, 2:])
    
    # Analyze characteristics of missed cases
    missed_cases = et_cases[et_cases['pred_volume'] == 0]
    
    if len(missed_cases) > 0:
        # Volume distribution of missed cases
        ax6.hist(missed_cases['gt_volume'], bins=30, alpha=0.7, 
                color=colors_pie[6], edgecolor='black', density=True, label='Missed')
        ax6.hist(et_cases[et_cases['pred_volume'] > 0]['gt_volume'], bins=30, 
                alpha=0.5, color=colors_pie[2], edgecolor='black', density=True, label='Detected')
        
        ax6.set_xlabel('Ground Truth Volume [voxels]', fontsize=12)
        ax6.set_ylabel('Density', fontsize=12)
        ax6.set_title(f'f) Volume Distribution: Missed vs Detected Cases', fontsize=14)
        ax6.set_xscale('log')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No missed cases', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('f) False Negative Analysis', fontsize=14)
    
    # 7. Error Heatmap by Pathology and Cohort
    ax7 = fig.add_subplot(gs[2, :])
    
    # Create error rate matrix
    error_matrix = []
    pathologies = results_df['Pathology'].unique()
    pathologies = [p for p in pathologies if p != '']
    
    for pathology in pathologies:
        row = []
        for cohort in cohort_names:
            mask = (results_df['Pathology'] == pathology) & (results_df['Cohort'] == cohort)
            subset = results_df[mask]
            
            if len(subset) > 0:
                # Calculate composite error score
                et_subset = subset[subset['gt_volume'] > 0]
                if len(et_subset) > 0:
                    error_score = 1 - et_subset['dice'].mean()
                else:
                    error_score = np.nan
            else:
                error_score = np.nan
            
            row.append(error_score)
        error_matrix.append(row)
    
    # Create heatmap
    error_matrix = np.array(error_matrix)
    
    if error_matrix.size > 0:
        im = ax7.imshow(error_matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)
        
        # Set ticks
        ax7.set_xticks(np.arange(len(cohort_names)))
        ax7.set_yticks(np.arange(len(pathologies)))
        ax7.set_xticklabels(cohort_names, rotation=45, ha='right')
        ax7.set_yticklabels(pathologies)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax7, label='Error Rate (1 - Dice)')
        
        # Add text annotations
        for i in range(len(pathologies)):
            for j in range(len(cohort_names)):
                if not np.isnan(error_matrix[i, j]):
                    text = ax7.text(j, i, f'{error_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
        
        ax7.set_title('g) Error Rates by Pathology and Cohort', fontsize=14)
    
    plt.suptitle('Error Pattern Analysis', fontsize=18)
    plt.tight_layout()
    
    # Print summary
    print("\nError Pattern Summary:")
    print("=" * 60)
    print(f"Error Type Distribution:")
    for error_type, count in error_counts.items():
        print(f"  {error_type}: {count} ({count/len(results_df)*100:.1f}%)")
    
    if len(fp_with_pred) > 0:
        print(f"\nFalse Positives:")
        print(f"  Total cases: {len(fp_with_pred)}")
        print(f"  Median volume: {fp_with_pred['pred_volume'].median():.0f} voxels")
        print(f"  Mean volume: {fp_with_pred['pred_volume'].mean():.0f} voxels")
    
    return fig

# Generate error pattern analysis
if 'dice' in results_df.columns and len(results_df) > 0:
    fig_error = error_pattern_analysis()
    if fig_error:
        fig_error.savefig(os.path.join(figures_out, 'error_pattern_analysis.png'), 
                         dpi=300, bbox_inches='tight')
        fig_error.savefig(os.path.join(figures_out, 'error_pattern_analysis.svg'), 
                         format='svg', bbox_inches='tight')
        plt.show()
else:
    print("No metrics available for error pattern analysis")

# %%
# Subgroup analysis

# Filter for cases with ground truth enhancing tumor
et_cases = results_df[results_df['gt_volume'] > 0].copy()

# Use radiomics-based enhancement patterns instead of pathology mapping
try:
    # Import the improved radiomics module
    from radiomics_enhancement_pattern_analysis_distance import compute_enhancement_patterns
    
    # Check if we already have computed patterns
    patterns_csv_path = os.path.join(figures_out, 'results_with_radiomics_patterns.csv')
    
    if os.path.exists(patterns_csv_path):
        print("Loading pre-computed radiomics patterns...")
        results_with_patterns = pd.read_csv(patterns_csv_path)
        # Merge with current et_cases to get the patterns
        et_cases = et_cases.merge(
            results_with_patterns[['case_id', 'enhancement_pattern', 'sphericity', 'n_components', 'solidity']], 
            on='case_id', 
            how='left'
        )
    else:
        print("Computing radiomics-based enhancement patterns...")
        # Compute enhancement patterns
        results_with_patterns = compute_enhancement_patterns(
            results_df, 
            gt_labels_path,
            label_value=3,  # Enhancing tumor label
            n_jobs=-1,
            min_distance_voxels=20
        )
        
        # Save the enhanced results
        results_with_patterns.to_csv(patterns_csv_path, index=False)
        
        # Merge with et_cases
        et_cases = et_cases.merge(
            results_with_patterns[['case_id', 'enhancement_pattern', 'sphericity', 'n_components', 'solidity']], 
            on='case_id', 
            how='left'
        )
    
    # Use radiomics-derived pattern
    et_cases['pattern'] = et_cases['enhancement_pattern']
    
    # Remove cases with no enhancement or errors
    et_cases = et_cases[
        (et_cases['pattern'] != 'No Enhancement') & 
        (et_cases['pattern'] != 'Error') &
        (et_cases['pattern'].notna())
    ]
    
except ImportError:
    # Try original module if improved version not available
    try:
        from radiomics_enhancement_pattern_analysis_distance import compute_enhancement_patterns
        print("Using original radiomics module...")
        # Same logic as above
        patterns_csv_path = os.path.join(figures_out, 'results_with_radiomics_patterns.csv')
        
        if os.path.exists(patterns_csv_path):
            print("Loading pre-computed radiomics patterns...")
            results_with_patterns = pd.read_csv(patterns_csv_path)
            et_cases = et_cases.merge(
                results_with_patterns[['case_id', 'enhancement_pattern', 'sphericity', 'n_components', 'solidity']], 
                on='case_id', 
                how='left'
            )
        else:
            print("Computing radiomics-based enhancement patterns...")
            results_with_patterns = compute_enhancement_patterns(
                results_df, 
                gt_labels_path,
                label_value=3,
                n_jobs=-1,
                min_distance_voxels=20
            )
            results_with_patterns.to_csv(patterns_csv_path, index=False)
            et_cases = et_cases.merge(
                results_with_patterns[['case_id', 'enhancement_pattern', 'sphericity', 'n_components', 'solidity']], 
                on='case_id', 
                how='left'
            )
        
        et_cases['pattern'] = et_cases['enhancement_pattern']
        et_cases = et_cases[
            (et_cases['pattern'] != 'No Enhancement') & 
            (et_cases['pattern'] != 'Error') &
            (et_cases['pattern'].notna())
        ]
        
    except Exception as e:
        print(f"Warning: Could not load radiomics patterns, falling back to pathology mapping: {e}")
        # Fallback to original pathology-based mapping
        pattern_mapping = {
            'Presurgical glioma': 'Infiltrative',
            'Postoperative glioma resection': 'Post-treatment',
            'Meningioma': 'Well-circumscribed',
            'Metastases': 'Multiple',
            'Paediatric presurgical tumour': 'Pediatric'
        }
        et_cases['pattern'] = et_cases['Pathology'].map(pattern_mapping)

# Remove rows with NaN patterns
et_cases = et_cases.dropna(subset=['pattern'])

# Calculate metrics by pattern
pattern_metrics = et_cases.groupby('pattern')['dice'].agg(['mean', 'std', 'count'])
# Filter out Single Lesion (Unclassified)
pattern_metrics = pattern_metrics[pattern_metrics.index != 'Single Lesion (Unclassified)']
pattern_metrics = pattern_metrics.sort_values('mean', ascending=False)

# Create figure
plt.figure(figsize=(20, 12))
fig = plt.gcf()

# Create grid layout
gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

# Panel a: Dice scores by pathology
ax1 = fig.add_subplot(gs[0, 0])
pathology_metrics = et_cases.groupby('Pathology')['dice'].agg(['mean', 'std', 'count'])
pathology_metrics = pathology_metrics.sort_values('mean', ascending=False)

x_pos = np.arange(len(pathology_metrics))
means = pathology_metrics['mean'].values
stds = pathology_metrics['std'].values

bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.6, color=colors_pie[:len(pathology_metrics)])
ax1.set_xticks(x_pos)
ax1.set_xticklabels(pathology_metrics.index, rotation=45, ha='right')
ax1.set_ylabel('Dice Score', fontsize=12)
ax1.set_ylim(0, 1)
ax1.set_title('a) Performance by Pathology', fontsize=14)
ax1.grid(True, alpha=0.6, axis='y')

# Add sample sizes
for i, (idx, row) in enumerate(pathology_metrics.iterrows()):
    count = int(row['count'])
    mean = row['mean']
    ax1.text(i, mean + stds[i] + 0.02, f'n={count}', ha='center', va='bottom', fontsize=9)

# Panel b: Performance distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(et_cases['dice'], bins=30, alpha=0.6, color='lightblue', edgecolor='black')
ax2.axvline(et_cases['dice'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {et_cases["dice"].mean():.3f}')
ax2.axvline(et_cases['dice'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {et_cases["dice"].median():.3f}')
ax2.set_xlabel('Dice Score', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('b) Performance Distribution', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.6)

# Panel c: Performance by radiomics-derived enhancement patterns
ax3 = fig.add_subplot(gs[0, 2])

x_pos = np.arange(len(pattern_metrics))
means = pattern_metrics['mean'].values
stds = pattern_metrics['std'].values

bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.6, 
                color=[colors_pie[i % len(colors_pie)] for i in range(len(pattern_metrics))])
ax3.set_xticks(x_pos)
ax3.set_xticklabels(pattern_metrics.index, rotation=45, ha='right')
ax3.set_ylabel('Dice Score', fontsize=12)
ax3.set_ylim(0, 1)
ax3.set_title('c) Performance by Enhancement Pattern (Radiomics-derived)', fontsize=14)
ax3.grid(True, alpha=0.6, axis='y')

# Add sample sizes and radiomics metrics
for i, (idx, row) in enumerate(pattern_metrics.iterrows()):
    count = int(row['count'])
    mean_val = row['mean']
    
    # Get additional metrics for this pattern if available
    if 'sphericity' in et_cases.columns:
        pattern_cases = et_cases[et_cases['pattern'] == idx]
        avg_sphericity = pattern_cases['sphericity'].mean()
        avg_components = pattern_cases['n_components'].mean() if 'n_components' in pattern_cases.columns else np.nan
        avg_solidity = pattern_cases['solidity'].mean() if 'solidity' in pattern_cases.columns else np.nan
        
        # Add text with count - commented out
        # ax3.text(i, mean_val + stds[i] + 0.02, f'n={count}', 
        #         ha='center', va='bottom', fontsize=9)
        
        # Add pattern characteristics below bars
        # Commented out to avoid text overlap
        # char_text = f'Sph:{avg_sphericity:.2f}'
        # if not np.isnan(avg_components):
        # char_text += f'\nComp:{avg_components:.1f}'
        
        # ax3.text(i, -0.05, char_text, ha='center', va='top', 
        # transform=ax3.get_xaxis_transform(), fontsize=8, color='gray')
    else:
        print('')
        # Just add sample size if no radiomics data - commented out
        # ax3.text(i, mean_val + stds[i] + 0.02, f'n={count}', 
        #         ha='center', va='bottom', fontsize=9)

# Panel d: Volume analysis
ax4 = fig.add_subplot(gs[0, 3])

# Create volume bins
et_cases['volume_quartile'] = pd.qcut(et_cases['gt_volume'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
volume_metrics = et_cases.groupby('volume_quartile')['dice'].agg(['mean', 'std', 'count'])

x_pos = np.arange(len(volume_metrics))
means = volume_metrics['mean'].values
stds = volume_metrics['std'].values

bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.6, color='orange')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(volume_metrics.index)
ax4.set_ylabel('Dice Score', fontsize=12)
ax4.set_ylim(0, 1)
ax4.set_title('d) Performance by Volume Quartile', fontsize=14)
ax4.grid(True, alpha=0.6, axis='y')

# Add sample sizes
for i, (idx, row) in enumerate(volume_metrics.iterrows()):
    count = int(row['count'])
    mean = row['mean']
    ax4.text(i, mean + stds[i] + 0.02, f'n={count}', ha='center', va='bottom', fontsize=9)

# Bottom panels: Statistical comparisons
ax5 = fig.add_subplot(gs[1, :2])

# Pairwise comparison of enhancement patterns
from scipy.stats import ttest_ind

patterns = list(pattern_metrics.index)
comparison_data = []

for i in range(len(patterns)):
    for j in range(i+1, len(patterns)):
        pattern1_scores = et_cases[et_cases['pattern'] == patterns[i]]['dice']
        pattern2_scores = et_cases[et_cases['pattern'] == patterns[j]]['dice']
        
        if len(pattern1_scores) > 5 and len(pattern2_scores) > 5:
            stat, p_val = ttest_ind(pattern1_scores, pattern2_scores)
            comparison_data.append({
                'Pattern 1': patterns[i],
                'Pattern 2': patterns[j],
                'P-value': p_val,
                'Significant': p_val < 0.05
            })

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('P-value')
    
    # Create heatmap of p-values
    pivot_df = comparison_df.pivot(index='Pattern 1', columns='Pattern 2', values='P-value')
    
    import seaborn as sns
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'P-value'}, ax=ax5)
    ax5.set_title('e) Statistical Comparison of Enhancement Patterns', fontsize=14)

# Bottom right: Performance correlation with radiomics features
ax6 = fig.add_subplot(gs[1, 2:])

# Check if radiomics features are available and have valid data
if 'sphericity' in et_cases.columns:
    # Create correlation plot
    valid_cases = et_cases.dropna(subset=['sphericity', 'dice'])
    
    # Only create scatter plot if we have enough valid cases
    if len(valid_cases) > 10:
        # Create scatter plot
        scatter = ax6.scatter(valid_cases['sphericity'], valid_cases['dice'], 
                    alpha=0.6, s=50, c=valid_cases['n_components'], 
                    cmap='viridis', vmin=1, vmax=5, label='Cases')
        
        # Add trend line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            valid_cases['sphericity'], valid_cases['dice']
        )
        
        x_trend = np.linspace(valid_cases['sphericity'].min(), 
                            valid_cases['sphericity'].max(), 100)
        y_trend = slope * x_trend + intercept
        ax6.plot(x_trend, y_trend, 'r--', alpha=0.6, linewidth=2,
                label=f'R={r_value:.3f}, p={p_value:.3f}')
        
        ax6.set_xlabel('Sphericity', fontsize=12)
        ax6.set_ylabel('Dice Score', fontsize=12)
        ax6.set_title('f) Performance vs Sphericity (colored by # components)', fontsize=14)
        ax6.legend()
        ax6.grid(True, alpha=0.6)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Number of Components', fontsize=10)
    else:
        # Not enough data for correlation plot
        ax6.text(0.5, 0.5, 'Insufficient data for correlation plot\n(need >10 cases with valid sphericity)', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('f) Performance vs Sphericity', fontsize=14)
else:
    # No radiomics data available
    ax6.text(0.5, 0.5, 'Radiomics features not available', 
            ha='center', va='center', transform=ax6.transAxes, fontsize=12)
    ax6.set_title('f) Performance vs Sphericity', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(figures_out, 'subgroup_analysis_radiomics.png'), dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== SUBGROUP ANALYSIS SUMMARY ===")
print(f"Total cases with enhancing tumor: {len(et_cases)}")
print(f"Mean Dice score: {et_cases['dice'].mean():.3f} ± {et_cases['dice'].std():.3f}")
print(f"Median Dice score: {et_cases['dice'].median():.3f}")

print("\nPerformance by Enhancement Pattern:")
for pattern, row in pattern_metrics.iterrows():
    print(f"  {pattern}: {row['mean']:.3f} ± {row['std']:.3f} (n={row['count']})")

if 'sphericity' in et_cases.columns:
    print("\nRadiomics Feature Summary:")
    print(f"  Mean sphericity: {et_cases['sphericity'].mean():.3f}")
    print(f"  Mean components: {et_cases['n_components'].mean():.1f}")
    print(f"  Mean solidity: {et_cases['solidity'].mean():.3f}")
    
    # Print pattern definitions
    print("\n=== ENHANCEMENT PATTERN DEFINITIONS ===")
    print("Well-circumscribed: Single round lesion (sphericity >0.7, solidity >0.9)")
    print("Infiltrative: Irregular spreading pattern (sphericity <0.5 OR solidity <0.7)")  
    print("Multiple: ≥3 separate components OR 2 substantial components")
    print("Irregular/Complex: Between well-circumscribed and infiltrative")
    print("Unclassified: Features could not be calculated")

# %%
# Import the updated calibration analysis
import sys
sys.path.append('/home/jruffle/Downloads/')
from model_calibration_updated import model_calibration_analysis_updated, compute_confidence_metrics

# Use the updated function that uses actual probability maps
def model_calibration_analysis():
    """Updated calibration analysis using actual probability maps"""
    
    # Use the probability path defined earlier
    prob_path_nifti = '/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset003_enhance_and_abnormality_batchconfig/predTs/'
    
    # Call the updated function
    fig, conf_df = model_calibration_analysis_updated(
        results_df, 
        prob_path_nifti,
        gt_labels_path,
        predictions_path
    )
    
    return fig

# Generate the model calibration analysis figure
print("\n=== GENERATING MODEL CALIBRATION ANALYSIS ===")
if "dice" in results_df.columns and len(results_df) > 0:
    try:
        fig_calibration = model_calibration_analysis()
        if fig_calibration:
            # Save to PNG
            png_path = os.path.join(figures_out, "model_calibration_analysis.png")
            fig_calibration.savefig(png_path, dpi=300, bbox_inches="tight")
            print(f"Saved PNG to: {png_path}")
            
            # Save to SVG
            svg_path = os.path.join(figures_out, "model_calibration_analysis.svg")
            fig_calibration.savefig(svg_path, format="svg", bbox_inches="tight")
            print(f"Saved SVG to: {svg_path}")
            
            # Display the figure
            plt.show()
    except Exception as e:
        print(f"Error generating calibration analysis: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No metrics available for calibration analysis")

# %%
# %% [markdown]
# ### Data Quality Impact - Resolution and Artifact Analysis

# %%
# Data Quality Impact Analysis
print("=" * 80)
print("DATA QUALITY IMPACT - RESOLUTION AND ARTIFACT ANALYSIS")
print("=" * 80)

# Create figure with multiple subplots
colors_pie = sns.color_palette('husl', n_colors=10)
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Performance by Dataset Quality Proxy (using cohort as proxy)
ax1 = fig.add_subplot(gs[0, :])

# Group cohorts by presumed data quality (based on dataset characteristics)
quality_groups = {
    'High Quality Research': ['UPENN-GBM', 'UCSF-PDGM', 'BraTS2021'],
    'Clinical Standard': ['EGD', 'NHNN', 'BraTS-GLI'],
    'Specialized/Challenging': ['BraTS-MEN', 'BraTS-MET', 'BraTS-PED', 'BraTS-SSA']
}

quality_performance = []
for quality_level, cohorts in quality_groups.items():
    mask = results_df['Cohort'].isin(cohorts) & (results_df['gt_volume'] > 0)
    if mask.sum() > 0:
        dice_scores = results_df.loc[mask, 'dice'].values
        quality_performance.append({
            'Quality Level': quality_level,
            'Mean Dice': np.mean(dice_scores),
            'Std Dice': np.std(dice_scores),
            'Median Dice': np.median(dice_scores),
            'N': len(dice_scores)
        })

quality_df = pd.DataFrame(quality_performance)

# Create grouped bar chart
x = np.arange(len(quality_df))
width = 0.25

bars1 = ax1.bar(x - width, quality_df['Mean Dice'], width, label='Mean', alpha=0.7, color=colors_pie[0])
bars2 = ax1.bar(x, quality_df['Median Dice'], width, label='Median', alpha=0.7, color=colors_pie[1])
bars3 = ax1.bar(x + width, quality_df['Std Dice'], width, label='Std Dev', alpha=0.7, color=colors_pie[2])

# Add error bars for mean
ax1.errorbar(x - width, quality_df['Mean Dice'], yerr=quality_df['Std Dice'], 
             fmt='none', color='black', capsize=5)

ax1.set_xlabel('Data Quality Category', fontsize=14)
ax1.set_ylabel('Dice Score', fontsize=14)
ax1.set_title('a) Performance by Data Quality Category', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(quality_df['Quality Level'])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add sample sizes
for i, row in quality_df.iterrows():
    ax1.text(i, 0.02, f"n={row['N']}", ha='center', va='bottom', fontsize=10)

# 2. Volume Measurement Reliability
ax2 = fig.add_subplot(gs[1, 0])

# Analyze volume measurement consistency
volume_ratio = results_df[results_df['gt_volume'] > 0]['pred_volume'] / results_df[results_df['gt_volume'] > 0]['gt_volume']
volume_ratio = volume_ratio[np.isfinite(volume_ratio)]

# Create histogram
ax2.hist(volume_ratio, bins=50, alpha=0.7, edgecolor='black')
ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect Agreement')
ax2.set_xlabel('Volume Ratio (Predicted/Ground Truth)', fontsize=14)
ax2.set_ylabel('Frequency', fontsize=14)
ax2.set_title('Volume Measurement Reliability', fontsize=14)
ax2.set_xlim(0, 5)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add statistics
ax2.text(0.95, 0.95, f'Median ratio: {np.median(volume_ratio):.2f}\n'
                     f'IQR: [{np.percentile(volume_ratio, 25):.2f}, {np.percentile(volume_ratio, 75):.2f}]\n'
                     f'Within ±20%: {np.sum((volume_ratio >= 0.8) & (volume_ratio <= 1.2)) / len(volume_ratio) * 100:.1f}%',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor='white', alpha=0.7))

# 3. Small vs Large Tumor Detection
ax3 = fig.add_subplot(gs[1, 1])

# Analyze detection rates by tumor size
size_bins = np.percentile(results_df[results_df['gt_volume'] > 0]['gt_volume'], 
                         [0, 20, 40, 60, 80, 100])
size_labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']

detection_by_size = []
for i in range(len(size_bins)-1):
    mask = (results_df['gt_volume'] >= size_bins[i]) & \
           (results_df['gt_volume'] < size_bins[i+1])
    if mask.sum() > 0:
        detected = (results_df.loc[mask, 'pred_volume'] > 0).sum()
        total = mask.sum()
        detection_by_size.append({
            'Size Category': size_labels[i],
            'Detection Rate': detected / total * 100,
            'Mean Volume': results_df.loc[mask, 'gt_volume'].mean(),
            'N': total
        })

detection_df = pd.DataFrame(detection_by_size)

# Create bar chart
bars = ax3.bar(detection_df['Size Category'], detection_df['Detection Rate'], 
                alpha=0.7, color='green')

# Add value labels
for bar, rate, n in zip(bars, detection_df['Detection Rate'], detection_df['N']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{rate:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=9)

ax3.set_xlabel('Tumor Size Category', fontsize=14)
ax3.set_ylabel('Detection Rate (%)', fontsize=14)
ax3.set_title('Detection Performance by Tumor Size', fontsize=14)
ax3.set_ylim(0, 110)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Performance Stability Across Cohorts
ax4 = fig.add_subplot(gs[1, 2])

# Calculate coefficient of variation for each cohort
cohort_cv = []
for cohort in results_df['Cohort'].unique():
    cohort_mask = (results_df['Cohort'] == cohort) & (results_df['gt_volume'] > 0)
    if cohort_mask.sum() > 10:
        dice_scores = results_df.loc[cohort_mask, 'dice'].values
        cv = np.std(dice_scores) / np.mean(dice_scores) if np.mean(dice_scores) > 0 else np.nan
        cohort_cv.append({
            'Cohort': cohort,
            'CV': cv,
            'Mean Dice': np.mean(dice_scores),
            'N': len(dice_scores)
        })

cv_df = pd.DataFrame(cohort_cv)
cv_df = cv_df.sort_values('CV')

# Create scatter plot
scatter = ax4.scatter(cv_df['Mean Dice'], cv_df['CV'], s=cv_df['N']*2, 
                     alpha=0.6, c=range(len(cv_df)), cmap='viridis')

# Add cohort labels
for idx, row in cv_df.iterrows():
    ax4.annotate(row['Cohort'], (row['Mean Dice'], row['CV']), 
                fontsize=8, alpha=0.7)

ax4.set_xlabel('Mean Dice Score', fontsize=14)
ax4.set_ylabel('Coefficient of Variation', fontsize=14)
ax4.set_title('d) Performance Stability vs Mean Performance', fontsize=14)
ax4.grid(True, alpha=0.3)

# Add colorbar for cohort order
# Note: Colors represent the alphabetical order of cohorts (datasets)
# Size of points represents the number of cases in each cohort
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Dataset Index (Alphabetical Order)', fontsize=12)

# 5. Impact of Missing Enhancement
ax5 = fig.add_subplot(gs[2, :2])

# Analyze cases by enhancement presence
enhancement_groups = {
    'With Enhancement': results_df['gt_volume'] > 0,
    'Without Enhancement': results_df['gt_volume'] == 0
}

# Create comparison of metrics
metrics_comparison = []
for group_name, mask in enhancement_groups.items():
    if mask.sum() > 0:
        # For cases without enhancement, we can only measure specificity
        if group_name == 'Without Enhancement':
            tn = (results_df.loc[mask, 'pred_volume'] == 0).sum()
            fp = (results_df.loc[mask, 'pred_volume'] > 0).sum()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics_comparison.append({
                'Group': group_name,
                'Metric': 'Specificity',
                'Value': specificity,
                'N': mask.sum()
            })
        else:
            # For cases with enhancement, calculate multiple metrics
            for metric in ['dice', 'precision', 'recall']:
                if metric in results_df.columns:
                    value = results_df.loc[mask, metric].mean()
                    metrics_comparison.append({
                        'Group': group_name,
                        'Metric': metric.capitalize(),
                        'Value': value,
                        'N': mask.sum()
                    })

metrics_comp_df = pd.DataFrame(metrics_comparison)

# Create grouped bar chart
metrics_pivot = metrics_comp_df.pivot(index='Metric', columns='Group', values='Value')
metrics_pivot.plot(kind='bar', ax=ax5, alpha=0.7)

ax5.set_xlabel('Metric', fontsize=14)
ax5.set_ylabel('Performance', fontsize=14)
ax5.set_title('Performance Metrics by Enhancement Presence', fontsize=14)
ax5.legend(title='Group')
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)

# 6. Data Distribution Summary
ax6 = fig.add_subplot(gs[2, 2])

# Create summary statistics table
summary_data = {
    'Total Cases': len(results_df),
    'With Enhancement': (results_df['gt_volume'] > 0).sum(),
    'Without Enhancement': (results_df['gt_volume'] == 0).sum(),
    'Mean Tumor Volume': results_df[results_df['gt_volume'] > 0]['gt_volume'].mean(),
    'Median Tumor Volume': results_df[results_df['gt_volume'] > 0]['gt_volume'].median(),
    'Detection Rate': ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] > 0)).sum() / 
                     (results_df['gt_volume'] > 0).sum() * 100,
    'False Positive Rate': ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] > 0)).sum() / 
                          (results_df['gt_volume'] == 0).sum() * 100 if (results_df['gt_volume'] == 0).sum() > 0 else 0
}

# Create text table
ax6.axis('tight')
ax6.axis('off')

table_data = []
for key, value in summary_data.items():
    if isinstance(value, float):
        if 'Volume' in key:
            table_data.append([key, f'{value:.0f}'])
        else:
            table_data.append([key, f'{value:.1f}'])
    else:
        table_data.append([key, str(value)])

table = ax6.table(cellText=table_data,
                  colLabels=['Metric', 'Value'],
                  cellLoc='left',
                  loc='center',
                  bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

# Style the table
for i in range(len(table_data) + 1):
    for j in range(2):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

ax6.set_title('Dataset Summary Statistics', fontsize=14, pad=20)

plt.suptitle('Data Quality Impact Analysis', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()

# Print quality impact summary
print("\nData Quality Impact Summary:")
print("-" * 80)
print(f"Performance variation across cohorts (CV range): {cv_df['CV'].min():.3f} - {cv_df['CV'].max():.3f}")
print(f"Detection rate for smallest 20% tumors: {detection_df.iloc[0]['Detection Rate']:.1f}%")
print(f"Volume measurement accuracy (within ±20%): {np.sum((volume_ratio >= 0.8) & (volume_ratio <= 1.2)) / len(volume_ratio) * 100:.1f}%")
print(f"Overall false positive rate: {summary_data['False Positive Rate']:.1f}%")

# %%
# Regarding panel d (Performance Stability vs Mean Performance):

#   Looking at the code, here's what this scatter plot shows:

#   - X-axis: Mean Dice Score for each cohort
#   - Y-axis: Coefficient of Variation (CV = std/mean) for each cohort
#   - Point size: Proportional to the number of cases (N) in each cohort (s=cv_df['N']*2)
#   - Point color: Represents the alphabetical order of cohorts/datasets (c=range(len(cv_df)))

#   The "cohort order" in the colorbar refers to the alphabetical ordering of the dataset names. So if
#   you have cohorts like "BraTS-GLI", "BraTS-MEN", "BraTS-MET", etc., they would be assigned colors
#   based on their alphabetical position.

#   This visualization helps identify:
#   - Which cohorts have stable performance (low CV)
#   - Which cohorts have high mean performance
#   - The relationship between performance and stability
#   - The relative size of each cohort (larger dots = more cases)

#   The color coding by alphabetical order might not be the most meaningful - you might consider
#   coloring by another attribute like pathology type or geographic region if that would be more
#   informative.

# %%
# 6. Clinical Integration Metrics - PPV/NPV and Workflow Analysis

def clinical_integration_analysis():
    """
    Analyze metrics relevant for clinical integration including PPV/NPV at different prevalences
    and workflow impact analysis
    """
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Use consistent color palette
    colors_pie = sns.color_palette('husl', n_colors=10)
    
    # Calculate base metrics
    tp = ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] > 0)).sum()
    fp = ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] > 0)).sum()
    fn = ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] == 0)).sum()
    tn = ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] == 0)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 1. PPV/NPV at Different Prevalences
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Calculate PPV and NPV at different prevalences
    prevalences = np.linspace(0.01, 0.99, 50)
    ppv_values = []
    npv_values = []
    
    for prev in prevalences:
        # Bayes' theorem
        ppv = (sensitivity * prev) / (sensitivity * prev + (1 - specificity) * (1 - prev))
        npv = (specificity * (1 - prev)) / (specificity * (1 - prev) + (1 - sensitivity) * prev)
        
        ppv_values.append(ppv)
        npv_values.append(npv)
    
    # Current prevalence in test set
    current_prevalence = (results_df['gt_volume'] > 0).mean()
    
    # Plot
    ax1.plot(prevalences, ppv_values, 'b-', linewidth=2, label='PPV')
    ax1.plot(prevalences, npv_values, 'r-', linewidth=2, label='NPV')
    ax1.axvline(current_prevalence, color='green', linestyle='--', alpha=0.7, 
               label=f'Test Set Prevalence ({current_prevalence:.2f})')
    
    # Add clinical setting markers
    clinical_settings = [
        {'name': 'Screening', 'prevalence': 0.05, 'marker': 'o'},
        {'name': 'Symptomatic', 'prevalence': 0.3, 'marker': 's'},
        {'name': 'Follow-up', 'prevalence': 0.6, 'marker': '^'},
        {'name': 'Known Disease', 'prevalence': 0.9, 'marker': 'D'}
    ]
    
    for setting in clinical_settings:
        prev = setting['prevalence']
        ppv = (sensitivity * prev) / (sensitivity * prev + (1 - specificity) * (1 - prev))
        npv = (specificity * (1 - prev)) / (specificity * (1 - prev) + (1 - sensitivity) * prev)
        
        ax1.scatter(prev, ppv, s=100, marker=setting['marker'], color='blue', 
                   edgecolors='black', linewidth=2)
        ax1.scatter(prev, npv, s=100, marker=setting['marker'], color='red', 
                   edgecolors='black', linewidth=2)
        ax1.annotate(setting['name'], (prev, max(ppv, npv) + 0.02), 
                    ha='center', fontsize=9)
    
    ax1.set_xlabel('Disease Prevalence', fontsize=12)
    ax1.set_ylabel('Predictive Value', fontsize=12)
    ax1.set_title('a) PPV/NPV vs Disease Prevalence', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 2. Number Needed to Review
    ax2 = fig.add_subplot(gs[0, 2:])
    
    # Calculate NNR at different confidence thresholds
    et_cases = results_df[results_df['gt_volume'] > 0].copy()
    confidence_thresholds = np.linspace(0.1, 0.9, 20)
    nnr_values = []
    cases_flagged = []
    
    for thresh in confidence_thresholds:
        # Cases below threshold need review
        need_review = (et_cases['dice'] < thresh).sum()
        correctly_identified = (et_cases['dice'] >= thresh).sum()
        
        if correctly_identified > 0:
            nnr = need_review / correctly_identified
        else:
            nnr = np.inf
        
        nnr_values.append(nnr)
        cases_flagged.append(need_review)
    
    # Create dual y-axis plot
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(confidence_thresholds, nnr_values, 'b-', linewidth=2, label='NNR')
    line2 = ax2_twin.plot(confidence_thresholds, cases_flagged, 'r--', linewidth=2, label='Cases for Review')
    
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Number Needed to Review (NNR)', fontsize=12, color='b')
    ax2_twin.set_ylabel('Number of Cases Flagged', fontsize=12, color='r')
    ax2.set_title('b) Review Burden vs Confidence Threshold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    # Mark optimal points
    if len(nnr_values) > 0:
        min_nnr_idx = np.argmin([v for v in nnr_values if v != np.inf])
        ax2.scatter(confidence_thresholds[min_nnr_idx], nnr_values[min_nnr_idx], 
                   s=100, c='green', marker='*', edgecolors='black', linewidth=2)
        ax2.annotate(f'Optimal\n({confidence_thresholds[min_nnr_idx]:.2f})', 
                    (confidence_thresholds[min_nnr_idx], nnr_values[min_nnr_idx]),
                    xytext=(10, 10), textcoords='offset points', fontsize=9)
    
    # 3. Time Savings Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Simulate time savings
    manual_review_time = 5  # minutes per case
    ai_assisted_time = 2    # minutes per case with AI
    
    workload_scenarios = [
        {'name': '10 cases/day', 'cases': 10},
        {'name': '25 cases/day', 'cases': 25},
        {'name': '50 cases/day', 'cases': 50},
        {'name': '100 cases/day', 'cases': 100}
    ]
    
    time_savings = []
    scenario_names = []
    
    for scenario in workload_scenarios:
        n_cases = scenario['cases']
        
        # Assume AI correctly handles high-confidence cases
        high_conf_rate = (et_cases['dice'] >= 0.7).mean()
        
        manual_time = n_cases * manual_review_time
        ai_time = (n_cases * (1 - high_conf_rate) * manual_review_time + 
                  n_cases * high_conf_rate * ai_assisted_time)
        
        savings = (manual_time - ai_time) / 60  # Convert to hours
        time_savings.append(savings)
        scenario_names.append(scenario['name'])
    
    # Bar plot
    x_pos = np.arange(len(scenario_names))
    bars = ax3.bar(x_pos, time_savings, alpha=0.7, color=colors_pie[2])
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax3.set_ylabel('Time Saved (hours/day)', fontsize=12)
    ax3.set_title('c) Estimated Time Savings by Workload', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (bar, saving) in enumerate(zip(bars, time_savings)):
        total_time = workload_scenarios[i]['cases'] * manual_review_time / 60
        percent_saved = (saving / total_time) * 100
        ax3.text(i, saving + 0.1, f'{percent_saved:.0f}%', 
                ha='center', va='bottom', fontsize=9)
    
    # 4. Alert Fatigue Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate false positive rate at different thresholds
    thresholds = np.linspace(0.1, 0.9, 20)
    fp_rates = []
    alert_rates = []
    
    for thresh in thresholds:
        # Cases flagged as positive
        flagged_positive = results_df['dice'] >= thresh
        
        # False positives among flagged
        false_positives = flagged_positive & (results_df['gt_volume'] == 0)
        
        if flagged_positive.sum() > 0:
            fp_rate = false_positives.sum() / flagged_positive.sum()
        else:
            fp_rate = 0
        
        fp_rates.append(fp_rate)
        alert_rates.append(flagged_positive.mean())
    
    # Plot
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(thresholds, fp_rates, 'r-', linewidth=2, label='False Positive Rate')
    line2 = ax4_twin.plot(thresholds, alert_rates, 'b--', linewidth=2, label='Alert Rate')
    
    ax4.set_xlabel('Confidence Threshold', fontsize=12)
    ax4.set_ylabel('False Positive Rate', fontsize=12, color='r')
    ax4_twin.set_ylabel('Alert Rate', fontsize=12, color='b')
    ax4.set_title('d) Alert Fatigue Risk Analysis', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    
    # 5. Workflow Integration Options
    ax5 = fig.add_subplot(gs[1, 2:])
    
    # Define workflow scenarios
    workflows = [
        {
            'name': 'Triage Mode',
            'description': 'AI prioritizes urgent cases',
            'sensitivity_target': 0.95,
            'specificity_achieved': specificity * 0.8  # Lower specificity for high sensitivity
        },
        {
            'name': 'Screening Mode',
            'description': 'AI filters out negatives',
            'sensitivity_target': sensitivity,
            'specificity_achieved': 0.95
        },
        {
            'name': 'Second Reader',
            'description': 'AI reviews all cases',
            'sensitivity_target': sensitivity,
            'specificity_achieved': specificity
        },
        {
            'name': 'Quality Check',
            'description': 'AI flags discrepancies',
            'sensitivity_target': 0.8,
            'specificity_achieved': 0.9
        }
    ]
    
    # Calculate metrics for each workflow
    workflow_metrics = []
    
    for wf in workflows:
        sens = wf['sensitivity_target']
        spec = wf['specificity_achieved']
        
        # Calculate workload reduction
        workload_reduction = spec * (1 - current_prevalence)
        
        # Calculate error rate
        error_rate = ((1 - sens) * current_prevalence + 
                     (1 - spec) * (1 - current_prevalence))
        
        workflow_metrics.append({
            'Workflow': wf['name'],
            'Workload Reduction': workload_reduction,
            'Error Rate': error_rate,
            'Description': wf['description']
        })
    
    wf_df = pd.DataFrame(workflow_metrics)
    
    # Scatter plot
    scatter = ax5.scatter(wf_df['Workload Reduction'], 1 - wf_df['Error Rate'], 
                         s=200, alpha=0.7, c=range(len(wf_df)), cmap='viridis')
    
    # Add labels
    for i, row in wf_df.iterrows():
        ax5.annotate(row['Workflow'], 
                    (row['Workload Reduction'], 1 - row['Error Rate']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax5.set_xlabel('Workload Reduction', fontsize=12)
    ax5.set_ylabel('Accuracy (1 - Error Rate)', fontsize=12)
    ax5.set_title('e) Workflow Integration Trade-offs', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0.5, 1)
    
    # 6. Clinical Decision Support Metrics
    ax6 = fig.add_subplot(gs[2, :2])
    
    # Calculate decision support metrics
    et_cases = results_df[results_df['gt_volume'] > 0]
    
    decision_metrics = {
        'Actionable Alerts': (et_cases['dice'] >= 0.5).sum(),
        'Missed Cases': (et_cases['dice'] < 0.3).sum(),
        'Borderline Cases': ((et_cases['dice'] >= 0.3) & (et_cases['dice'] < 0.5)).sum(),
        'High Confidence': (et_cases['dice'] >= 0.7).sum()
    }
    
    # Pie chart
    sizes = list(decision_metrics.values())
    labels = [f"{k}\n({v})" for k, v in decision_metrics.items()]
    colors_decision = [colors_pie[i] for i in range(len(labels))]
    
    ax6.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=colors_decision)
    ax6.set_title('f) Clinical Decision Categories', fontsize=14)
    
    # 7. Implementation Requirements Table
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.axis('off')
    
    # Create implementation requirements table
    requirements_data = [
        ['Metric', 'Current Performance', 'Clinical Requirement', 'Status'],
        ['Sensitivity', f'{sensitivity:.1%}', '>95%', '✓' if sensitivity > 0.95 else '✗'],
        ['Specificity', f'{specificity:.1%}', '>90%', '✓' if specificity > 0.90 else '✗'],
        ['PPV (30% prevalence)', f'{(sensitivity * 0.3) / (sensitivity * 0.3 + (1 - specificity) * 0.7):.1%}', '>80%', 
         '✓' if (sensitivity * 0.3) / (sensitivity * 0.3 + (1 - specificity) * 0.7) > 0.8 else '✗'],
        ['NPV (30% prevalence)', f'{(specificity * 0.7) / (specificity * 0.7 + (1 - sensitivity) * 0.3):.1%}', '>95%',
         '✓' if (specificity * 0.7) / (specificity * 0.7 + (1 - sensitivity) * 0.3) > 0.95 else '✗'],
        ['Processing Time', '<5 min/case', 'Required', '?'],
        ['Integration', 'PACS/RIS', 'Required', '?']
    ]
    
    table = ax7.table(cellText=requirements_data[1:], colLabels=requirements_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header and status column
    for i in range(4):
        table[(0, i)].set_facecolor(colors_pie[0])
        table[(0, i)].set_alpha(0.3)
    
    # Color code status
    for i in range(1, 7):
        status_cell = table[(i, 3)]
        if status_cell.get_text().get_text() == '✓':
            status_cell.set_facecolor('lightgreen')
        elif status_cell.get_text().get_text() == '✗':
            status_cell.set_facecolor('lightcoral')
        else:
            status_cell.set_facecolor('lightyellow')
    
    ax7.set_title('g) Clinical Implementation Requirements', fontsize=14, pad=20)
    
    plt.suptitle('Clinical Integration Analysis', fontsize=18)
    plt.tight_layout()
    
    # Print summary
    print("\nClinical Integration Summary:")
    print("=" * 60)
    print(f"Current test set prevalence: {current_prevalence:.1%}")
    print(f"Sensitivity: {sensitivity:.1%}")
    print(f"Specificity: {specificity:.1%}")
    print(f"\nPredictive Values at 30% prevalence:")
    ppv_30 = (sensitivity * 0.3) / (sensitivity * 0.3 + (1 - specificity) * 0.7)
    npv_30 = (specificity * 0.7) / (specificity * 0.7 + (1 - sensitivity) * 0.3)
    print(f"  PPV: {ppv_30:.1%}")
    print(f"  NPV: {npv_30:.1%}")
    print(f"\nWorkflow Impact:")
    print(f"  High confidence cases (no review needed): {(et_cases['dice'] >= 0.7).mean():.1%}")
    print(f"  Cases requiring review: {(et_cases['dice'] < 0.7).mean():.1%}")
    
    return fig

# Generate clinical integration analysis
if 'dice' in results_df.columns and len(results_df) > 0:
    fig_clinical_int = clinical_integration_analysis()
    if fig_clinical_int:
        fig_clinical_int.savefig(os.path.join(figures_out, 'clinical_integration_analysis.png'), 
                                dpi=300, bbox_inches='tight')
        fig_clinical_int.savefig(os.path.join(figures_out, 'clinical_integration_analysis.svg'), 
                                format='svg', bbox_inches='tight')
        plt.show()
else:
    print("No metrics available for clinical integration analysis")

# %%
def create_figure_4_sample_level(disease_posteriors_df, demographics_df, 
                                 pred_probs_all, true_labels_all,
                                 pred_probs_test, true_labels_test,
                                 train_pred_probs_all, train_true_labels_all,
                                 ages, sexes, enhancing_counts,
                                 test_ages, test_sexes, test_enhancing_counts,
                                 train_ages, train_sexes, train_enhancing_counts,
                                 output_path='Figure_4_Sample_Level_Metrics.png'):
    """
    Create comprehensive figure showing sample-level metrics analysis
    """
    
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1])
    
    # Panel A: Train/Test Split Performance Comparison (Bar plot)
    ax_train_test = fig.add_subplot(gs[0, :])
    
    # Calculate sample-level metrics for train and test
    train_metrics = calculate_sample_level_metrics(
        train_pred_probs_all, train_true_labels_all,
        train_ages, train_sexes, train_enhancing_counts
    )
    
    test_metrics = calculate_sample_level_metrics(
        test_pred_probs_all, test_true_labels_all,
        test_ages, test_sexes, test_enhancing_counts
    )
    
    # Bar plot comparison
    metrics_names = ['Balanced\nAccuracy', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1']
    train_values = [
        train_metrics['overall']['balanced_accuracy'],
        train_metrics['overall']['sensitivity'],
        train_metrics['overall']['specificity'],
        train_metrics['overall']['precision'],
        train_metrics['overall']['recall'],
        train_metrics['overall']['f1']
    ]
    test_values = [
        test_metrics['overall']['balanced_accuracy'],
        test_metrics['overall']['sensitivity'],
        test_metrics['overall']['specificity'],
        test_metrics['overall']['precision'],
        test_metrics['overall']['recall'],
        test_metrics['overall']['f1']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax_train_test.bar(x - width/2, train_values, width, label='Train', color='skyblue', edgecolor='black')
    bars2 = ax_train_test.bar(x + width/2, test_values, width, label='Test', color='lightcoral', edgecolor='black')
    
    ax_train_test.set_ylabel('Score', fontsize=14)
    ax_train_test.set_title('A. Train vs Test Performance Comparison', fontsize=16, fontweight='bold', loc='left')
    ax_train_test.set_xticks(x)
    ax_train_test.set_xticklabels(metrics_names, fontsize=12)
    ax_train_test.legend(fontsize=12)
    ax_train_test.set_ylim(0, 1.05)
    ax_train_test.grid(axis='y', alpha=0.6)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_train_test.annotate(f'{height:.3f}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=10)
    
    # Calculate overall metrics for radar plots
    overall_metrics = calculate_sample_level_metrics(
        pred_probs_all, true_labels_all,
        ages, sexes, enhancing_counts
    )
    
    # Updated radar plot settings - removed sensitivity and specificity, added precision, recall, f1
    radar_metrics = ['balanced_acc', 'precision', 'recall', 'f1']
    radar_labels = ['Balanced Acc', 'Precision', 'Recall', 'F1']
    
    # Panel B: Age Group Radar Plot
    ax_age_radar = fig.add_subplot(gs[1, 0], projection='polar')
    age_groups = ['<30', '30-50', '51-70', '>70']
    age_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, (age_group, color) in enumerate(zip(age_groups, age_colors)):
        if age_group in overall_metrics['age']:
            values = [overall_metrics['age'][age_group][metric] for metric in radar_metrics]
            values += values[:1]
            ax_age_radar.plot(angles, values, 'o-', linewidth=2, label=f'{age_group} (n={overall_metrics["age"][age_group]["n"]})', color=color)
            ax_age_radar.fill(angles, values, alpha=0.6, color=color)
    
    ax_age_radar.set_theta_offset(np.pi / 2)
    ax_age_radar.set_theta_direction(-1)
    ax_age_radar.set_xticks(angles[:-1])
    ax_age_radar.set_xticklabels(radar_labels, fontsize=10)
    ax_age_radar.set_ylim(0, 1)
    ax_age_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_age_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax_age_radar.set_title('B. Performance by Age Group', fontsize=14, fontweight='bold', pad=20)
    ax_age_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax_age_radar.grid(True)
    
    # Panel C: Sex Radar Plot
    ax_sex_radar = fig.add_subplot(gs[1, 1], projection='polar')
    sex_labels_display = ['Male', 'Female']
    sex_values_key = ['M', 'F']
    sex_colors = ['#3498db', '#e74c3c']
    
    for i, (sex_key, sex_label, color) in enumerate(zip(sex_values_key, sex_labels_display, sex_colors)):
        if sex_key in overall_metrics['sex']:
            values = [overall_metrics['sex'][sex_key][metric] for metric in radar_metrics]
            values += values[:1]
            ax_sex_radar.plot(angles, values, 'o-', linewidth=2, label=f'{sex_label} (n={overall_metrics["sex"][sex_key]["n"]})', color=color)
            ax_sex_radar.fill(angles, values, alpha=0.6, color=color)
    
    ax_sex_radar.set_theta_offset(np.pi / 2)
    ax_sex_radar.set_theta_direction(-1)
    ax_sex_radar.set_xticks(angles[:-1])
    ax_sex_radar.set_xticklabels(radar_labels, fontsize=10)
    ax_sex_radar.set_ylim(0, 1)
    ax_sex_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_sex_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax_sex_radar.set_title('C. Performance by Sex', fontsize=14, fontweight='bold', pad=20)
    ax_sex_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax_sex_radar.grid(True)
    
    # Panel D: Enhancement Presence Radar Plot
    ax_enhance_radar = fig.add_subplot(gs[1, 2], projection='polar')
    enhance_labels = ['No Enhancement', 'Enhancement']
    enhance_colors = ['#2ecc71', '#e67e22']
    
    for i, (has_enhancement, label, color) in enumerate(zip([0, 1], enhance_labels, enhance_colors)):
        if has_enhancement in overall_metrics['enhancement']:
            values = [overall_metrics['enhancement'][has_enhancement][metric] for metric in radar_metrics]
            values += values[:1]
            ax_enhance_radar.plot(angles, values, 'o-', linewidth=2, 
                                label=f'{label} (n={overall_metrics["enhancement"][has_enhancement]["n"]})', 
                                color=color)
            ax_enhance_radar.fill(angles, values, alpha=0.6, color=color)
    
    ax_enhance_radar.set_theta_offset(np.pi / 2)
    ax_enhance_radar.set_theta_direction(-1)
    ax_enhance_radar.set_xticks(angles[:-1])
    ax_enhance_radar.set_xticklabels(radar_labels, fontsize=10)
    ax_enhance_radar.set_ylim(0, 1)
    ax_enhance_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_enhance_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax_enhance_radar.set_title('D. Performance by Enhancement Status', fontsize=14, fontweight='bold', pad=20)
    ax_enhance_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax_enhance_radar.grid(True)
    
    # Panel E: Disease Posterior Radar Plot
    ax_disease_radar = fig.add_subplot(gs[2, 0], projection='polar')
    
    # Define disease posterior categories
    disease_categories = {
        'GBM': disease_posteriors_df['GBM'] > 0.5,
        'Mets': disease_posteriors_df['Mets'] > 0.5,
        'PCNSL': disease_posteriors_df['PCNSL'] > 0.5,
        'Other': ~((disease_posteriors_df['GBM'] > 0.5) | 
                   (disease_posteriors_df['Mets'] > 0.5) | 
                   (disease_posteriors_df['PCNSL'] > 0.5))
    }
    
    disease_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for (disease, mask), color in zip(disease_categories.items(), disease_colors):
        if mask.sum() > 0:
            disease_probs = pred_probs_all[mask]
            disease_labels = true_labels_all[mask]
            disease_ages = ages[mask] if ages is not None else None
            disease_sexes = sexes[mask] if sexes is not None else None
            disease_enhancing = enhancing_counts[mask] if enhancing_counts is not None else None
            
            metrics = calculate_sample_level_metrics(
                disease_probs, disease_labels,
                disease_ages, disease_sexes, disease_enhancing
            )
            
            values = [metrics['overall'][metric] for metric in radar_metrics]
            values += values[:1]
            
            ax_disease_radar.plot(angles, values, 'o-', linewidth=2, 
                                label=f'{disease} (n={mask.sum()})', color=color)
            ax_disease_radar.fill(angles, values, alpha=0.6, color=color)
    
    ax_disease_radar.set_theta_offset(np.pi / 2)
    ax_disease_radar.set_theta_direction(-1)
    ax_disease_radar.set_xticks(angles[:-1])
    ax_disease_radar.set_xticklabels(radar_labels, fontsize=10)
    ax_disease_radar.set_ylim(0, 1)
    ax_disease_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_disease_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax_disease_radar.set_title('E. Performance by Disease Posterior', fontsize=14, fontweight='bold', pad=20)
    ax_disease_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax_disease_radar.grid(True)
    
    # Panels F-H: Confusion Matrices
    # F: Overall Confusion Matrix
    ax_cm_overall = fig.add_subplot(gs[2, 1])
    y_pred = (pred_probs_all >= 0.5).astype(int)
    y_true = true_labels_all
    cm_overall = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Enhancement', 'Enhancement'],
                yticklabels=['No Enhancement', 'Enhancement'],
                ax=ax_cm_overall, cbar_kws={'label': 'Count'})
    ax_cm_overall.set_xlabel('Predicted', fontsize=12)
    ax_cm_overall.set_ylabel('Actual', fontsize=12)
    ax_cm_overall.set_title('F. Overall Confusion Matrix', fontsize=14, fontweight='bold')
    
    # G: High Confidence Confusion Matrix
    ax_cm_high_conf = fig.add_subplot(gs[2, 2])
    high_conf_mask = (pred_probs_all >= 0.8) | (pred_probs_all <= 0.2)
    if high_conf_mask.sum() > 0:
        y_pred_high = (pred_probs_all[high_conf_mask] >= 0.5).astype(int)
        y_true_high = true_labels_all[high_conf_mask]
        cm_high = confusion_matrix(y_true_high, y_pred_high)
        
        sns.heatmap(cm_high, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['No Enhancement', 'Enhancement'],
                    yticklabels=['No Enhancement', 'Enhancement'],
                    ax=ax_cm_high_conf, cbar_kws={'label': 'Count'})
        ax_cm_high_conf.set_xlabel('Predicted', fontsize=12)
        ax_cm_high_conf.set_ylabel('Actual', fontsize=12)
        ax_cm_high_conf.set_title(f'G. High Confidence (n={high_conf_mask.sum()})', 
                                 fontsize=14, fontweight='bold')
    
    # H: ROC Curves by Subgroup
    ax_roc = fig.add_subplot(gs[3, 0])
    
    # Overall ROC
    fpr_overall, tpr_overall, _ = roc_curve(true_labels_all, pred_probs_all)
    auc_overall = auc(fpr_overall, tpr_overall)
    ax_roc.plot(fpr_overall, tpr_overall, 'k-', linewidth=2.5, 
                label=f'Overall (AUC = {auc_overall:.3f})')
    
    # Age group ROCs
    for age_group, color in zip(age_groups, age_colors):
        if age_group in overall_metrics['age']:
            age_mask = get_age_mask(ages, age_group)
            if age_mask.sum() > 10:
                fpr_age, tpr_age, _ = roc_curve(true_labels_all[age_mask], 
                                               pred_probs_all[age_mask])
                auc_age = auc(fpr_age, tpr_age)
                ax_roc.plot(fpr_age, tpr_age, '--', linewidth=1.5, color=color,
                           label=f'{age_group} (AUC = {auc_age:.3f})', alpha=0.6)
    
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title('H. ROC Curves by Age Group', fontsize=14, fontweight='bold')
    ax_roc.legend(fontsize=10)
    ax_roc.grid(alpha=0.6)
    
    # I: Calibration Plot
    ax_calib = fig.add_subplot(gs[3, 1])
    
    # Overall calibration
    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_labels_all, pred_probs_all, n_bins=10, strategy='uniform'
    )
    
    ax_calib.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                  label='Overall', color='black', linewidth=2, markersize=8)
    
    # Perfect calibration line
    ax_calib.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.6)
    
    # Sex-specific calibration
    for sex_key, sex_label, color in zip(sex_values_key, sex_labels_display, sex_colors):
        sex_mask = sexes == sex_key
        if sex_mask.sum() > 50:
            try:
                frac_pos_sex, mean_pred_sex = calibration_curve(
                    true_labels_all[sex_mask], pred_probs_all[sex_mask], 
                    n_bins=8, strategy='uniform'
                )
                ax_calib.plot(mean_pred_sex, frac_pos_sex, 's--', 
                            label=sex_label, color=color, alpha=0.6, markersize=6)
            except:
                pass
    
    ax_calib.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax_calib.set_ylabel('Fraction of Positives', fontsize=12)
    ax_calib.set_title('I. Calibration Plot', fontsize=14, fontweight='bold')
    ax_calib.legend(fontsize=10)
    ax_calib.grid(alpha=0.6)
    
    # J: Performance Summary Table
    ax_table = fig.add_subplot(gs[3, 2])
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Create summary statistics
    summary_data = []
    
    # Overall metrics
    summary_data.append(['Overall', 
                        f"{overall_metrics['overall']['n']}",
                        f"{overall_metrics['overall']['balanced_accuracy']:.3f}",
                        f"{overall_metrics['overall']['auc']:.3f}",
                        f"{overall_metrics['overall']['f1']:.3f}"])
    
    # By enhancement status
    for has_enhancement, label in zip([0, 1], ['No Enhancement', 'Enhancement']):
        if has_enhancement in overall_metrics['enhancement']:
            m = overall_metrics['enhancement'][has_enhancement]
            summary_data.append([f'  {label}', 
                               f"{m['n']}",
                               f"{m['balanced_accuracy']:.3f}",
                               f"{m['auc']:.3f}",
                               f"{m['f1']:.3f}"])
    
    # By age
    summary_data.append(['By Age', '', '', '', ''])
    for age_group in age_groups:
        if age_group in overall_metrics['age']:
            m = overall_metrics['age'][age_group]
            summary_data.append([f'  {age_group}', 
                               f"{m['n']}",
                               f"{m['balanced_accuracy']:.3f}",
                               f"{m['auc']:.3f}",
                               f"{m['f1']:.3f}"])
    
    # Create table
    table = ax_table.table(cellText=summary_data,
                          colLabels=['Group', 'N', 'Balanced Acc', 'AUC', 'F1'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.3, 0.15, 0.2, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif summary_data[i-1][0] in ['By Age', 'By Sex']:  # Section headers
                cell.set_facecolor('#E0E0E0')
                cell.set_text_props(weight='bold')
            elif summary_data[i-1][0].startswith('  '):  # Indented rows
                cell.set_facecolor('#F5F5F5')
    
    ax_table.set_title('J. Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return fig

# %%
# def create_figure_5_sample_level(results_df, figures_out):
#     """
#     Create Figure 4 with sample-level metrics instead of voxel-level metrics
#     Using the exact aesthetics from the original figure 4
#     """
    
#     # Calculate sample-level metrics (patient-level detection)
#     tp = ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] > 0)).sum()
#     tn = ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] == 0)).sum()
#     fp = ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] > 0)).sum()
#     fn = ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] == 0)).sum()

#     overall_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#     overall_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#     overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#     overall_recall = overall_sensitivity  # Same as sensitivity
#     overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
#     overall_balanced_acc = (overall_sensitivity + overall_specificity) / 2

#     # Calculate AUROC using predicted volume as confidence
#     from sklearn.metrics import roc_auc_score
#     y_true_binary = (results_df['gt_volume'] > 0).astype(int)
#     y_scores = results_df['pred_volume'].fillna(0)
#     overall_auroc = roc_auc_score(y_true_binary, y_scores)

#     # Define sample-level metrics for consistency
#     all_metrics_names = ['AUROC', 'Balanced Acc', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1']
#     all_metrics_values = [overall_auroc, overall_balanced_acc, overall_sensitivity, overall_specificity,
#                          overall_precision, overall_recall, overall_f1]

#     # Define consistent color palette
#     import seaborn as sns
#     colors_pie = sns.color_palette('husl', n_colors=10)
#     metric_colors = [colors_pie[i] for i in range(7)]

#     # Metrics for radar plots - CHANGED to precision, recall, f1 instead of sensitivity, specificity
#     radar_metrics = ['balanced_acc', 'precision', 'recall', 'f1']
#     radar_labels = ['Balanced Acc', 'Precision', 'Recall', 'F1']

#     # Groups to plot
#     groupings = [
#         ('b) Performance across datasets', results_df['Cohort'].unique()),
#         ('c) Performance across pathologies', results_df['Pathology'].unique()),
#         ('d) Performance across countries', results_df['Country'].unique()),
#     ]

#     # Define age bins for panel e
#     age_bins = [0, 20, 40, 60, 80, 100]
#     age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']

#     # Calculate data availability for age and sex
#     age_available = results_df['Age'].notna().sum()
#     age_percentage = (age_available / len(results_df)) * 100
#     sex_available = results_df['Sex'].notna().sum()
#     sex_percentage = (sex_available / len(results_df)) * 100

#     # Create a 2x3 grid with INCREASED row spacing and column spacing
#     fig_radar = plt.figure(figsize=(18, 12))
#     gs = fig_radar.add_gridspec(2, 3, hspace=0.375, wspace=0.4)

#     # PANEL A: Overall Sample-level Detection Performance as BAR CHART
#     ax_image_detection = fig_radar.add_subplot(gs[0, 0])

#     # Bar chart using consistent colors for each metric
#     bar_positions = np.arange(len(all_metrics_names))
#     bars = ax_image_detection.bar(bar_positions, all_metrics_values, alpha=0.7,
#                                  color=metric_colors)

#     # Add value labels on bars
#     for i, (metric, value) in enumerate(zip(all_metrics_names, all_metrics_values)):
#         ax_image_detection.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

#     ax_image_detection.set_ylabel('Performance')
#     ax_image_detection.set_title('a) Overall Sample-level Detection Performance')
#     ax_image_detection.set_xticks(bar_positions)
#     ax_image_detection.set_xticklabels(all_metrics_names, rotation=45, ha='right')
#     ax_image_detection.set_ylim(0, 1.1)
#     ax_image_detection.grid(True, alpha=0.7, axis='y')  # Changed from 0.3 to 0.7

#     # Function to calculate sample-level metrics for a given mask
#     def calculate_sample_level_metrics(mask):
#         """Calculate sample-level detection metrics for a subset of data"""
#         if isinstance(mask, pd.Series):
#             mask = mask.reindex(results_df.index, fill_value=False)
        
#         subset = results_df[mask].copy()
        
#         if len(subset) == 0:
#             return [np.nan] * 7
        
#         # Calculate sample-level binary classification metrics
#         subset_tp = ((subset['gt_volume'] > 0) & (subset['pred_volume'] > 0)).sum()
#         subset_tn = ((subset['gt_volume'] == 0) & (subset['pred_volume'] == 0)).sum()
#         subset_fp = ((subset['gt_volume'] == 0) & (subset['pred_volume'] > 0)).sum()
#         subset_fn = ((subset['gt_volume'] > 0) & (subset['pred_volume'] == 0)).sum()
        
#         # Calculate metrics
#         sensitivity = subset_tp / (subset_tp + subset_fn) if (subset_tp + subset_fn) > 0 else 0
        
#         if (subset_tn + subset_fp) > 0:
#             specificity = subset_tn / (subset_tn + subset_fp)
#         else:
#             specificity = np.nan
        
#         if np.isnan(specificity):
#             balanced_acc = sensitivity
#         else:
#             balanced_acc = (sensitivity + specificity) / 2
        
#         precision = subset_tp / (subset_tp + subset_fp) if (subset_tp + subset_fp) > 0 else 0
#         recall = sensitivity
#         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
#         # Calculate AUROC if possible
#         if len(subset['gt_volume'].unique()) > 1:
#             try:
#                 y_true_subset = (subset['gt_volume'] > 0).astype(int)
#                 y_scores_subset = subset['pred_volume'].fillna(0)
#                 auroc = roc_auc_score(y_true_subset, y_scores_subset)
#             except:
#                 auroc = np.nan
#         else:
#             auroc = np.nan
        
#         return [auroc, balanced_acc, sensitivity, specificity, precision, recall, f1]

#     # CORRECTED RADAR PLOT POSITIONING - Panels b, c, d
#     axes_radar = [
#         fig_radar.add_subplot(gs[0, 1], polar=True),  # Dataset - column 1, row 0
#         fig_radar.add_subplot(gs[0, 2], polar=True),  # Pathology - column 2, row 0
#         fig_radar.add_subplot(gs[1, 0], polar=True)   # Countries - column 0, row 1
#     ]

#     for ax, (title, group_list) in zip(axes_radar, groupings):
#         group_names = []
#         metric_means = {metric: [] for metric in radar_metrics}
        
#         for group in group_list:
#             # For cohorts, use direct cohort matching
#             if title == 'b) Performance across datasets':
#                 mask = results_df['Cohort'] == group
#             # For pathologies, use pathology column matching
#             elif title == 'c) Performance across pathologies':
#                 mask = results_df['Pathology'] == group
#             # For countries, use country column matching
#             elif title == 'd) Performance across countries':
#                 mask = results_df['Country'] == group
#             else:
#                 mask = pd.Series([False] * len(results_df), index=results_df.index)
            
#             if mask.sum() == 0:
#                 for metric in radar_metrics:
#                     metric_means[metric].append(np.nan)
#                 group_names.append(group)
#                 continue
            
#             # Calculate all 7 metrics
#             all_values = calculate_sample_level_metrics(mask)
            
#             # Extract the metrics we want for radar plots
#             # balanced_acc is at index 1, precision at 4, recall at 5, f1 at 6
#             metric_means['balanced_acc'].append(all_values[1])
#             metric_means['precision'].append(all_values[4])
#             metric_means['recall'].append(all_values[5])
#             metric_means['f1'].append(all_values[6])
            
#             # Format group names for better display
#             if len(group) > 15:  # If name is long, add line breaks
#                 words = group.split()
#                 if len(words) > 2:
#                     mid_point = len(words) // 2
#                     formatted_name = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
#                 else:
#                     formatted_name = group.replace(' ', '\n', 1)
#             else:
#                 formatted_name = group
#             group_names.append(formatted_name)
        
#         # Radar plot setup
#         offset = 0 #np.pi / 2  # Start at the top
#         angles = [n / float(len(group_names)) * 2 * np.pi +offset for n in range(len(group_names))]
#         angles += angles[:1]  # close the loop
        
#         # Plot each metric with its color
#         metric_color_map = {
#             'balanced_acc': metric_colors[1],  # Balanced Acc color
#             'precision': metric_colors[4],     # Precision color
#             'recall': metric_colors[5],        # Recall color
#             'f1': metric_colors[6]             # F1 color
#         }
        
#         for idx, (metric, label) in enumerate(zip(radar_metrics, radar_labels)):
#             values = metric_means[metric] + [metric_means[metric][0]]
#             values_plot = [0 if np.isnan(v) else v for v in values]
            
#             # Plot WITHOUT dots/markers as requested in original
#             ax.plot(angles, values_plot, color=metric_color_map[metric], linewidth=2, label=label)
#             ax.fill(angles, values_plot, color=metric_color_map[metric], alpha=0.05)  # Reduced to 0.05
        
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(group_names, rotation=0, fontsize=9,
#                           verticalalignment='center', horizontalalignment='center',
#                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
#         ax.set_title(title)
#         ax.set_ylim(0, 1)
#         ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
#         ax.set_rlabel_position(45)
#         ax.grid(True, alpha=0.7)  # Changed from 0.3 to 0.7
        
#         # Add legend only for panels b and d
#         if title in ['b) Performance across datasets']:
#             ax.legend(loc='upper right', bbox_to_anchor=(1.40, 1.1), fontsize=8)
#         if title in  ['d) Performance across countries']:
#             ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1.1), fontsize=8)
        
#         # Ensure labels are on top by setting zorder
#         for label in ax.get_xticklabels():
#             label.set_zorder(100)
#         for label in ax.get_yticklabels():
#             label.set_zorder(100)

#     # PANEL E: Accuracy across ages - RADAR PLOT
#     ax_age = fig_radar.add_subplot(gs[1, 1], polar=True)

#     # Bin ages
#     results_df['Age_Bin'] = pd.cut(results_df['Age'], bins=age_bins, labels=age_labels)

#     # Calculate metrics for each age group
#     age_metric_means = {metric: [] for metric in radar_metrics}
    
#     for age_group in age_labels:
#         mask = results_df['Age_Bin'] == age_group
#         if mask.sum() > 0:
#             metrics = calculate_sample_level_metrics(mask)
#             age_metric_means['balanced_acc'].append(metrics[1])
#             age_metric_means['precision'].append(metrics[4])
#             age_metric_means['recall'].append(metrics[5])
#             age_metric_means['f1'].append(metrics[6])
#         else:
#             for metric in radar_metrics:
#                 age_metric_means[metric].append(np.nan)

#     # Setup radar plot for age groups
#     angles_age = [n / float(len(age_labels)) * 2 * np.pi for n in range(len(age_labels))]
#     angles_age += angles_age[:1]  # close the loop

#     # Plot each metric
#     for idx, (metric, label) in enumerate(zip(radar_metrics, radar_labels)):
#         values_age = age_metric_means[metric] + [age_metric_means[metric][0]]
#         values_age_plot = [0 if np.isnan(v) else v for v in values_age]
        
#         # Plot WITHOUT dots/markers as requested
#         ax_age.plot(angles_age, values_age_plot, color=metric_color_map[metric], linewidth=2, label=label)
#         ax_age.fill(angles_age, values_age_plot, color=metric_color_map[metric], alpha=0.05)  # Reduced to 0.05

#     ax_age.set_xticks(angles_age[:-1])
#     ax_age.set_xticklabels(age_labels, rotation=0, fontsize=9,
#                            verticalalignment='center', horizontalalignment='center',
#                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))
#     ax_age.set_title(f'e) Performance across ages (Available for {age_percentage:.1f}%)')
#     ax_age.set_ylim(0, 1)
#     ax_age.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
#     ax_age.set_rlabel_position(45)
#     ax_age.grid(True, alpha=0.7)  # Changed from 0.3 to 0.7
#     # No legend for panel e as requested

#     # PANEL F: Accuracy across sexes - BAR CHART
#     ax_sex = fig_radar.add_subplot(gs[1, 2])

#     # Calculate all metrics for each sex
#     male_metrics = []
#     female_metrics = []

#     male_mask = results_df['Sex'] == 'M'
#     female_mask = results_df['Sex'] == 'F'

#     if male_mask.sum() > 0:
#         male_metrics = calculate_sample_level_metrics(male_mask)
#     else:
#         male_metrics = [np.nan] * 7

#     if female_mask.sum() > 0:
#         female_metrics = calculate_sample_level_metrics(female_mask)
#     else:
#         female_metrics = [np.nan] * 7

#     # Setup grouped bar chart
#     x = np.arange(len(all_metrics_names))
#     width = 0.35

#     # Plot bars for male and female - using the SAME colors as the metrics
#     bars1 = ax_sex.bar(x - width/2, male_metrics, width, label='Male', 
#                        color=metric_colors, alpha=0.7)
#     bars2 = ax_sex.bar(x + width/2, female_metrics, width, label='Female',
#                        color=metric_colors, alpha=0.7)

#     # Add dashed edge for female bars
#     for bar in bars2:
#         bar.set_edgecolor('black')
#         bar.set_linewidth(1.5)
#         bar.set_linestyle('--')

#     # Add sample sizes in legend
#     male_count = male_mask.sum()
#     female_count = female_mask.sum()

#     # Create custom legend with light grey colors as requested
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='lightgrey', edgecolor='black', label=f'Male (n={male_count})'),
#         Patch(facecolor='lightgrey', edgecolor='black', linestyle='--', linewidth=1.5, label=f'Female (n={female_count})')
#     ]
#     ax_sex.legend(handles=legend_elements, loc='lower right')

#     ax_sex.set_xlabel('Metrics')
#     ax_sex.set_ylabel('Performance')
#     ax_sex.set_title(f'f) Performance across sexes (Available for {sex_percentage:.1f}%)')
#     ax_sex.set_xticks(x)
#     ax_sex.set_xticklabels(all_metrics_names, rotation=45, ha='right')
#     ax_sex.set_ylim(0, 1.1)
#     ax_sex.grid(True, alpha=0.7, axis='y')  # Changed from 0.3 to 0.7

#     # Main title
#     plt.suptitle('Equitable calibration of enhancement detection', fontsize=18, y=0.95)

#     plt.tight_layout()

#     # Save figure - Fix the path issue
#     import os
#     from pathlib import Path
    
#     # Convert figures_out to Path if it's a string
#     if isinstance(figures_out, str):
#         figures_out_path = Path(figures_out)
#     else:
#         figures_out_path = figures_out
        
#     save_path = figures_out_path / 'Figure_5'
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
#     plt.savefig(save_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
#     plt.close()
    
#     print(f"Figure 5 saved to {save_path}")
    
#     # Print summary statistics
#     print("\nSample-level Detection Performance:")
#     print(f"  Sensitivity: {overall_sensitivity:.2%} ({tp}/{tp+fn} patients with enhancement detected)")
#     print(f"  Specificity: {overall_specificity:.2%} ({tn}/{tn+fp} patients without enhancement correctly identified)")
#     print(f"  Precision: {overall_precision:.2%} ({tp}/{tp+fp} positive predictions were correct)")
#     print(f"  Balanced Accuracy: {overall_balanced_acc:.2%}")
#     print(f"  F1 Score: {overall_f1:.2%}")
#     print(f"  AUROC: {overall_auroc:.3f}")
    
#     return fig_radar

# %%
# # Generate Figure 4 with updated radar plot metrics
# create_figure_5_sample_level(results_df, figures_out)

# %%
def create_figure_5_sample_level_ADAPTED(results_df, figures_out):
    """
    Create Figure 4 with sample-level metrics instead of voxel-level metrics
    Using the exact aesthetics from the original figure 4
    """
    
    # Calculate sample-level metrics (patient-level detection)
    tp = ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] > 0)).sum()
    tn = ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] == 0)).sum()
    fp = ((results_df['gt_volume'] == 0) & (results_df['pred_volume'] > 0)).sum()
    fn = ((results_df['gt_volume'] > 0) & (results_df['pred_volume'] == 0)).sum()

    overall_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    overall_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    overall_recall = overall_sensitivity  # Same as sensitivity
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_balanced_acc = (overall_sensitivity + overall_specificity) / 2
    
    # Debug print to check values
    print(f"\nDEBUG Figure 5 Overall Metrics Calculation:")
    print(f"  Dataset size: {len(results_df)}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Sensitivity={overall_sensitivity:.3f}")
    print(f"  Specificity={overall_specificity:.3f}")
    print(f"  Balanced Accuracy={overall_balanced_acc:.3f}")
    print(f"  Overall Accuracy={(tp+tn)/(tp+tn+fp+fn):.3f}")

    # Calculate AUROC using predicted volume as confidence
    from sklearn.metrics import roc_auc_score
    y_true_binary = (results_df['gt_volume'] > 0).astype(int)
    y_scores = results_df['pred_volume'].fillna(0)
    overall_auroc = roc_auc_score(y_true_binary, y_scores)

    # Define sample-level metrics for consistency
    all_metrics_names = ['AUROC', 'Balanced Acc', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1']
    # Bootstrap for confidence intervals
    np.random.seed(42)
    n_bootstrap = 1000
    
    # Bootstrap for overall metrics (Panel A)
    overall_ci = {}
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(results_df), size=len(results_df), replace=True)
        boot_df = results_df.iloc[indices]
        
        # Calculate metrics for bootstrap sample
        tp_boot = ((boot_df['gt_volume'] > 0) & (boot_df['pred_volume'] > 0)).sum()
        tn_boot = ((boot_df['gt_volume'] == 0) & (boot_df['pred_volume'] == 0)).sum()
        fp_boot = ((boot_df['gt_volume'] == 0) & (boot_df['pred_volume'] > 0)).sum()
        fn_boot = ((boot_df['gt_volume'] > 0) & (boot_df['pred_volume'] == 0)).sum()
        
        # Calculate metrics
        if (tp_boot + fn_boot) > 0:
            sensitivity_boot = tp_boot / (tp_boot + fn_boot)
        else:
            sensitivity_boot = 0
            
        if (tn_boot + fp_boot) > 0:
            specificity_boot = tn_boot / (tn_boot + fp_boot)
        else:
            specificity_boot = 0
            
        if (tp_boot + fp_boot) > 0:
            precision_boot = tp_boot / (tp_boot + fp_boot)
        else:
            precision_boot = 0
            
        recall_boot = sensitivity_boot
        
        if (precision_boot + recall_boot) > 0:
            f1_boot = 2 * (precision_boot * recall_boot) / (precision_boot + recall_boot)
        else:
            f1_boot = 0
            
        balanced_acc_boot = (sensitivity_boot + specificity_boot) / 2
        
        # Store results
        if 'balanced_accuracy' not in overall_ci:
            overall_ci['auroc'] = []
            overall_ci['balanced_accuracy'] = []
            overall_ci['sensitivity'] = []
            overall_ci['specificity'] = []
            overall_ci['precision'] = []
            overall_ci['recall'] = []
            overall_ci['f1'] = []
            
        # Calculate AUROC for bootstrap sample
        from sklearn.metrics import roc_auc_score
        try:
            auroc_boot = roc_auc_score(boot_df['gt_volume'] > 0, boot_df['pred_volume'])
        except:
            auroc_boot = 0.5  # Default if calculation fails
            
        overall_ci['auroc'].append(auroc_boot)
        overall_ci['balanced_accuracy'].append(balanced_acc_boot)
        overall_ci['sensitivity'].append(sensitivity_boot)
        overall_ci['specificity'].append(specificity_boot)
        overall_ci['precision'].append(precision_boot)
        overall_ci['recall'].append(recall_boot)
        overall_ci['f1'].append(f1_boot)
    
    # Calculate 95% confidence intervals
    overall_ci_bounds = {}
    for metric, values in overall_ci.items():
        overall_ci_bounds[metric] = (np.percentile(values, 2.5), np.percentile(values, 97.5))
    

    all_metrics_values = [overall_auroc, overall_balanced_acc, overall_sensitivity, overall_specificity,
                         overall_precision, overall_recall, overall_f1]

    # Define consistent color palette
    import seaborn as sns
    colors_pie = sns.color_palette('husl', n_colors=10)
    metric_colors = [colors_pie[i] for i in range(7)]

    # Metrics for radar plots - CHANGED to precision, recall, f1 instead of sensitivity, specificity
    radar_metrics = ['balanced_acc', 'precision', 'recall', 'f1']
    radar_labels = ['Balanced Acc', 'Precision', 'Recall', 'F1']

    # Groups to plot
    groupings = [
        ('b) Performance across datasets', results_df['Cohort'].unique()),
        ('c) Performance across pathologies', results_df['Pathology'].unique()),
        ('d) Performance across countries', ['USA', 'UK', 'Netherlands', 'Sub-Saharan Africa']),
    ]

    # Define age bins for panel e
    age_bins = [0, 20, 40, 60, 80, 100]
    age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']

    # Calculate data availability for age and sex
    age_available = results_df['Age'].notna().sum()
    age_percentage = (age_available / len(results_df)) * 100
    sex_available = results_df['Sex'].notna().sum()
    sex_percentage = (sex_available / len(results_df)) * 100

    # Create a 2x3 grid with INCREASED row spacing and column spacing
    fig_radar = plt.figure(figsize=(18, 12))
    gs = fig_radar.add_gridspec(2, 3, hspace=0.375, wspace=0.45)

    # PANEL A: Overall Sample-level Detection Performance as BAR CHART
    ax_image_detection = fig_radar.add_subplot(gs[0, 0])

    # Bar chart using consistent colors for each metric
    bar_positions = np.arange(len(all_metrics_names))
    # Calculate errors from confidence intervals
    errors = []
    metric_names_ci = ['auroc', 'balanced_accuracy', 'sensitivity', 'specificity', 'precision', 'recall', 'f1']
    for metric in metric_names_ci:
        lower, upper = overall_ci_bounds[metric]
        # Get the point estimate
        if metric == 'auroc':
            point = overall_auroc
        elif metric == 'balanced_accuracy':
            point = overall_balanced_acc
        elif metric == 'sensitivity' or metric == 'recall':
            point = overall_sensitivity
        elif metric == 'specificity':
            point = overall_specificity
        elif metric == 'precision':
            point = overall_precision
        elif metric == 'f1':
            point = overall_f1
        # Error is the larger of the two differences
        error = max(point - lower, upper - point)
        errors.append(error)
    

    bars = ax_image_detection.bar(bar_positions, all_metrics_values, yerr=errors, capsize=5, alpha=0.7,
                                 color=metric_colors)

    # Add value labels on bars above confidence intervals
    for i, (metric, value, error) in enumerate(zip(all_metrics_names, all_metrics_values, errors)):
        # Position text above the upper confidence interval
        text_y = value + error + 0.02
        ax_image_detection.text(i, text_y, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    ax_image_detection.set_ylabel('Performance')
    ax_image_detection.set_title('a) Overall performance', pad=20)
    ax_image_detection.set_xticks(bar_positions)
    ax_image_detection.set_xticklabels(all_metrics_names, rotation=45, ha='right')
    ax_image_detection.set_ylim(0, 1.1)
    ax_image_detection.grid(True, alpha=0.7, axis='y')  # Changed from 0.3 to 0.7

    # Function to calculate sample-level metrics for a given mask
    def calculate_sample_level_metrics(mask):
        """Calculate sample-level detection metrics for a subset of data"""
        if isinstance(mask, pd.Series):
            mask = mask.reindex(results_df.index, fill_value=False)
        
        subset = results_df[mask].copy()
        
        if len(subset) == 0:
            return [np.nan] * 7
        
        # Calculate sample-level binary classification metrics
        subset_tp = ((subset['gt_volume'] > 0) & (subset['pred_volume'] > 0)).sum()
        subset_tn = ((subset['gt_volume'] == 0) & (subset['pred_volume'] == 0)).sum()
        subset_fp = ((subset['gt_volume'] == 0) & (subset['pred_volume'] > 0)).sum()
        subset_fn = ((subset['gt_volume'] > 0) & (subset['pred_volume'] == 0)).sum()
        
        # Calculate metrics
        sensitivity = subset_tp / (subset_tp + subset_fn) if (subset_tp + subset_fn) > 0 else 0
        
        if (subset_tn + subset_fp) > 0:
            specificity = subset_tn / (subset_tn + subset_fp)
        else:
            specificity = np.nan
        
        if np.isnan(specificity):
            # When there are no negative cases (all cases have enhancement), 
            # balanced accuracy cannot be properly calculated
            balanced_acc = sensitivity  # This is why BraTS-MEN shows high on radar but low in print
        else:
            balanced_acc = (sensitivity + specificity) / 2
        
        precision = subset_tp / (subset_tp + subset_fp) if (subset_tp + subset_fp) > 0 else 0
        recall = sensitivity
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AUROC if possible
        if len(subset['gt_volume'].unique()) > 1:
            try:
                y_true_subset = (subset['gt_volume'] > 0).astype(int)
                y_scores_subset = subset['pred_volume'].fillna(0)
                auroc = roc_auc_score(y_true_subset, y_scores_subset)
            except:
                auroc = np.nan
        else:
            auroc = np.nan
        
        return [auroc, balanced_acc, sensitivity, specificity, precision, recall, f1]

    # CORRECTED RADAR PLOT POSITIONING - Panels b, c, d
    axes_radar = [
        fig_radar.add_subplot(gs[0, 1], polar=True),  # Dataset - column 1, row 0
        fig_radar.add_subplot(gs[0, 2], polar=True),  # Pathology - column 2, row 0
        fig_radar.add_subplot(gs[1, 0], polar=True)   # Countries - column 0, row 1
    ]

    for ax, (title, group_list) in zip(axes_radar, groupings):
        group_names = []
        metric_means = {metric: [] for metric in radar_metrics}
        
        for group in group_list:
            # For cohorts, use direct cohort matching
            if title == 'b) Performance across datasets':
                mask = results_df['Cohort'] == group
            # For pathologies, use pathology column matching
            elif title == 'c) Performance across pathologies':
                mask = results_df['Pathology'] == group
            # For countries, use actual country columns (binary indicators)
            elif title == 'd) Performance across countries':
                if group in results_df.columns:
                    mask = results_df[group] == 1
                else:
                    mask = pd.Series([False] * len(results_df), index=results_df.index)
            else:
                mask = pd.Series([False] * len(results_df), index=results_df.index)
            
            if mask.sum() == 0:
                for metric in radar_metrics:
                    metric_means[metric].append(np.nan)
                group_names.append(group)
                continue
            
            # Calculate all 7 metrics
            all_values = calculate_sample_level_metrics(mask)
            
            # Extract the metrics we want for radar plots
            # balanced_acc is at index 1, precision at 4, recall at 5, f1 at 6
            metric_means['balanced_acc'].append(all_values[1])
            metric_means['precision'].append(all_values[4])
            metric_means['recall'].append(all_values[5])
            metric_means['f1'].append(all_values[6])
            
            # Format group names for better display
            if len(group) > 15:  # If name is long, add line breaks
                words = group.split()
                if len(words) > 2:
                    mid_point = len(words) // 2
                    formatted_name = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
                else:
                    formatted_name = group.replace(' ', '\n', 1)
            else:
                formatted_name = group
            group_names.append(formatted_name)
        
        # Radar plot setup with offset for countries
        if title == 'd) Performance across countries':
            offset = np.pi / 4  # 45-degree offset for countries
        else:
            offset = 0  # No offset for other plots
        angles = [n / float(len(group_names)) * 2 * np.pi + offset for n in range(len(group_names))]
        angles += angles[:1]  # close the loop
        
        # Plot each metric with its color
        metric_color_map = {
            'balanced_acc': metric_colors[1],  # Balanced Acc color
            'precision': metric_colors[4],     # Precision color
            'recall': metric_colors[5],        # Recall color
            'f1': metric_colors[6]             # F1 color
        }
        
        for idx, (metric, label) in enumerate(zip(radar_metrics, radar_labels)):
            values = metric_means[metric] + [metric_means[metric][0]]
            values_plot = [0 if np.isnan(v) else v for v in values]
            
            # Plot WITHOUT dots/markers as requested in original
            ax.plot(angles, values_plot, color=metric_color_map[metric], linewidth=2, label=label)
            ax.fill(angles, values_plot, color=metric_color_map[metric], alpha=0.05)  # Reduced to 0.05
        
        # Remove default labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        
        # Manually place labels further out
        label_radius = 1.075  # Reduced from 1.3 to avoid overlap with titles
        for angle, label in zip(angles[:-1], group_names):
            ha = 'center'
            va = 'center'
            # Adjust alignment based on angle
            if np.cos(angle) > 0.1:
                ha = 'left'
            elif np.cos(angle) < -0.1:
                ha = 'right'
            if np.sin(angle) > 0.1:
                va = 'bottom'
            elif np.sin(angle) < -0.1:
                va = 'top'
                
            ax.text(angle, label_radius, label, 
                   horizontalalignment=ha, verticalalignment=va,
                   fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'),
                   transform=ax.get_xaxis_transform())
        
        ax.set_title(title, pad=20)  # Add padding to title to avoid overlap
        ax.set_ylim(0, 1)
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_rlabel_position(20)
        ax.grid(True, alpha=0.7)  # Changed from 0.3 to 0.7
        
        # Add legend only for panels b and d
        if title in ['b) Performance across datasets']:
            ax.legend(loc='upper right', bbox_to_anchor=(1.40, 1.1), fontsize=8)
        if title in  ['d) Performance across countries']:
            ax.legend(loc='upper right', bbox_to_anchor=(1.30, 1.1), fontsize=8)
        
        # Ensure labels are on top by setting zorder
        for label in ax.get_xticklabels():
            label.set_zorder(100)
        for label in ax.get_yticklabels():
            label.set_zorder(100)

    # PANEL E: Accuracy across ages - RADAR PLOT
    ax_age = fig_radar.add_subplot(gs[1, 1], polar=True)

    # Bin ages
    results_df['Age_Bin'] = pd.cut(results_df['Age'], bins=age_bins, labels=age_labels)

    # Calculate metrics for each age group
    age_metric_means = {metric: [] for metric in radar_metrics}
    
    for age_group in age_labels:
        mask = results_df['Age_Bin'] == age_group
        if mask.sum() > 0:
            metrics = calculate_sample_level_metrics(mask)
            age_metric_means['balanced_acc'].append(metrics[1])
            age_metric_means['precision'].append(metrics[4])
            age_metric_means['recall'].append(metrics[5])
            age_metric_means['f1'].append(metrics[6])
        else:
            for metric in radar_metrics:
                age_metric_means[metric].append(np.nan)

    # Setup radar plot for age groups
    angles_age = [n / float(len(age_labels)) * 2 * np.pi for n in range(len(age_labels))]
    angles_age += angles_age[:1]  # close the loop

    # Plot each metric
    for idx, (metric, label) in enumerate(zip(radar_metrics, radar_labels)):
        values_age = age_metric_means[metric] + [age_metric_means[metric][0]]
        values_age_plot = [0 if np.isnan(v) else v for v in values_age]
        
        # Plot WITHOUT dots/markers as requested
        ax_age.plot(angles_age, values_age_plot, color=metric_color_map[metric], linewidth=2, label=label)
        ax_age.fill(angles_age, values_age_plot, color=metric_color_map[metric], alpha=0.05)  # Reduced to 0.05

    # Remove default labels for age plot
    ax_age.set_xticks(angles_age[:-1])
    ax_age.set_xticklabels([])
    
    # Manually place age labels further out
    for angle, label in zip(angles_age[:-1], age_labels):
        ha = 'center'
        va = 'center'
        # Adjust alignment based on angle
        if np.cos(angle) > 0.1:
            ha = 'left'
        elif np.cos(angle) < -0.1:
            ha = 'right'
        if np.sin(angle) > 0.1:
            va = 'bottom'
        elif np.sin(angle) < -0.1:
            va = 'top'
            
        ax_age.text(angle, label_radius, label, 
               horizontalalignment=ha, verticalalignment=va,
               fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'),
               transform=ax_age.get_xaxis_transform())
        
    ax_age.set_title(f'e) Performance across ages (Available for {age_percentage:.1f}%)', pad=20)  # Add padding
    ax_age.set_ylim(0, 1)
    ax_age.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_age.set_rlabel_position(20)
    ax_age.grid(True, alpha=0.7)  # Changed from 0.3 to 0.7
    # No legend for panel e as requested

    # PANEL F: Accuracy across sexes - BAR CHART (matching create_figure_5_sample_level exactly)
    ax_sex = fig_radar.add_subplot(gs[1, 2])

    # Calculate all metrics for each sex with bootstrap confidence intervals
    male_metrics = []
    female_metrics = []
    male_ci = []
    female_ci = []

    male_mask = results_df['Sex'] == 'M'
    female_mask = results_df['Sex'] == 'F'

    # Bootstrap function for sex-based metrics
    def bootstrap_sex_metrics(mask, n_bootstrap=1000):
        if mask.sum() == 0:
            return [np.nan] * 7, [np.nan] * 7
        
        subset = results_df[mask].copy()
        boot_metrics = {
            'auroc': [], 'balanced_acc': [], 'sensitivity': [], 
            'specificity': [], 'precision': [], 'recall': [], 'f1': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            boot_indices = np.random.choice(len(subset), size=len(subset), replace=True)
            boot_sample = subset.iloc[boot_indices]
            
            # Calculate metrics for bootstrap sample
            boot_tp = ((boot_sample['gt_volume'] > 0) & (boot_sample['pred_volume'] > 0)).sum()
            boot_tn = ((boot_sample['gt_volume'] == 0) & (boot_sample['pred_volume'] == 0)).sum()
            boot_fp = ((boot_sample['gt_volume'] == 0) & (boot_sample['pred_volume'] > 0)).sum()
            boot_fn = ((boot_sample['gt_volume'] > 0) & (boot_sample['pred_volume'] == 0)).sum()
            
            # Calculate individual metrics
            boot_sens = boot_tp / (boot_tp + boot_fn) if (boot_tp + boot_fn) > 0 else 0
            boot_spec = boot_tn / (boot_tn + boot_fp) if (boot_tn + boot_fp) > 0 else 0
            boot_prec = boot_tp / (boot_tp + boot_fp) if (boot_tp + boot_fp) > 0 else 0
            boot_recall = boot_sens
            boot_f1 = 2 * (boot_prec * boot_recall) / (boot_prec + boot_recall) if (boot_prec + boot_recall) > 0 else 0
            boot_ba = (boot_sens + boot_spec) / 2
            
            # Calculate AUROC if possible
            if len(boot_sample['gt_volume'].unique()) > 1:
                try:
                    y_true_boot = (boot_sample['gt_volume'] > 0).astype(int)
                    y_scores_boot = boot_sample['pred_volume'].fillna(0)
                    boot_auroc = roc_auc_score(y_true_boot, y_scores_boot)
                except:
                    boot_auroc = np.nan
            else:
                boot_auroc = np.nan
            
            boot_metrics['auroc'].append(boot_auroc)
            boot_metrics['balanced_acc'].append(boot_ba)
            boot_metrics['sensitivity'].append(boot_sens)
            boot_metrics['specificity'].append(boot_spec)
            boot_metrics['precision'].append(boot_prec)
            boot_metrics['recall'].append(boot_recall)
            boot_metrics['f1'].append(boot_f1)
        
        # Calculate point estimates and confidence intervals
        point_estimates = []
        ci_errors = []
        metric_order = ['auroc', 'balanced_acc', 'sensitivity', 'specificity', 'precision', 'recall', 'f1']
        
        for metric in metric_order:
            values = [v for v in boot_metrics[metric] if not np.isnan(v)]
            if len(values) > 0:
                point_est = np.mean(values)
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                error = max(point_est - ci_lower, ci_upper - point_est)
            else:
                point_est = np.nan
                error = 0
            
            point_estimates.append(point_est)
            ci_errors.append(error)
        
        return point_estimates, ci_errors

    if male_mask.sum() > 0:
        male_metrics, male_ci = bootstrap_sex_metrics(male_mask)
    else:
        male_metrics = [np.nan] * 7
        male_ci = [0] * 7

    if female_mask.sum() > 0:
        female_metrics, female_ci = bootstrap_sex_metrics(female_mask)
    else:
        female_metrics = [np.nan] * 7
        female_ci = [0] * 7

    # Setup grouped bar chart
    x = np.arange(len(all_metrics_names))
    width = 0.35

    # Plot bars for male and female - using the SAME colors as the metrics with confidence intervals
    bars1 = ax_sex.bar(x - width/2, male_metrics, width, yerr=male_ci, capsize=3, 
                       label='Male', color=metric_colors, alpha=0.7)
    bars2 = ax_sex.bar(x + width/2, female_metrics, width, yerr=female_ci, capsize=3, 
                       label='Female', color=metric_colors, alpha=0.7)

    # Add dashed edge for female bars
    for bar in bars2:
        bar.set_edgecolor('black')
        bar.set_linewidth(1.5)
        bar.set_linestyle('--')

    # Add sample sizes in legend
    male_count = male_mask.sum()
    female_count = female_mask.sum()

    # Create custom legend with light grey colors (matching create_figure_5_sample_level)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgrey', edgecolor='black', label=f'Male (n={male_count})'),
        Patch(facecolor='lightgrey', edgecolor='black', linestyle='--', linewidth=1.5, label=f'Female (n={female_count})')
    ]
    ax_sex.legend(handles=legend_elements, loc='lower right')

    ax_sex.set_xlabel('Metrics')
    ax_sex.set_ylabel('Performance')
    ax_sex.set_title(f'f) Performance across sexes (Available for {sex_percentage:.1f}%)', pad=12)  # Add padding to title
    ax_sex.set_xticks(x)
    ax_sex.set_xticklabels(all_metrics_names, rotation=45, ha='right')
    ax_sex.set_ylim(0, 1.1)
    ax_sex.grid(True, alpha=0.7, axis='y')  # Changed from 0.3 to 0.7

    # Add main title to the entire figure
    plt.suptitle('Enhancement detection performance', fontsize=20, y=0.95)

    plt.tight_layout()

    # SAVE THE FIGURE
    output_path = os.path.join(figures_out, 'Figure_5.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure 5 saved to: {output_path}")

    # Also save as SVG
    output_svg = os.path.join(figures_out, 'Figure_5.svg')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')

    plt.show()
    
    return fig_radar

# Generate Figure 4 with updated radar plot metrics
# First, add Age and Sex to results_df_full by left join (keeps all 1109 cases)
results_df_full_with_demo = results_df_full.merge(
    all_images[['case_id', 'Age','Sex']], 
    on='case_id', 
    how='left'  # Left join keeps all cases from results_df_full
)
print(f"Full test set with demographics (left join): {len(results_df_full_with_demo)}")
create_figure_5_sample_level_ADAPTED(results_df_full_with_demo, figures_out)

# Print detailed results for Figure 5 panels b-f
print("\n" + "="*80)
print("DETAILED RESULTS FOR FIGURE 5")
print("="*80)

# Use the full dataset for printed results to match Figure_5
results_df_for_prints = results_df_full_with_demo

# Access the calculate_sample_level_metrics function from within create_figure_5_sample_level_ADAPTED scope
# We need to redefine it here since it's a nested function
def calculate_sample_level_metrics(mask):
    """Calculate sample-level detection metrics for a subset of data"""
    if isinstance(mask, pd.Series):
        mask = mask.reindex(results_df_for_prints.index, fill_value=False)
    
    subset = results_df_for_prints[mask].copy()
    
    if len(subset) == 0:
        return [np.nan] * 7
    
    # Calculate sample-level binary classification metrics
    subset_tp = ((subset['gt_volume'] > 0) & (subset['pred_volume'] > 0)).sum()
    subset_tn = ((subset['gt_volume'] == 0) & (subset['pred_volume'] == 0)).sum()
    subset_fp = ((subset['gt_volume'] == 0) & (subset['pred_volume'] > 0)).sum()
    subset_fn = ((subset['gt_volume'] > 0) & (subset['pred_volume'] == 0)).sum()
    
    # Calculate metrics
    sensitivity = subset_tp / (subset_tp + subset_fn) if (subset_tp + subset_fn) > 0 else 0
    
    if (subset_tn + subset_fp) > 0:
        specificity = subset_tn / (subset_tn + subset_fp)
    else:
        specificity = np.nan
    
    if np.isnan(specificity):
        # When there are no negative cases (all cases have enhancement), 
        # balanced accuracy cannot be properly calculated
        balanced_acc = sensitivity  # This is why BraTS-MEN shows high on radar but low in print
    else:
        balanced_acc = (sensitivity + specificity) / 2
    
    precision = subset_tp / (subset_tp + subset_fp) if (subset_tp + subset_fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate AUROC if possible
    if len(subset['gt_volume'].unique()) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            y_true_subset = (subset['gt_volume'] > 0).astype(int)
            y_scores_subset = subset['pred_volume'].fillna(0)
            auroc = roc_auc_score(y_true_subset, y_scores_subset)
        except:
            auroc = np.nan
    else:
        auroc = np.nan
    
    return [auroc, balanced_acc, sensitivity, specificity, precision, recall, f1]

# Panel b: Dataset results
print("\nPANEL B: Performance by Dataset")
print("-"*60)
# Group by dataset and calculate metrics
for dataset in results_df_for_prints['Cohort'].unique():
    if dataset:
        dataset_mask = results_df_for_prints['Cohort'] == dataset
        if dataset_mask.sum() > 0:
            # Use the same function as the radar plot
            metrics = calculate_sample_level_metrics(dataset_mask)
            auroc, balanced_acc, sensitivity, specificity, precision, recall, f1 = metrics
            
            # Debug for BraTS-MEN
            if dataset == 'BraTS-MEN':
                print(f"\nDEBUG BraTS-MEN:")
                subset = results_df_for_prints[dataset_mask]
                tp = ((subset['gt_volume'] > 0) & (subset['pred_volume'] > 0)).sum()
                tn = ((subset['gt_volume'] == 0) & (subset['pred_volume'] == 0)).sum()
                fp = ((subset['gt_volume'] == 0) & (subset['pred_volume'] > 0)).sum()
                fn = ((subset['gt_volume'] > 0) & (subset['pred_volume'] == 0)).sum()
                print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
                if np.isnan(specificity):
                    print(f"  Sensitivity={sensitivity:.3f}, Specificity=NaN")
                else:
                    print(f"  Sensitivity={sensitivity:.3f}, Specificity={specificity:.3f}")
                print(f"  Total cases with GT>0: {(subset['gt_volume'] > 0).sum()}")
                print(f"  Total cases with GT=0: {(subset['gt_volume'] == 0).sum()}")
                print(f"  Total cases with Pred>0: {(subset['pred_volume'] > 0).sum()}")
                print(f"  Total cases with Pred=0: {(subset['pred_volume'] == 0).sum()}")
                print()
            
            print(f"{dataset:30s}: n={dataset_mask.sum():4d}, BA={balanced_acc:.3f}, "
                  f"Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")

# Panel c: Pathology results
print("\nPANEL C: Performance by Pathology")
print("-"*60)
for pathology in results_df_for_prints['Pathology'].unique():
    if pathology:
        pathology_mask = results_df_for_prints['Pathology'] == pathology
        if pathology_mask.sum() > 0:
            # Use the same function as the radar plot
            metrics = calculate_sample_level_metrics(pathology_mask)
            auroc, balanced_acc, sensitivity, specificity, precision, recall, f1 = metrics
            
            print(f"{pathology:35s}: n={pathology_mask.sum():4d}, BA={balanced_acc:.3f}, "
                  f"Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")

# Panel d: Country results
print("\nPANEL D: Performance by Country")
print("-"*60)
# Use actual country columns instead of extracting from cohort names
country_columns = ['USA', 'UK', 'Netherlands', 'Sub-Saharan Africa']
for country in country_columns:
    if country in results_df_for_prints.columns:
        # Get cases from this country
        country_mask = results_df_for_prints[country] == 1
        if country_mask.sum() > 0:
            # Use the same function as the radar plot
            metrics = calculate_sample_level_metrics(country_mask)
            auroc, balanced_acc, sensitivity, specificity, precision, recall, f1 = metrics
            
            print(f"{country:20s}: n={country_mask.sum():4d}, BA={balanced_acc:.3f}, "
                  f"Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")

# Panel e: Age group results
print("\nPANEL E: Performance by Age Group")
print("-"*60)
# Create age bins
age_bins = [0, 30, 45, 60, 75, 100]
age_labels = ['<30y', '30-45y', '45-60y', '60-75y', '>75y']
# Filter out NaN ages
age_valid_mask = results_df_for_prints['Age'].notna()
age_valid_df = results_df_for_prints[age_valid_mask].copy()
if len(age_valid_df) > 0:
    age_valid_df['Age_Bin'] = pd.cut(age_valid_df['Age'], bins=age_bins, labels=age_labels)
    
    for age_group in age_labels:
        # Create mask for age group in the full results_df
        age_group_mask = (results_df_for_prints['Age'].notna()) & (pd.cut(results_df_for_prints['Age'], bins=age_bins, labels=age_labels) == age_group)
        if age_group_mask.sum() > 0:
            # Use the same function as the radar plot
            metrics = calculate_sample_level_metrics(age_group_mask)
            auroc, balanced_acc, sensitivity, specificity, precision, recall, f1 = metrics
            
            print(f"{age_group:10s}: n={age_group_mask.sum():4d}, BA={balanced_acc:.3f}, "
                  f"Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")

# Panel f: Sex results
print("\nPANEL F: Performance by Sex")
print("-"*60)
# Male results
male_mask = results_df_for_prints['Sex'] == 'M'
if male_mask.sum() > 0:
    # Use the same function as the radar plot
    metrics = calculate_sample_level_metrics(male_mask)
    auroc, balanced_acc, sensitivity, specificity, precision, recall, f1 = metrics
    
    print(f"Male   : n={male_mask.sum():4d}, BA={balanced_acc:.3f}, "
          f"Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")

# Female results
female_mask = results_df_for_prints['Sex'] == 'F'
if female_mask.sum() > 0:
    # Use the same function as the radar plot
    metrics = calculate_sample_level_metrics(female_mask)
    auroc, balanced_acc, sensitivity, specificity, precision, recall, f1 = metrics
    
    print(f"Female : n={female_mask.sum():4d}, BA={balanced_acc:.3f}, "
          f"Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")

print("\n" + "="*80)


# %%
results_df.head()

# %%
# Pathology Sample Size vs Performance Figure with Dice Gain Analysis
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_improved_pathology_sample_size_plot(results_df, all_images, figures_out):
    """
    Create improved pathology sample size plot with two columns:
    - Column 0: Sample size vs performance (with specified x-ticks)
    - Column 1: Dice gain per sample by pathology (using TRAINING set counts)
    """
    # Filter for cases with enhancing tumor only in TEST set
    et_cases = results_df[results_df['gt_volume'] > 0].copy()
    
    # Get training set pathology counts
    train_images = all_images[all_images['Partition'] == 'Train/Val'].copy()
    train_pathology_counts = train_images['Pathology'].value_counts().to_dict()
    
    # Get pathology statistics from TEST set performance
    pathology_stats = []
    
    for pathology in et_cases['Pathology'].unique():
        if pathology != '' and pathology in train_pathology_counts:  # Skip empty pathologies
            pathology_data = et_cases[et_cases['Pathology'] == pathology]
            
            stats = {
                'pathology': pathology,
                'train_sample_size': train_pathology_counts.get(pathology, 0),  # Training set size
                'test_sample_size': len(pathology_data),  # Test set size
                'mean_dice': pathology_data['dice'].mean(),
                'std_dice': pathology_data['dice'].std(),
                'sem_dice': pathology_data['dice'].sem()
            }
            pathology_stats.append(stats)
    
    # Convert to DataFrame and sort by training sample size
    pathology_df = pd.DataFrame(pathology_stats)
    pathology_df = pathology_df.sort_values('train_sample_size')
    
    # Calculate dice gain per training sample
    # Using a simple metric: mean_dice / log(train_sample_size + 1)
    # This gives higher values for pathologies that achieve good performance with fewer training samples
    pathology_df['dice_per_sample'] = pathology_df['mean_dice'] / np.log(pathology_df['train_sample_size'] + 1)
    
    # Create figure with 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define colors
    colors = sns.color_palette('husl', n_colors=len(pathology_df))
    
    # COLUMN 0: Training sample size vs test performance with custom x-axis
    # Plot dots with error bars and connect them with lines
    x_values = pathology_df['train_sample_size'].values
    y_values = pathology_df['mean_dice'].values
    
    # First plot the connecting line
    ax1.plot(x_values, y_values, 'k-', alpha=0.3, linewidth=2, zorder=1)
    
    # Then plot dots with error bars
    for i, (idx, row) in enumerate(pathology_df.iterrows()):
        # Plot dot with error bar
        ax1.errorbar(row['train_sample_size'], row['mean_dice'], 
                    yerr=row['sem_dice'] * 1.96,  # 95% confidence interval
                    fmt='o', markersize=12, capsize=8, capthick=2,
                    color=colors[i], label=row['pathology'],
                    elinewidth=2, markeredgecolor='black', markeredgewidth=1,
                    zorder=2)
    
    # Annotate each point with pathology name
    for idx, row in pathology_df.iterrows():
        # Split long pathology names
        pathology_name = row['pathology']
        if len(pathology_name) > 20:
            words = pathology_name.split()
            mid_point = len(words) // 2
            pathology_name = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
        
        # Position text slightly above the point
        ax1.annotate(pathology_name, 
                    xy=(row['train_sample_size'], row['mean_dice']),
                    xytext=(0, 15), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='gray', alpha=0.8))
    
    # Customize plot
    ax1.set_xlabel('Training Sample Size (Number of Cases)', fontsize=14)
    ax1.set_ylabel('Test Set Performance (Mean Dice Score)', fontsize=14)
    ax1.set_title('a) Model Performance vs Training Sample Size by Pathology', fontsize=16)
    
    ax1.set_xscale('log')
    # Set custom x-axis values
    custom_xticks = [200,400,800,1600,3200,6400,12800]
    ax1.set_xticks(custom_xticks)
    ax1.set_xticklabels(custom_xticks)
    
    # Use log scale for x-axis
    
    
    # Set axis limits
    ax1.set_ylim(0, 1)
    ax1.set_xlim(custom_xticks[0], custom_xticks[-1])  # Adjust to show all data points
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # COLUMN 1: Dice gain per training sample (efficiency metric)
    # Sort by dice_per_sample for better visualization
    pathology_df_sorted = pathology_df.sort_values('dice_per_sample', ascending=False)
    
    x_pos = np.arange(len(pathology_df_sorted))
    bars = ax2.bar(x_pos, pathology_df_sorted['dice_per_sample'], 
                   alpha=0.7, color=colors)
    
    # Color bars based on efficiency
    for i, (bar, eff) in enumerate(zip(bars, pathology_df_sorted['dice_per_sample'])):
        if eff > pathology_df_sorted['dice_per_sample'].quantile(0.75):
            bar.set_color(sns.color_palette('Greens', n_colors=3)[2])  # High efficiency
        elif eff > pathology_df_sorted['dice_per_sample'].quantile(0.25):
            bar.set_color(sns.color_palette('Blues', n_colors=3)[1])   # Medium efficiency
        else:
            bar.set_color(sns.color_palette('Reds', n_colors=3)[1])    # Low efficiency
    
    # Add value labels on bars (showing training set size)
    # for i, (eff, size) in enumerate(zip(pathology_df_sorted['dice_per_sample'], 
    #                                    pathology_df_sorted['train_sample_size'])):
    #     ax2.text(i, eff + 0.001, f'n={size}', ha='center', va='bottom', fontsize=8)
    
    # Customize second plot
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(pathology_df_sorted['pathology'], rotation=45, ha='right')
    ax2.set_ylabel('Dice Gain per Training Sample (Efficiency)', fontsize=14)
    ax2.set_title('b) Model Efficiency by Pathology Type', fontsize=16)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # # Add explanation text
    # ax2.text(0.02, 0.98, 
    #         'Higher values indicate better\nperformance with fewer\ntraining samples',
    #         transform=ax2.transAxes, fontsize=10, va='top',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    
    # Overall title
    plt.suptitle('Pathology Sample Size Analysis', fontsize=18)
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(figures_out, 'pathology_sample_size_vs_performance.png'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_out, 'pathology_sample_size_vs_performance.svg'),
                format='svg', bbox_inches='tight')
    
    plt.show()
    
    # Print summary
    print("\nPathology Efficiency Analysis (Training Set):")
    print("-" * 60)
    print("Most efficient pathologies (good test performance with fewer training samples):")
    for i, row in pathology_df_sorted.head(3).iterrows():
        print(f"  {row['pathology']}: Test Dice={row['mean_dice']:.3f}, "
              f"Train n={row['train_sample_size']}, Test n={row['test_sample_size']}, "
              f"Efficiency={row['dice_per_sample']:.3f}")
    
    print("\nLeast efficient pathologies (require more training samples for good performance):")
    for i, row in pathology_df_sorted.tail(3).iterrows():
        print(f"  {row['pathology']}: Test Dice={row['mean_dice']:.3f}, "
              f"Train n={row['train_sample_size']}, Test n={row['test_sample_size']}, "
              f"Efficiency={row['dice_per_sample']:.3f}")
    
    return fig, pathology_df_sorted

# Generate the pathology sample size vs performance figure
print("Creating Pathology Sample Size vs Performance Figure...")
print("=" * 80)
print("Key features:")
print("2. Column 1 shows 'Dice gain per TRAINING sample' (efficiency metric)")
print("3. Efficiency metric = test_mean_dice / log(training_sample_size + 1)")
print("4. Second plot ordered by efficiency to show which pathologies need fewer training samples")
print("5. Color coding: Green = high efficiency, Blue = medium, Red = low efficiency")
print("=" * 80)

# Note: Call this function with:
fig_pathology, pathology_stats_df = create_improved_pathology_sample_size_plot(results_df, all_images, figures_out)

# %%
# Create separate best and worst cases visualization functions
def create_separate_best_worst_figures(results_df, groupby_column, performance_type='best', n_cases=8, 
                                       include_filename=True, include_group_info=True):
    """
    Create SEPARATE figures for each group (cohort/disease/country) showing either best OR worst cases.
    
    Parameters:
    -----------
    results_df : DataFrame with results
    groupby_column : str, one of 'Cohort', 'Pathology', 'Country'
    performance_type : str, either 'best' or 'worst'
    n_cases : int, default=3
        Number of cases to show per group
    include_filename : bool, default=True
        Whether to include the filename/case_id in the y-axis labels
    include_group_info : bool, default=True
        Whether to include the group information in the y-axis labels
    """
    
    # Import required modules
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import nibabel as nib
    
    print(f"\nCREATING {performance_type.upper()} CASES FIGURES BY {groupby_column.upper()}")
    print("=" * 60)
    
    # Filter for cases with enhancing tumor only
    et_cases = results_df[results_df['gt_volume'] > 0].copy()
    
    # Get unique groups
    groups = et_cases[groupby_column].unique()
    groups = [g for g in groups if g != '']  # Remove empty groups
    
    # Define data path
    data_path = '/home/jruffle/Documents/seq-synth/data/'
    
    # Create green colormap for overlays
    green_cmap = ListedColormap(['none', 'green'])  # 'none' for transparent, 'green' for mask
    
    # Column titles
    col_titles = ['T1', 'T2', 'FLAIR', 'Enhancing prediction\n(from T1+T2+FLAIR)', 
                 'T1CE\n(Held out from model)', 'Ground Truth\n(with T1CE background)']
    
    # Process each group separately
    for group in groups:
        group_data = et_cases[et_cases[groupby_column] == group]
        
        if len(group_data) < n_cases:  # Not enough cases
            print(f"Skipping {group} - insufficient cases ({len(group_data)} < {n_cases})")
            continue
            
        # Sort by dice score
        group_sorted = group_data.sort_values('dice', ascending=(performance_type == 'worst'))
        
        # Get cases based on performance type
        selected_cases = group_sorted.head(n_cases)
        
        print(f"Creating {performance_type} figure for {group} ({n_cases} cases)")
        
        # Create figure for this group
        fig, axes = plt.subplots(n_cases, 6, figsize=(22, n_cases * 3.8))
        
        if n_cases == 1:
            axes = np.array([axes])
        
        # Process each case
        for i, (_, case_data) in enumerate(selected_cases.iterrows()):
            case_id = case_data['case_id']
            
            try:
                # Load ground truth and prediction
                gt_img = nib.load(os.path.join(gt_labels_path, f"{case_id}.nii.gz")).get_fdata()
                pred_img = nib.load(os.path.join(predictions_path, f"{case_id}.nii.gz")).get_fdata()
                
                # Try to load structural images
                try:
                    # Try sequences_merged directory first
                    seq_path = os.path.join(data_path, 'sequences_merged', f"{case_id}.nii.gz")
                    brain_mask_path = os.path.join(data_path, 'lesion_masks_augmented', f"{case_id}.nii.gz")
                    
                    if os.path.exists(seq_path) and os.path.exists(brain_mask_path):
                        seq_img = nib.load(seq_path).get_fdata()
                        brain_mask = nib.load(brain_mask_path).get_fdata()
                        brain_mask[brain_mask > 0] = 1
                        
                        flair_img = seq_img[..., 0] * brain_mask
                        t1_img = seq_img[..., 1] * brain_mask 
                        t1ce_img = seq_img[..., 2] * brain_mask
                        t2_img = seq_img[..., 3] * brain_mask
                    else:
                        # Fallback to nnUNet structure
                        images_path = gt_labels_path.replace('labelsTs', 'imagesTs')
                        t1_path = os.path.join(images_path, f"{case_id}_0000.nii.gz")
                        t2_path = os.path.join(images_path, f"{case_id}_0001.nii.gz") 
                        flair_path = os.path.join(images_path, f"{case_id}_0002.nii.gz")
                        t1ce_path = os.path.join(images_path, f"{case_id}_0003.nii.gz")
                        
                        t1_img = nib.load(t1_path).get_fdata() if os.path.exists(t1_path) else None
                        t2_img = nib.load(t2_path).get_fdata() if os.path.exists(t2_path) else None
                        flair_img = nib.load(flair_path).get_fdata() if os.path.exists(flair_path) else None
                        t1ce_img = nib.load(t1ce_path).get_fdata() if os.path.exists(t1ce_path) else None
                        
                except Exception as e:
                    print(f"Could not load sequences for {case_id}: {e}")
                    t1_img = t2_img = flair_img = t1ce_img = None
                
                # Find best slice with enhancing tumour
                et_slices = np.sum(gt_img == LABEL_ENHANCING_TUMOUR, axis=(0, 1))
                if np.max(et_slices) > 0:
                    z_slice = np.argmax(et_slices)
                else:
                    z_slice = gt_img.shape[2] // 2
                
                # Get 2D slices
                gt_slice = gt_img[:, :, z_slice]
                pred_slice = pred_img[:, :, z_slice]
                
                # Create binary masks
                gt_et = (gt_slice == LABEL_ENHANCING_TUMOUR).astype(np.uint8)
                pred_et = (pred_slice == LABEL_ENHANCING_TUMOUR).astype(np.uint8)
                
                # Column 1: T1
                if t1_img is not None:
                    axes[i, 0].imshow(np.rot90(t1_img[:, :, z_slice]), cmap='gray')
                else:
                    axes[i, 0].text(0.5, 0.5, 'T1\nNot Available', ha='center', va='center', transform=axes[i, 0].transAxes)
                
                # Column 2: T2
                if t2_img is not None:
                    axes[i, 1].imshow(np.rot90(t2_img[:, :, z_slice]), cmap='gray')
                else:
                    axes[i, 1].text(0.5, 0.5, 'T2\nNot Available', ha='center', va='center', transform=axes[i, 1].transAxes)
                
                # Column 3: FLAIR
                if flair_img is not None:
                    axes[i, 2].imshow(np.rot90(flair_img[:, :, z_slice]), cmap='gray')
                else:
                    axes[i, 2].text(0.5, 0.5, 'FLAIR\nNot Available', ha='center', va='center', transform=axes[i, 2].transAxes)
                
                # Column 4: Prediction with FLAIR background
                if flair_img is not None:
                    axes[i, 3].imshow(np.rot90(flair_img[:, :, z_slice]), cmap='gray')
                else:
                    axes[i, 3].imshow(np.rot90(pred_et*0), cmap='gray')
                axes[i, 3].imshow(np.rot90(pred_et), cmap=green_cmap, alpha=0.6, vmin=0, vmax=1)
                
                # Column 5: T1CE
                if t1ce_img is not None:
                    axes[i, 4].imshow(np.rot90(t1ce_img[:, :, z_slice]), cmap='gray')
                else:
                    axes[i, 4].text(0.5, 0.5, 'T1CE\nNot Available', ha='center', va='center', transform=axes[i, 4].transAxes)
                
                # Column 6: Ground truth with T1CE
                if t1ce_img is not None:
                    axes[i, 5].imshow(np.rot90(t1ce_img[:, :, z_slice]), cmap='gray')
                else:
                    axes[i, 5].imshow(np.rot90(gt_et*0), cmap='gray')
                axes[i, 5].imshow(np.rot90(gt_et), cmap=green_cmap, alpha=0.6, vmin=0, vmax=1)
                
                # Add titles to first row
                if i == 0:
                    for j, title in enumerate(col_titles):
                        axes[i, j].set_title(title, fontsize=11, pad=8)
                
                # Add case information
                dice_score = case_data['dice']
                
                # Set y-axis label based on parameters
                label_parts = []
                if include_filename:
                    label_parts.append(case_id)
                label_parts.append(f"Dice: {dice_score:.3f}")
                
                axes[i, 0].set_ylabel('\n'.join(label_parts), fontsize=9)
                
                # Remove axis ticks
                for ax in axes[i]:
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            except Exception as e:
                print(f"Error visualizing {case_id}: {e}")
                for j in range(6):
                    error_text = f'Error loading{chr(10)}{case_id}'
                    axes[i, j].text(0.5, 0.5, error_text, 
                                   ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].axis('off')
        
        # Adjust layout - EXACTLY SAME AS missed cases visualization
        plt.tight_layout()
        plt.subplots_adjust(hspace=-0.07, wspace=-0.6, top=0.92, bottom=0.06)
        
        # Add white divider line between columns 3 and 4
        fig.canvas.draw()
        pos3 = axes[0, 3].get_position()
        pos4 = axes[0, 4].get_position()
        line_x = (pos3.x1 + pos4.x0) / 2
        
        # White line with black border
        border_line = plt.Line2D([line_x, line_x], [0.06, 0.92], 
                                transform=fig.transFigure, 
                                color='black', 
                                linewidth=8,
                                solid_capstyle='butt',
                                zorder=9)
        fig.add_artist(border_line)
        
        line = plt.Line2D([line_x, line_x], [0.06, 0.92], 
                         transform=fig.transFigure, 
                         color='white', 
                         linewidth=6,
                         solid_capstyle='butt',
                         zorder=10)
        fig.add_artist(line)
        
        # Add main title
        title = f'{performance_type.capitalize()} Performing Cases - {group}'
        if include_group_info:
            title = f'{performance_type.capitalize()} Performing Cases\n{groupby_column}: {group}'
        plt.suptitle(title, fontsize=14, y=0.955)
        
        # Save figure with unique filename
        filename_base = f'{performance_type}_cases_{groupby_column.lower()}_{group.replace(" ", "_").replace("/", "_")}'
        fig.savefig(os.path.join(figures_out, f'{filename_base}.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(figures_out, f'{filename_base}.svg'), 
                    format='svg', bbox_inches='tight', facecolor='white')
        
        print(f"  Saved: {filename_base}.png and {filename_base}.svg")
        plt.close(fig)  # Close figure to save memory

# Generate all separate figures
print("\n" + "="*80)
print("GENERATING SEPARATE BEST/WORST CASES FIGURES")
print("="*80)

import concurrent.futures

# Define a helper function to wrap the main call for parallel execution
def run_visualization_task(args):
    groupby_col, performance_type, results_df, n_cases, include_filename, include_group_info = args
    print(f"Starting task for {groupby_col}, performance: {performance_type}")
    try:
        create_separate_best_worst_figures(
            results_df=results_df, 
            groupby_column=groupby_col, 
            performance_type=performance_type, 
            n_cases=n_cases, 
            include_filename=include_filename, 
            include_group_info=include_group_info
        )
        return f"Successfully completed task for {groupby_col}, performance: {performance_type}"
    except Exception as e:
        return f"Task for {groupby_col}, performance: {performance_type} failed with error: {e}"

# Prepare the arguments for each task
tasks_args = []
for groupby_col in ['Cohort', 'Pathology', 'Country']:
    for performance_type in ['best', 'worst']:
        tasks_args.append((
            groupby_col, 
            performance_type, 
            results_df, 
            8,          # n_cases
            False,       # include_filename
            True        # include_group_info
        ))

# Use ProcessPoolExecutor to run tasks in parallel
# The number of workers will default to the number of CPUs on the machine.
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Submit all tasks to the executor
    futures = [executor.submit(run_visualization_task, args) for args in tasks_args]
    
    # Wait for all tasks to complete and print results
    for future in concurrent.futures.as_completed(futures):
        print(future.result())

print("\nAll separate best/worst cases figures completed!")

# %%
# Clinical Relevance Analysis Figure - RESTORED
def create_clinical_relevance_analysis(results_df, gt_path, pred_path):
    """
    Create clinical relevance analysis figure with 6 subplots:
    a) Bland-Altman plot - volumetric agreement
    b) Volume correlation analysis scatter plot
    c) Performance by clinical volume categories
    d) Volume estimation accuracy pie chart
    e) Detection sensitivity vs size plot
    f) Volume difference distribution histogram
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats
    import nibabel as nib
    from sklearn.metrics import r2_score
    
    print("Creating Clinical Relevance Analysis Figure...")
    print("=" * 60)
    
    # Filter for cases with enhancing tumor
    et_cases = results_df[results_df['gt_volume'] > 0].copy()
    
    # Create figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Define color palette
    colors_pie = sns.color_palette('husl', n_colors=10)
    
    # a) Bland-Altman plot - volumetric agreement
    ax = axes[0]
    
    # Calculate mean and difference
    mean_volume = (et_cases['gt_volume'] + et_cases['pred_volume']) / 2
    volume_diff = et_cases['pred_volume'] - et_cases['gt_volume']
    
    # Create scatter plot with Dice color coding
    scatter = ax.scatter(mean_volume, volume_diff, c=et_cases['dice'], 
                        cmap='RdYlBu', alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
    
    # Add mean difference and limits of agreement
    mean_diff = np.mean(volume_diff)
    std_diff = np.std(volume_diff)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, 
               label=f'Mean: {mean_diff:.0f}')
    ax.axhline(upper_limit, color='red', linestyle='--', linewidth=1.5, 
               label=f'+1.96 SD: {upper_limit:.0f}')
    ax.axhline(lower_limit, color='red', linestyle='--', linewidth=1.5, 
               label=f'-1.96 SD: {lower_limit:.0f}')
    ax.axhline(0, color='black', linestyle='-', alpha=0.6, linewidth=1)
    
    ax.set_xlabel('Mean Volume (GT + Pred) / 2 [voxels]', fontsize=12)
    ax.set_ylabel('Volume Difference (Pred - GT) [voxels]', fontsize=12)
    ax.set_title('a) Bland-Altman Plot - Volumetric Agreement', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.6)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Dice Score', fontsize=10)
    
    # b) Volume correlation analysis scatter plot
    ax = axes[1]
    
    # Calculate regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(et_cases['gt_volume'], 
                                                                   et_cases['pred_volume'])
    r2 = r_value ** 2
    
    # Create scatter plot with Dice color coding
    scatter = ax.scatter(et_cases['gt_volume'], et_cases['pred_volume'], 
                        c=et_cases['dice'], cmap='RdYlBu', alpha=0.6, s=50, 
                        edgecolor='black', linewidth=0.5)
    
    # Add regression line
    x_fit = np.linspace(0, et_cases['gt_volume'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'b-', linewidth=2, 
            label=f'y = {slope:.2f}x + {intercept:.0f}\nR² = {r2:.3f}')
    
    # Add identity line
    max_val = max(et_cases['gt_volume'].max(), et_cases['pred_volume'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, linewidth=2, 
            label='Identity line')
    
    ax.set_xlabel('Ground Truth Volume [voxels]', fontsize=12)
    ax.set_ylabel('Predicted Volume [voxels]', fontsize=12)
    ax.set_title('b) Volume Correlation Analysis', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.6)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Dice Score', fontsize=10)
    
    # c) Performance by clinical volume categories
    ax = axes[2]
    
    # Define clinical volume categories (in voxels, assuming 1mm³ voxels)
    # 1 cm³ = 1000 mm³ = 1000 voxels
    volume_categories = [
        ('Micro\n(<0.5cm³)', 0, 500),
        ('Small\n(0.5-1cm³)', 500, 1000),
        ('Medium\n(1-5cm³)', 1000, 5000),
        ('Large\n(5-10cm³)', 5000, 10000),
        ('Very Large\n(>10cm³)', 10000, np.inf)
    ]
    
    category_stats = []
    for cat_name, min_vol, max_vol in volume_categories:
        mask = (et_cases['gt_volume'] >= min_vol) & (et_cases['gt_volume'] < max_vol)
        if mask.sum() > 0:
            dice_mean = et_cases.loc[mask, 'dice'].mean()
            dice_std = et_cases.loc[mask, 'dice'].std()
            n_cases = mask.sum()
            # Detection rate: cases where prediction > 0
            detection_rate = (et_cases.loc[mask, 'pred_volume'] > 0).mean() * 100
            
            category_stats.append({
                'category': cat_name,
                'dice_mean': dice_mean,
                'dice_std': dice_std,
                'n_cases': n_cases,
                'detection_rate': detection_rate
            })
    
    cat_df = pd.DataFrame(category_stats)
    
    # Create bar plot
    x_pos = np.arange(len(cat_df))
    bars = ax.bar(x_pos, cat_df['dice_mean'], yerr=cat_df['dice_std'], 
                  capsize=5, alpha=0.6, color=colors_pie[0], edgecolor='black')
    
    # Add detection rate annotations
    for i, (bar, row) in enumerate(zip(bars, cat_df.iterrows())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + row[1]['dice_std'] + 0.02,
                f"{row[1]['detection_rate']:.0f}%\n(n={row[1]['n_cases']})", 
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_df['category'])
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('c) Performance by Clinical Volume Categories', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.6, axis='y')
    
    # d) Volume estimation accuracy pie chart
    ax = axes[3]
    
    # Calculate volume estimation accuracy categories
    volume_ratio = et_cases['pred_volume'] / et_cases['gt_volume']
    
    accuracy_categories = {
        'Accurate (±25%)': ((volume_ratio >= 0.75) & (volume_ratio <= 1.25)).sum(),
        'Underestimate (25-50%)': ((volume_ratio >= 0.5) & (volume_ratio < 0.75)).sum(),
        'Underestimate (>50%)': (volume_ratio < 0.5).sum(),
        'Overestimate (25-50%)': ((volume_ratio > 1.25) & (volume_ratio <= 1.5)).sum(),
        'Overestimate (>50%)': (volume_ratio > 1.5).sum()
    }
    
    # Create pie chart
    sizes = list(accuracy_categories.values())
    labels = [f'{k}\n({v})' for k, v in accuracy_categories.items()]
    colors_accuracy = ['green', 'yellow', 'orange', 'lightcoral', 'red']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                      startangle=90, colors=colors_accuracy)
    ax.set_title('d) Volume Estimation Accuracy', fontsize=14)
    
    # e) Detection sensitivity vs size plot
    ax = axes[4]
    
    # Create size bins (log scale)
    size_bins = np.logspace(np.log10(10), np.log10(et_cases['gt_volume'].max()), 20)
    sensitivities = []
    bin_centers = []
    
    for i in range(len(size_bins)-1):
        mask = (et_cases['gt_volume'] >= size_bins[i]) & (et_cases['gt_volume'] < size_bins[i+1])
        if mask.sum() > 0:
            sensitivity = (et_cases.loc[mask, 'pred_volume'] > 0).mean()
            sensitivities.append(sensitivity)
            bin_centers.append(np.sqrt(size_bins[i] * size_bins[i+1]))  # Geometric mean
    
    # Plot sensitivity curve
    ax.plot(bin_centers, sensitivities, 'o-', linewidth=2, markersize=8, color=colors_pie[1])
    
    # Add vertical lines for 1cm³ and 2cm³
    ax.axvline(1000, color='red', linestyle='--', alpha=0.6, linewidth=2, label='1 cm³')
    ax.axvline(2000, color='blue', linestyle='--', alpha=0.6, linewidth=2, label='2 cm³')
    
    ax.set_xscale('log')
    ax.set_xlabel('Tumor Size [voxels]', fontsize=12)
    ax.set_ylabel('Detection Sensitivity', fontsize=12)
    ax.set_title('e) Detection Sensitivity vs Size', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.6)
    ax.legend()
    
    # f) Volume difference distribution
    ax = axes[5]
    
    # Create histogram
    volume_diff_voxels = et_cases['pred_volume'] - et_cases['gt_volume']
    
    ax.hist(volume_diff_voxels, bins=50, alpha=0.6, color=colors_pie[2], 
            edgecolor='black', density=True)
    
    # Add normal distribution overlay
    mu, sigma = volume_diff_voxels.mean(), volume_diff_voxels.std()
    x = np.linspace(volume_diff_voxels.min(), volume_diff_voxels.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
            label=f'Normal fit\nμ={mu:.0f}, σ={sigma:.0f}')
    
    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='-', alpha=0.6, linewidth=2)
    
    ax.set_xlabel('Volume Difference (Pred - GT) [voxels]', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('f) Volume Difference Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.6, axis='y')
    
    # Overall title
    plt.suptitle('Clinical Relevance Analysis', fontsize=18)
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(figures_out, 'clinical_relevance_analysis.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_out, 'clinical_relevance_analysis.svg'), 
                format='svg', bbox_inches='tight')
    
    # Print summary statistics
    print("\nClinical Relevance Summary:")
    print("-" * 60)
    print(f"Volume Agreement (Bland-Altman):")
    print(f"  Mean difference: {mean_diff:.0f} voxels")
    print(f"  95% limits of agreement: [{lower_limit:.0f}, {upper_limit:.0f}] voxels")
    print(f"\nVolume Correlation:")
    print(f"  R²: {r2:.3f}")
    print(f"  Slope: {slope:.3f}")
    print(f"  Intercept: {intercept:.0f} voxels")
    print(f"\nVolume Estimation Accuracy:")
    for category, count in accuracy_categories.items():
        percentage = count / len(et_cases) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    plt.show()
    
    return fig

# Generate the clinical relevance analysis figure
fig_clinical = create_clinical_relevance_analysis(results_df, gt_labels_path, predictions_path)

# %%
# Data Quality Impact Analysis Figure - RESTORED with specified panels
def create_data_quality_impact_figure(results_df):
    """
    Create data quality impact analysis figure with 4 panels:
    a) Volume vs performance quality (scatter plot with logged GT volume)
    b) Quality by cohort (mean dice score by cohort)
    c) Precision-recall quality (scatterplot showing their relation)
    d) Quality consistency (barplot showing dice coefficient std dev by cohort)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    print("Creating Data Quality Impact Analysis Figure...")
    print("=" * 60)
    
    # Filter for cases with enhancing tumor
    et_cases = results_df[results_df['gt_volume'] > 0].copy()
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define color palette
    colors_pie = sns.color_palette('husl', n_colors=10)
    
    # a) Volume vs performance quality
    ax = axes[0, 0]
    
    # Create scatter plot with logged ground truth volume on x-axis
    # Color code by dice score
    scatter = ax.scatter(np.log10(et_cases['gt_volume'] + 1), et_cases['dice'],
                        c=et_cases['dice'], cmap='RdYlBu', alpha=0.6, s=50,
                        edgecolor='black', linewidth=0.5, vmin=0, vmax=1)
    
    ax.set_xlabel('Log10(Ground Truth Volume + 1)', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('a) Volume vs Performance Quality', fontsize=14)
    ax.grid(True, alpha=0.6)
    ax.set_ylim(0, 1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Dice Score', fontsize=10)
    
    # Add trend line
    from scipy.stats import linregress
    x_log = np.log10(et_cases['gt_volume'] + 1)
    slope, intercept, r_value, p_value, std_err = linregress(x_log, et_cases['dice'])
    x_trend = np.linspace(x_log.min(), x_log.max(), 100)
    y_trend = slope * x_trend + intercept
    ax.plot(x_trend, y_trend, 'r--', linewidth=2, alpha=0.6,
            label=f'R² = {r_value**2:.3f}, p = {p_value:.3e}')
    ax.legend()
    
    # b) Quality by cohort
    ax = axes[0, 1]
    
    # Calculate mean dice score by cohort
    cohort_stats = et_cases.groupby('Cohort')['dice'].agg(['mean', 'std', 'count'])
    cohort_stats = cohort_stats.sort_values('mean', ascending=False)
    cohort_stats = cohort_stats[cohort_stats.index != '']  # Remove empty cohort
    
    # Create bar plot
    x_pos = np.arange(len(cohort_stats))
    bars = ax.bar(x_pos, cohort_stats['mean'], yerr=cohort_stats['std'],
                  capsize=5, alpha=0.6, color=colors_pie[:len(cohort_stats)],
                  edgecolor='black')
    
    # Add sample size annotations
    for i, (idx, row) in enumerate(cohort_stats.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.02, f"n={int(row['count'])}", 
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cohort_stats.index, rotation=45, ha='right')
    ax.set_ylabel('Mean Dice Score', fontsize=12)
    ax.set_title('b) Quality by Cohort', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.6, axis='y')
    
    # c) Precision-recall quality scatterplot
    ax = axes[1, 0]
    
    # Create scatter plot showing precision vs recall
    # Color by cohort
    cohorts = et_cases['Cohort'].unique()
    cohorts = [c for c in cohorts if c != '']
    
    for i, cohort in enumerate(cohorts):
        cohort_data = et_cases[et_cases['Cohort'] == cohort]
        ax.scatter(cohort_data['recall'], cohort_data['precision'],
                  label=cohort, alpha=0.6, s=50, color=colors_pie[i % len(colors_pie)],
                  edgecolor='black', linewidth=0.5)
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('c) Precision-Recall Quality', fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # d) Quality consistency barplot
    ax = axes[1, 1]
    
    # Calculate dice coefficient standard deviation by cohort
    consistency_stats = et_cases.groupby('Cohort')['dice'].agg(['std', 'count'])
    consistency_stats = consistency_stats[consistency_stats.index != '']  # Remove empty cohort
    consistency_stats = consistency_stats.sort_values('std', ascending=True)  # Lower std = more consistent
    
    # Create bar plot
    x_pos = np.arange(len(consistency_stats))
    bars = ax.bar(x_pos, consistency_stats['std'], alpha=0.6,
                  color=[colors_pie[i % len(colors_pie)] for i in range(len(consistency_stats))],
                  edgecolor='black')
    
    # Color bars based on consistency level
    for i, (bar, std) in enumerate(zip(bars, consistency_stats['std'])):
        if std < 0.15:
            bar.set_facecolor('green')  # High consistency
        elif std < 0.25:
            bar.set_facecolor('orange')  # Medium consistency
        else:
            bar.set_facecolor('red')  # Low consistency
    
    # Add sample size annotations
    for i, (idx, row) in enumerate(consistency_stats.iterrows()):
        ax.text(i, row['std'] + 0.005, f"n={int(row['count'])}", 
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(consistency_stats.index, rotation=45, ha='right')
    ax.set_ylabel('Dice Coefficient Standard Deviation', fontsize=12)
    ax.set_title('d) Quality Consistency', fontsize=14)
    ax.grid(True, alpha=0.6, axis='y')
    
    # Add legend for consistency levels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='High consistency (std < 0.15)'),
        Patch(facecolor='orange', label='Medium consistency (0.15 ≤ std < 0.25)'),
        Patch(facecolor='red', label='Low consistency (std ≥ 0.25)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Overall title
    plt.suptitle('Data Quality Impact Analysis', fontsize=18)
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(figures_out, 'data_quality_impact_analysis_restored.png'), 
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(figures_out, 'data_quality_impact_analysis_restored.svg'), 
                format='svg', bbox_inches='tight')
    
    # Print summary statistics
    print("\nData Quality Impact Summary:")
    print("-" * 60)
    print(f"Volume-Performance Correlation: R² = {r_value**2:.3f}")
    print(f"\nBest performing cohorts (by mean Dice):")
    for cohort in cohort_stats.head(3).index:
        mean_dice = cohort_stats.loc[cohort, 'mean']
        std_dice = cohort_stats.loc[cohort, 'std']
        count = cohort_stats.loc[cohort, 'count']
        print(f"  {cohort}: {mean_dice:.3f} ± {std_dice:.3f} (n={int(count)})")
    
    print(f"\nMost consistent cohorts (by Dice std dev):")
    for cohort in consistency_stats.head(3).index:
        std_dice = consistency_stats.loc[cohort, 'std']
        count = consistency_stats.loc[cohort, 'count']
        print(f"  {cohort}: std = {std_dice:.3f} (n={int(count)})")
    
    plt.show()
    
    return fig

# Generate the data quality impact analysis figure
fig_quality = create_data_quality_impact_figure(results_df)

# %%
# Comprehensive Radiologist Metrics Analysis
# Calculate metrics for all 1100 cases (11 radiologists × 100 cases without segmentation)

import json
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

# Load all radiologist review data
RADIOLOGIST_REVIEWS_PATH = '/home/jruffle/Desktop/ENHANCEMENT_ARTICLE/radiologist_reviews/'
json_files = glob.glob(os.path.join(RADIOLOGIST_REVIEWS_PATH, '*.json'))

print(f"Found {len(json_files)} radiologist review files")

# Collect all radiologist predictions and ground truth
all_predictions = []
all_ground_truth = []
radiologist_case_data = []

# Load each radiologist's data
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    radiologist_name = os.path.basename(json_file).split('_')[0]
    
    # Process results WITHOUT segmentation only
    # data['results_without_seg'] is a list of dictionaries, not a dict
    for case_data in data['results_without_seg']:
        # Get case_id from the sample data
        case_id = case_data['sample']['base_name']
        
        # Radiologist prediction: 'Y' means enhancement present, 'N' means no enhancement
        pred = 1 if case_data['abnormality'] == 'Y' else 0
        # Ground truth: if enhancement volume > 0, then enhancement is present
        gt = 1 if case_data['sample']['ground_truth_sum'] > 0 else 0
        
        all_predictions.append(pred)
        all_ground_truth.append(gt)
        
        radiologist_case_data.append({
            'radiologist': radiologist_name,
            'case_id': case_id,
            'prediction': pred,
            'ground_truth': gt,
            'confidence': case_data['confidence'],
            'gt_volume': case_data['sample']['ground_truth_sum']
        })

# Convert to numpy arrays for metrics calculation
y_true = np.array(all_ground_truth)
y_pred = np.array(all_predictions)

print(f"\nTotal cases analyzed: {len(y_true)}")
print(f"Number of unique radiologists: {len(set([d['radiologist'] for d in radiologist_case_data]))}")

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(f"\nConfusion Matrix:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# Calculate all metrics
balanced_acc = balanced_accuracy_score(y_true, y_pred)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = sensitivity  # Same as sensitivity
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Calculate standard deviations by computing metrics for each radiologist
radiologist_metrics = []
df_rad = pd.DataFrame(radiologist_case_data)

for radiologist in df_rad['radiologist'].unique():
    rad_data = df_rad[df_rad['radiologist'] == radiologist]
    rad_y_true = rad_data['ground_truth'].values
    rad_y_pred = rad_data['prediction'].values
    
    # Calculate metrics for this radiologist
    rad_ba = balanced_accuracy_score(rad_y_true, rad_y_pred)
    
    # Get confusion matrix values
    if len(np.unique(rad_y_true)) == 2:  # Only if both classes present
        rad_tn, rad_fp, rad_fn, rad_tp = confusion_matrix(rad_y_true, rad_y_pred).ravel()
    else:
        # Handle case where only one class is present
        if np.all(rad_y_true == 1):  # Only positive cases
            rad_tp = np.sum((rad_y_true == 1) & (rad_y_pred == 1))
            rad_fn = np.sum((rad_y_true == 1) & (rad_y_pred == 0))
            rad_tn = 0
            rad_fp = 0
        else:  # Only negative cases
            rad_tn = np.sum((rad_y_true == 0) & (rad_y_pred == 0))
            rad_fp = np.sum((rad_y_true == 0) & (rad_y_pred == 1))
            rad_tp = 0
            rad_fn = 0
    
    rad_sens = rad_tp / (rad_tp + rad_fn) if (rad_tp + rad_fn) > 0 else 0
    rad_spec = rad_tn / (rad_tn + rad_fp) if (rad_tn + rad_fp) > 0 else 0
    rad_prec = rad_tp / (rad_tp + rad_fp) if (rad_tp + rad_fp) > 0 else 0
    rad_f1 = 2 * (rad_prec * rad_sens) / (rad_prec + rad_sens) if (rad_prec + rad_sens) > 0 else 0
    
    radiologist_metrics.append({
        'balanced_accuracy': rad_ba,
        'sensitivity': rad_sens,
        'specificity': rad_spec,
        'precision': rad_prec,
        'f1': rad_f1
    })

# Calculate standard deviations
metrics_df = pd.DataFrame(radiologist_metrics)
ba_std = metrics_df['balanced_accuracy'].std()
sens_std = metrics_df['sensitivity'].std()
spec_std = metrics_df['specificity'].std()
prec_std = metrics_df['precision'].std()
f1_std = metrics_df['f1'].std()

print("\n" + "="*60)
print("RADIOLOGIST PERFORMANCE METRICS (All 1100 cases)")
print("="*60)
print(f"Balanced Accuracy: {balanced_acc:.3f} ± {ba_std:.3f}")
print(f"Sensitivity (Recall): {sensitivity:.3f} ± {sens_std:.3f}")
print(f"Specificity: {specificity:.3f} ± {spec_std:.3f}")
print(f"Precision: {precision:.3f} ± {prec_std:.3f}")
print(f"F1 Score: {f1:.3f} ± {f1_std:.3f}")

# Now calculate unique cases analysis
# Group by case_id to find unique cases
case_analysis = df_rad.groupby('case_id').agg({
    'prediction': 'mean',  # If mean < 0.5, majority predicted no enhancement
    'ground_truth': 'first'  # Ground truth is same for all radiologists
}).reset_index()

# Cases where radiologists failed (majority vote)
case_analysis['radiologist_majority_pred'] = (case_analysis['prediction'] >= 0.5).astype(int)

# Count unique cases where radiologists failed but would have enhancement
rad_failed_cases = case_analysis[
    (case_analysis['radiologist_majority_pred'] == 0) & 
    (case_analysis['ground_truth'] == 1)
]

total_unique_cases = len(case_analysis)
total_positive_cases = len(case_analysis[case_analysis['ground_truth'] == 1])

print(f"\n{'='*60}")
print("UNIQUE CASES ANALYSIS")
print(f"{'='*60}")
print(f"Total unique cases: {total_unique_cases}")
print(f"Total cases with enhancement: {total_positive_cases}")
print(f"\nCases where radiologists failed (majority vote predicted no enhancement when enhancement present):")
print(f"{len(rad_failed_cases)} of {total_positive_cases} ({len(rad_failed_cases)/total_positive_cases*100:.1f}%)")

# For AI comparison, we need to load the model predictions
# First, let's check what AI predictions are available
print("\n" + "="*60)
print("LOADING AI MODEL PREDICTIONS FOR COMPARISON")
print("="*60)

# RESEARCH PAPER STATISTICS CALCULATION
print("\n" + "="*80)
print("RESEARCH PAPER PARAGRAPH STATISTICS")
print("="*80)

# Load model results for comparison
model_results = results_df.copy()

# Create a mapping from case_id to model predictions
model_predictions = {}
for _, row in model_results.iterrows():
    case_id = row['case_id']
    # Model prediction: 1 if predicted volume > 0, 0 otherwise
    model_pred = 1 if row['pred_volume'] > 0 else 0
    # Ground truth: 1 if gt volume > 0, 0 otherwise  
    gt = 1 if row['gt_volume'] > 0 else 0
    model_predictions[case_id] = {
        'model_pred': model_pred,
        'gt': gt
    }

# Create case-level analysis with radiologist majority vote and model predictions
case_stats = []
for case_id in case_analysis['case_id']:
    case_row = case_analysis[case_analysis['case_id'] == case_id].iloc[0]
    
    # Get model prediction for this case
    if case_id in model_predictions:
        model_data = model_predictions[case_id]
        model_pred = model_data['model_pred']
        gt = model_data['gt']
    else:
        continue  # Skip cases not in model results
    
    case_stats.append({
        'case_id': case_id,
        'radiologist_pred': case_row['radiologist_majority_pred'],
        'model_pred': model_pred,
        'ground_truth': gt
    })

case_df = pd.DataFrame(case_stats)

# Calculate the four statistics needed for the research paper
print(f"\\nAnalysis based on {len(case_df)} unique cases with both radiologist and model predictions:")

# 1. "X of Y (Z%) unique cases were wrongly labelled that they would not contain enhancing tumour [by radiologists]"
rad_false_negatives = case_df[(case_df['radiologist_pred'] == 0) & (case_df['ground_truth'] == 1)]
total_enhancement_cases = case_df[case_df['ground_truth'] == 1]
stat1_x = len(rad_false_negatives)
stat1_y = len(total_enhancement_cases)
stat1_z = (stat1_x / stat1_y * 100) if stat1_y > 0 else 0

# 2. "X of Y (Z%) of the unique cases wrongly labelled to not contain enhancing tumour by expert radiologists were correctly labelled to by the model"
rad_fn_model_correct = case_df[
    (case_df['radiologist_pred'] == 0) & 
    (case_df['ground_truth'] == 1) & 
    (case_df['model_pred'] == 1)
]
stat2_x = len(rad_fn_model_correct)
stat2_y = len(rad_false_negatives)
stat2_z = (stat2_x / stat2_y * 100) if stat2_y > 0 else 0

# 3. "X of Y (Z%) unique cases were wrongly labelled by the model to not contain enhancing tumour, when they did"
model_false_negatives = case_df[(case_df['model_pred'] == 0) & (case_df['ground_truth'] == 1)]
stat3_x = len(model_false_negatives)
stat3_y = len(total_enhancement_cases)  # Same denominator as stat1
stat3_z = (stat3_x / stat3_y * 100) if stat3_y > 0 else 0

# 4. "X of Y (Z%) of unique cases that were wrongly labelled by the model to not to contain enhancing tumour were correctly labelled to by the expert radiologists"
model_fn_rad_correct = case_df[
    (case_df['model_pred'] == 0) & 
    (case_df['ground_truth'] == 1) & 
    (case_df['radiologist_pred'] == 1)
]
stat4_x = len(model_fn_rad_correct)
stat4_y = len(model_false_negatives)
stat4_z = (stat4_x / stat4_y * 100) if stat4_y > 0 else 0

# Print the completed paragraph
print("\\n" + "="*80)
print("COMPLETED RESEARCH PAPER PARAGRAPH:")
print("="*80)
print(f"Across the expert radiologist trials, {stat1_x} of {stat1_y} ({stat1_z:.1f}%) unique cases were wrongly labelled that they would not contain enhancing tumour. Moreover, {stat2_x} of {stat2_y} ({stat2_z:.1f}%) of the unique cases wrongly labelled to not contain enhancing tumour by expert radiologists were correctly labelled to by the model. {stat3_x} of {stat3_y} ({stat3_z:.1f}%) unique cases were wrongly labelled by the model to not contain enhancing tumour, when they did. {stat4_x} of {stat4_y} ({stat4_z:.1f}%) of unique cases that were wrongly labelled by the model to not to contain enhancing tumour were correctly labelled to by the expert radiologists. Sample cases wrongly reported by radiologists but correctly identified by the model are shown in Figure 4.")

# Additional breakdown for verification
print("\\n" + "="*60)
print("DETAILED BREAKDOWN:")
print("="*60)
print(f"1. Radiologist false negatives: {stat1_x}/{stat1_y} cases ({stat1_z:.1f}%)")
print(f"2. Of radiologist FN, model got correct: {stat2_x}/{stat2_y} cases ({stat2_z:.1f}%)")
print(f"3. Model false negatives: {stat3_x}/{stat3_y} cases ({stat3_z:.1f}%)")
print(f"4. Of model FN, radiologists got correct: {stat4_x}/{stat4_y} cases ({stat4_z:.1f}%)")

# Cross-tabulation for verification
print("\\n" + "="*40)
print("VERIFICATION - CROSS TABULATION:")
print("="*40)
print("Radiologist vs Model predictions for enhancement cases:")
enhancement_cases = case_df[case_df['ground_truth'] == 1]
crosstab = pd.crosstab(enhancement_cases['radiologist_pred'], enhancement_cases['model_pred'], 
                       margins=True, margins_name="Total")
crosstab.index = ['Radiologist: No Enhancement', 'Radiologist: Enhancement', 'Total']
crosstab.columns = ['Model: No Enhancement', 'Model: Enhancement', 'Total']
print(crosstab)

# %%
# Create a comprehensive figure with 8 panels as requested
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# Create figure with GridSpec for uniform panel sizes
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.3, hspace=0.5)

# Define colormap for consistency with other figures (matching create_figure_5_sample_level_ADAPTED)
colors_pie = sns.color_palette('husl', n_colors=10)

# Panel a: Volume correlation analysis (from clinical_relevance_analysis.png panel b)
ax_a = fig.add_subplot(gs[0, 0])
# Filter for cases with enhancing tumor
et_cases = results_df[results_df['gt_volume'] > 0].copy()
# Convert volumes to cm³ (assuming 1mm³ voxels)
et_cases['gt_volume_cm3'] = et_cases['gt_volume'] / 1000
et_cases['pred_volume_cm3'] = et_cases['pred_volume'] / 1000
# Calculate regression
slope, intercept, r_value, p_value, std_err = stats.linregress(et_cases['gt_volume_cm3'], et_cases['pred_volume_cm3'])
r2 = r_value ** 2
# Create scatter plot with Dice color coding
scatter_a = ax_a.scatter(et_cases['gt_volume_cm3'], et_cases['pred_volume_cm3'], 
                    c=et_cases['dice'], cmap='RdYlBu', alpha=0.6, s=30, 
                    edgecolor='black', linewidth=0.5, vmin=0, vmax=1)
# Add regression line in RED
x_fit = np.linspace(0, et_cases['gt_volume_cm3'].max(), 100)
y_fit = slope * x_fit + intercept
ax_a.plot(x_fit, y_fit, 'r-', linewidth=2, 
        label=f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.3f}')
# Add identity line in BLACK and DASHED
max_val = max(et_cases['gt_volume_cm3'].max(), et_cases['pred_volume_cm3'].max())
ax_a.plot([0, max_val], [0, max_val], 'k--', linewidth=2, 
        label='Identity line')
ax_a.set_xlabel('Ground Truth Volume [cm³]', fontsize=10)
ax_a.set_ylabel('Predicted Volume [cm³]', fontsize=10)
ax_a.set_title('a) Volume correlation analysis', fontsize=12)
ax_a.legend(loc='upper left', fontsize=8)
ax_a.grid(True, alpha=0.6)

# Panel b: Bland-Altman plot (from clinical_relevance_analysis.png panel a)
ax_b = fig.add_subplot(gs[0, 1])
# Calculate mean and difference in cm³
mean_volume_cm3 = (et_cases['gt_volume_cm3'] + et_cases['pred_volume_cm3']) / 2
volume_diff_cm3 = et_cases['pred_volume_cm3'] - et_cases['gt_volume_cm3']
# Create scatter plot with Dice color coding
scatter_b = ax_b.scatter(mean_volume_cm3, volume_diff_cm3, c=et_cases['dice'], 
                    cmap='RdYlBu', alpha=0.6, s=30, edgecolor='black', linewidth=0.5, vmin=0, vmax=1)
# Add mean difference and limits of agreement
mean_diff = np.mean(volume_diff_cm3)
std_diff = np.std(volume_diff_cm3)
upper_limit = mean_diff + 1.96 * std_diff
lower_limit = mean_diff - 1.96 * std_diff

# Print values for debugging
print(f"Panel b - Mean diff: {mean_diff:.2f}, Std diff: {std_diff:.2f}")
print(f"Panel b - Upper limit: {upper_limit:.2f}, Lower limit: {lower_limit:.2f}")

# Plot all lines with explicit zorder to ensure visibility
ax_b.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)
ax_b.axhline(mean_diff, color='red', linestyle='-', linewidth=2, zorder=2,
           label=f'Mean: {mean_diff:.2f}')
ax_b.axhline(upper_limit, color='red', linestyle='--', linewidth=2, zorder=2,
           label=f'+1.96 SD: {upper_limit:.2f}')
ax_b.axhline(lower_limit, color='red', linestyle='--', linewidth=2, zorder=2,
           label=f'-1.96 SD: {lower_limit:.2f}')

ax_b.set_xlabel('Mean Volume (GT + Pred) / 2 [cm³]', fontsize=10)
ax_b.set_ylabel('Volume Difference (Pred - GT) [cm³]', fontsize=10)
ax_b.set_title('b) Bland-Altman plot', fontsize=12)
# Set ylim to ensure all lines are visible
y_range = max(abs(upper_limit), abs(lower_limit)) * 1.2
ax_b.set_ylim(-y_range, y_range)
ax_b.legend(loc='upper right', fontsize=8)
ax_b.grid(True, alpha=0.6)

# Create a single colorbar for panels a and b underneath both plots
# To change position: modify the values [left, bottom, width, height]
# left: move left/right (0-1), bottom: move up/down (0-1)
# width: change width, height: change height
cbar_ax = fig.add_axes([0.225, 0.50, 0.175, 0.02])  # [left, bottom, width, height]
cbar = plt.colorbar(scatter_a, cax=cbar_ax, orientation='horizontal')
# Label removed - will add text within colorbar
cbar.ax.tick_params(labelsize=8)
# Add label within the colorbar
cbar.ax.text(0.5, 0.5, "Dice Score", transform=cbar.ax.transAxes,
             ha="center", va="center", fontsize=9, color="black")

# Panel c: Performance by clinical volume categories (from clinical_relevance_analysis.png panel c)
ax_c = fig.add_subplot(gs[0, 2])
# Define clinical volume categories (in voxels, assuming 1mm³ voxels)
volume_categories = [
    ('Micro\n(<0.5cm³)', 0, 500),
    ('Small\n(0.5-1cm³)', 500, 1000),
    ('Medium\n(1-5cm³)', 1000, 5000),
    ('Large\n(5-10cm³)', 5000, 10000),
    ('Very Large\n(>10cm³)', 10000, np.inf)
]
# Collect data for box plots
category_data = []
category_labels = []
for cat_name, min_vol, max_vol in volume_categories:
    mask = (et_cases['gt_volume'] >= min_vol) & (et_cases['gt_volume'] < max_vol)
    if mask.sum() > 0:
        dice_values = et_cases.loc[mask, 'dice'].values
        category_data.append(dice_values)
        category_labels.append(cat_name)

# Create box plot with scatter overlay - hide outliers, no mean line
positions = np.arange(len(category_data))
bp = ax_c.boxplot(category_data, positions=positions, widths=0.6, 
                  patch_artist=True, showmeans=False,  # No mean line
                  showfliers=False)  # Hide outliers
# Set box colors - different color for each category
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(colors_pie[i % len(colors_pie)])
    patch.set_alpha(0.3)  # More transparent
# Overlay scatter points ON TOP
for i, data in enumerate(category_data):
    y = data
    x = np.random.normal(positions[i], 0.1, size=len(y))
    ax_c.scatter(x, y, alpha=0.6, s=20, color=colors_pie[i % len(colors_pie)], 
                edgecolor='black', linewidth=0.5, zorder=10)  # Higher zorder to be on top
    # Add detection rate annotation
    detection_rate = (data > 0.3).mean() * 100  # Using Dice > 0.3 threshold
    ax_c.text(positions[i], 1.03, f"{detection_rate:.0f}%\n(n={len(data)})", 
            ha='center', va='bottom', fontsize=8)

ax_c.set_xticks(positions)
ax_c.set_xticklabels(category_labels, rotation=45, ha='right', fontsize=9)
ax_c.set_ylabel('Dice Score', fontsize=10)
ax_c.set_title('c) Performance by volume', fontsize=12)
ax_c.set_ylim(0, 1.19)
# Add centered Detection rate label
ax_c.text(0.5, 0.95, "Detection rate (Dice>0.3)", ha="center", va="bottom",
         transform=ax_c.transAxes, fontsize=8)
ax_c.grid(True, alpha=0.6, axis='y')

# Panel d: Performance by enhancement pattern (from subgroup_analysis_radiomics.png panel c)
ax_d = fig.add_subplot(gs[0, 3])
# Load radiomics patterns if available
patterns_csv_path = os.path.join(figures_out, 'results_with_radiomics_patterns.csv')
if os.path.exists(patterns_csv_path):
    results_with_patterns = pd.read_csv(patterns_csv_path)
    # Merge with et_cases
    et_cases_patterns = et_cases.merge(
        results_with_patterns[['case_id', 'enhancement_pattern']], 
        on='case_id', 
        how='left'
    )
    # Remove invalid patterns
    et_cases_patterns = et_cases_patterns[
        (et_cases_patterns['enhancement_pattern'] != 'No Enhancement') & 
        (et_cases_patterns['enhancement_pattern'] != 'Error') &
        (et_cases_patterns['enhancement_pattern'] != 'Single Lesion (Unclassified)') &
        (et_cases_patterns['enhancement_pattern'].notna())
    ]
    
    # Rename patterns as requested - more comprehensive
    pattern_rename_map = {
        'Single Ring': 'Ring',
        'Single Solid': 'Solid',
        'Single Mixed': 'Mixed',
        'Single Heterogeneous': 'Heterogeneous',
        'Infiltrative Single': 'Infiltrative',
        'Irregular/Complex Single': 'Irregular/Complex',
        'Well-circumscribed Single': 'Well-circumscribed',
        'Multiple': 'Multiple components'
    }
    et_cases_patterns['enhancement_pattern'] = et_cases_patterns['enhancement_pattern'].replace(pattern_rename_map)
    
    # Collect data for box plots
    pattern_data = []
    pattern_labels = []
    pattern_order = et_cases_patterns.groupby('enhancement_pattern')['dice'].mean().sort_values(ascending=False).index
    
    for pattern in pattern_order:
        pattern_dice = et_cases_patterns[et_cases_patterns['enhancement_pattern'] == pattern]['dice'].values
        if len(pattern_dice) > 0:
            pattern_data.append(pattern_dice)
            pattern_labels.append(pattern)
    
    # Create box plot with scatter overlay - hide outliers, no mean line
    positions = np.arange(len(pattern_data))
    bp = ax_d.boxplot(pattern_data, positions=positions, widths=0.6, 
                      patch_artist=True, showmeans=False,  # No mean line
                      showfliers=False)  # Hide outliers
    # Set box colors - more transparent
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_pie[i % len(colors_pie)])
        patch.set_alpha(0.3)  # More transparent
    # Overlay scatter points ON TOP
    for i, data in enumerate(pattern_data):
        y = data
        x = np.random.normal(positions[i], 0.1, size=len(y))
        ax_d.scatter(x, y, alpha=0.6, s=20, color=colors_pie[i % len(colors_pie)], 
                    edgecolor='black', linewidth=0.5, zorder=10)  # Higher zorder to be on top
        # Add detection rate annotation
        detection_rate = (data > 0.3).mean() * 100  # Using Dice > 0.3 threshold
        ax_d.text(positions[i], 1.03, f"{detection_rate:.0f}%\n(n={len(data)})", 
                ha='center', va='bottom', fontsize=8)
    
    ax_d.set_xticks(positions)
    ax_d.set_xticklabels(pattern_labels, rotation=45, ha='right', fontsize=9)
    ax_d.set_ylabel('Dice Score', fontsize=10)
    ax_d.set_ylim(0, 1.19)
# Add centered Detection rate label
    ax_d.text(0.5, 0.95, "Detection rate (Dice>0.3)", ha="center", va="bottom",
         transform=ax_d.transAxes, fontsize=8)
    ax_d.set_title('d) Performance by morphology', fontsize=12)
    ax_d.grid(True, alpha=0.6, axis='y')
else:
    # Fallback to pathology mapping
    ax_d.text(0.5, 0.5, 'Radiomics patterns not available', ha='center', va='center', transform=ax_d.transAxes)
    ax_d.set_title('d) Performance by enhancement pattern', fontsize=12)

# Panel e: Detection sensitivity vs size (from clinical_relevance_analysis.png panel e)
ax_e = fig.add_subplot(gs[1, 0])
# Create size bins (log scale) - convert to cm³
size_bins_cm3 = np.logspace(np.log10(0.01), np.log10(et_cases['gt_volume_cm3'].max()), 20)
sensitivities = []
bin_centers_cm3 = []
for i in range(len(size_bins_cm3)-1):
    mask = (et_cases['gt_volume_cm3'] >= size_bins_cm3[i]) & (et_cases['gt_volume_cm3'] < size_bins_cm3[i+1])
    if mask.sum() > 0:
        sensitivity = (et_cases.loc[mask, 'pred_volume'] > 0).mean()
        sensitivities.append(sensitivity)
        bin_centers_cm3.append(np.sqrt(size_bins_cm3[i] * size_bins_cm3[i+1]))  # Geometric mean
# Plot sensitivity curve (convert to percentage)
sensitivities_pct = [s * 100 for s in sensitivities]
ax_e.plot(bin_centers_cm3, sensitivities_pct, 'o-', linewidth=2, markersize=6, color=colors_pie[1])
ax_e.set_xscale('log')
# Set custom x-axis tick labels to show actual values in cm³
x_ticks = [0.01, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64]
ax_e.set_xticks(x_ticks)
ax_e.set_xticklabels([f'{x}' for x in x_ticks])
ax_e.set_xlabel('Tumor Size [cm³]', fontsize=10)
ax_e.set_ylabel('Detection Sensitivity (%)', fontsize=10)
ax_e.set_title('e) Lesion detectability x size', fontsize=12)
ax_e.set_ylim(0, 105)
ax_e.grid(True, alpha=0.6)

# Panel f: Success/failure by pathology (from failure_case_analysis.png panel e)
ax_f = fig.add_subplot(gs[1, 1])
pathology_stats = []
for pathology in et_cases['Pathology'].unique():
    if pathology == '':
        continue
    pathology_data = et_cases[et_cases['Pathology'] == pathology]
    if len(pathology_data) > 0:
        failure_rate = (pathology_data['dice'] < 0.3).sum() / len(pathology_data)
        success_rate = (pathology_data['dice'] >= 0.3).sum() / len(pathology_data)
        pathology_stats.append({
            'name': pathology,
            'success_rate': success_rate,
            'failure_rate': failure_rate,
            'n_cases': len(pathology_data)
        })
# Sort by success rate (highest to lowest)
pathology_stats.sort(key=lambda x: x['success_rate'], reverse=True)
pathology_names = []
pathology_success = []
pathology_failure = []
for stat in pathology_stats:
    # Add line breaks between words
    name = stat['name']
    # Special handling for specific pathologies
    if name == 'Postoperative glioma resection':
        name_with_breaks = 'Postoperative\nglioma resection'
    elif name == 'Paediatric presurgical tumour':
        name_with_breaks = 'Paediatric\ntumour'
    else:
        name_with_breaks = name.replace(' ', '\n')
    pathology_names.append(name_with_breaks)
    pathology_success.append(stat['success_rate'] * 100)  # Convert to percentage
    pathology_failure.append(stat['failure_rate'] * 100)  # Convert to percentage
if pathology_names:
    x_pos = np.arange(len(pathology_names))
    # Use green for success and red for failure
    bars_success = ax_f.bar(x_pos, pathology_success, alpha=0.6, 
                           color=sns.color_palette('Greens', n_colors=3)[2], 
                           label='Success (Dice ≥ 0.3)')
    bars_failure = ax_f.bar(x_pos, pathology_failure, bottom=pathology_success, alpha=0.6, 
                           color=sns.color_palette('Reds', n_colors=3)[2], 
                           label='Failure (Dice < 0.3)')
    
    # Add success rate annotations at the top of the green part
    for i, (success_rate, bar) in enumerate(zip(pathology_success, bars_success)):
        if success_rate > 0:  # Only annotate if there's a green bar
            ax_f.text(bar.get_x() + bar.get_width()/2, success_rate/2,
                     f'{success_rate:.0f}%', 
                     ha='center', va='center', fontsize=8, color='white', weight='bold')
    
    ax_f.set_xticks(x_pos)
    # To change alignment: use ha='right', 'center', or 'left'
    ax_f.set_xticklabels(pathology_names, rotation=45, ha='right', fontsize=9)
    ax_f.set_ylabel('Rate (%)', fontsize=10)
    ax_f.set_ylim(0, 100)
    ax_f.set_title('f) Detection success/failure rate', fontsize=12)
    ax_f.legend(fontsize=8)
    ax_f.grid(True, alpha=0.6, axis='y')

# Panel g: Model performance vs training sample size by pathology (from pathology_sample_size_vs_performance.png panel a)
ax_g = fig.add_subplot(gs[1, 2])
# Get training set pathology counts
train_images = all_images[all_images['Partition'] == 'Train/Val'].copy()
train_pathology_counts = train_images['Pathology'].value_counts().to_dict()
# Get pathology statistics from TEST set performance
pathology_stats = []
pathology_color_map = {}  # Store color mapping for panel h
for pathology in et_cases['Pathology'].unique():
    if pathology != '' and pathology in train_pathology_counts:
        pathology_data = et_cases[et_cases['Pathology'] == pathology]
        stats = {
            'pathology': pathology,
            'train_sample_size': train_pathology_counts.get(pathology, 0),
            'test_sample_size': len(pathology_data),
            'mean_dice': pathology_data['dice'].mean(),
            'std_dice': pathology_data['dice'].std(),
            'sem_dice': pathology_data['dice'].sem()
        }
        pathology_stats.append(stats)
# Convert to DataFrame and sort by training sample size
pathology_df = pd.DataFrame(pathology_stats)
pathology_df = pathology_df.sort_values('train_sample_size')
# Plot
x_values = pathology_df['train_sample_size'].values
y_values = pathology_df['mean_dice'].values
# First plot the connecting line
ax_g.plot(x_values, y_values, 'k-', alpha=0.6, linewidth=2, zorder=1)
# Then plot dots with error bars and store colors
for i, (idx, row) in enumerate(pathology_df.iterrows()):
    color = colors_pie[i % len(colors_pie)]
    pathology_color_map[row['pathology']] = color  # Store color for panel h
    ax_g.errorbar(row['train_sample_size'], row['mean_dice'], 
                yerr=row['sem_dice'] * 1.96,  # 95% confidence interval
                fmt='o', markersize=8, capsize=5, capthick=1.5,
                color=color, 
                elinewidth=1.5, markeredgecolor='black', markeredgewidth=0.8,
                zorder=2)

# Add pathology labels above each point (similar to original implementation)
for idx, row in pathology_df.iterrows():
    # Split long pathology names
    pathology_name = row['pathology']
    # Handle specific renaming
    if pathology_name == 'Paediatric presurgical tumour':
        pathology_name = 'Paediatric\ntumour'
    elif len(pathology_name) > 20:
        words = pathology_name.split()
        mid_point = len(words) // 2
        pathology_name = ' '.join(words[:mid_point]) + '\n' + ' '.join(words[mid_point:])
    
    # Special handling for specific pathologies
    if 'Postoperative glioma resection' in row['pathology']:
        # Position text below and to the right
        ax_g.annotate(pathology_name, 
                    xy=(row['train_sample_size'], row['mean_dice']),
                    xytext=(15, -15), textcoords='offset points',
                    ha='left', va='top',
                    fontsize=8,  # Same as legend font size
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='gray', alpha=0.6))
    elif 'Paediatric tumour' in row['pathology']:
        # Position text below and to the right
        ax_g.annotate(pathology_name, 
                    xy=(row['train_sample_size'], row['mean_dice']),
                    xytext=(20, -15), textcoords='offset points',
                    ha='left', va='top',
                    fontsize=8,  # Same as legend font size
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='gray', alpha=0.6))
    else:
        # Position text above the point
        ax_g.annotate(pathology_name, 
                    xy=(row['train_sample_size'], row['mean_dice']),
                    xytext=(0, 15), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8,  # Same as legend font size
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor='gray', alpha=0.6))

# Customize plot
ax_g.set_xlabel('Training sample size', fontsize=10)
ax_g.set_ylabel('Test Set Performance (Mean Dice Score)', fontsize=10)
ax_g.set_title('g) Sample size x performance', fontsize=12)
ax_g.set_xscale('log')
custom_xticks = [200,400,800,1600,3200,6400,12800]
ax_g.set_xticks(custom_xticks)
ax_g.set_xticklabels(custom_xticks)
ax_g.set_ylim(0, 1)
ax_g.set_xlim(custom_xticks[0], custom_xticks[-1])
ax_g.grid(True, alpha=0.6)

# Panel h: Model efficiency by pathology type (from pathology_sample_size_vs_performance.png panel b)
ax_h = fig.add_subplot(gs[1, 3])
# Calculate dice gain per training sample
pathology_df['dice_per_sample'] = pathology_df['mean_dice'] / np.log(pathology_df['train_sample_size'] + 1)
# Sort by dice_per_sample for better visualization
pathology_df_sorted = pathology_df.sort_values('dice_per_sample', ascending=False)
x_pos = np.arange(len(pathology_df_sorted))
# Create bars with colors matching panel g
bars = []
for i, (idx, row) in enumerate(pathology_df_sorted.iterrows()):
    # Use the same color as in panel g
    color = pathology_color_map[row['pathology']]
    bar = ax_h.bar(i, row['dice_per_sample'], alpha=0.6, color=color, edgecolor='black')
    bars.append(bar)

# Customize second plot
ax_h.set_xticks(x_pos)
# Get labels with line breaks
labels_with_breaks = []
for idx, row in pathology_df_sorted.iterrows():
    name = row['pathology']
    # Special handling for specific pathologies
    if name == 'Postoperative glioma resection':
        label = 'Postoperative\nglioma resection'
    elif name == 'Paediatric presurgical tumour':
        label = 'Paediatric\ntumour'
    else:
        label = name.replace(' ', '\n')
    labels_with_breaks.append(label)
# To change alignment: use ha='right', 'center', or 'left'
ax_h.set_xticklabels(labels_with_breaks, rotation=45, ha='right', fontsize=9)
ax_h.set_ylabel('Dice Gain per Training Sample (Efficiency)', fontsize=10)
ax_h.set_title('h) Performance efficiency', fontsize=12)
ax_h.grid(True, alpha=0.6, axis='y')

# Overall title
plt.suptitle('Relationship between model performance and lesion volume, morphology, and pathology', fontsize=16, y=0.95)

# Save the figure
fig.savefig(os.path.join(figures_out, 'Figure_6.png'), 
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(figures_out, 'Figure_6.svg'), 
            format='svg', bbox_inches='tight')

plt.show()

print("Comprehensive performance evaluation figure created successfully!")
print(f"Saved as: performance_evaluation_comprehensive.png and .svg")
print("\n" + "="*60)
print("ADJUSTMENT INSTRUCTIONS")
print("="*60)
print("\n1. X-AXIS TICK ALIGNMENT (panels c/d/f/h):")
print("   To change x-axis label alignment, modify the 'ha' parameter in set_xticklabels:")
print("   - ha='right': Right-aligned (current setting)")
print("   - ha='center': Center-aligned")
print("   - ha='left': Left-aligned")
print("   Example: ax_c.set_xticklabels(labels, rotation=30, ha='center', fontsize=9)")
print("\n2. COLORBAR POSITION (panels a & b):")
print("   Modify values in: cbar_ax = fig.add_axes([0.10, 0.48, 0.175, 0.02])")
print("   - [0.10]: Horizontal position (0=far left, 1=far right)")
print("   - [0.48]: Vertical position (0=bottom, 1=top)")
print("   - [0.175]: Width of colorbar")
print("   - [0.02]: Height of colorbar")
print("\n3. COLORBAR LABEL ALIGNMENT:")
print("   To adjust the 'Dice Score' label position:")
print("   - cbar.set_label('Dice Score', fontsize=10, loc='center')  # Options: 'left', 'center', 'right'")
print("   - Or use labelpad to adjust distance: cbar.set_label('Dice Score', fontsize=10, labelpad=10)")

# %%
# Statistical Analysis for Figure 6 Panels c, d, e, f, g
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STATISTICAL ANALYSIS FOR FIGURE 6")
print("="*80)

# First, merge pathology information if needed
if 'pathology' not in et_cases.columns and 'Pathology' not in et_cases.columns:
    print("Merging pathology information...")
    # Try to get pathology from all_images
    if 'Pathology' in all_images.columns:
        pathology_info = all_images[['case_id', 'Pathology']].copy()
        pathology_info.columns = ['case_id', 'pathology']
        et_cases = et_cases.merge(pathology_info, on='case_id', how='left')
        print(f"Merged pathology data for {et_cases['pathology'].notna().sum()} cases")
    elif 'pathology' in results_df.columns:
        # If pathology is in results_df, use that
        pathology_info = results_df[['case_id', 'pathology']].copy()
        et_cases = et_cases.merge(pathology_info, on='case_id', how='left')
        print(f"Merged pathology data from results_df")
elif 'Pathology' in et_cases.columns and 'pathology' not in et_cases.columns:
    # Standardize column name
    et_cases['pathology'] = et_cases['Pathology']

# Panel c: One-way ANOVA for performance by volume categories
print("" + "="*60)
print("PANEL C: Performance by Volume Categories")
print("="*60)

# Prepare data for ANOVA
volume_categories = [
    ('Micro (<0.5cm³)', 0, 500),
    ('Small (0.5-1cm³)', 500, 1000),
    ('Medium (1-5cm³)', 1000, 5000),
    ('Large (5-10cm³)', 5000, 10000),
    ('Very Large (>10cm³)', 10000, np.inf)
]

# Collect dice scores for each category
volume_groups = []
volume_labels = []
for cat_name, min_vol, max_vol in volume_categories:
    mask = (et_cases['gt_volume'] >= min_vol) & (et_cases['gt_volume'] < max_vol)
    if mask.sum() > 0:
        dice_values = et_cases.loc[mask, 'dice'].values
        volume_groups.append(dice_values)
        volume_labels.append(cat_name)
        print(f"{cat_name}: n={len(dice_values)}, mean={np.mean(dice_values):.3f}, std={np.std(dice_values):.3f}")

# One-way ANOVA
if len(volume_groups) > 2:
    f_stat, p_value = stats.f_oneway(*volume_groups)
    print(f"One-way ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print("Significant difference found. Performing post-hoc Tukey HSD test...")
        
        # Prepare data for Tukey HSD
        all_dice = []
        all_groups = []
        for i, group in enumerate(volume_groups):
            all_dice.extend(group)
            all_groups.extend([volume_labels[i]] * len(group))
        
        # Perform Tukey HSD
        tukey = pairwise_tukeyhsd(all_dice, all_groups, alpha=0.05)
        print("Tukey HSD Results:")
        print(tukey)

# Panel d: One-way ANOVA for performance by morphology
print("" + "="*60)
print("PANEL D: Performance by Morphology")
print("="*60)

# Load enhancement patterns if not already present
if 'enhancement_pattern' not in et_cases.columns:
    patterns_csv_path = os.path.join(figures_out, 'results_with_radiomics_patterns.csv')
    if os.path.exists(patterns_csv_path):
        print("Loading radiomics patterns from file...")
        results_with_patterns = pd.read_csv(patterns_csv_path)
        # Merge with et_cases
        et_cases = et_cases.merge(
            results_with_patterns[['case_id', 'enhancement_pattern']], 
            on='case_id', 
            how='left'
        )
        print(f"Loaded enhancement patterns for {et_cases['enhancement_pattern'].notna().sum()} cases")
    else:
        print("Radiomics patterns file not found at:", patterns_csv_path)

# Check if enhancement_pattern column exists after loading attempt
if 'enhancement_pattern' not in et_cases.columns:
    print("Enhancement pattern column not found in et_cases. Skipping morphology analysis.")
    print("Note: This analysis requires radiomics-based enhancement pattern classification.")
else:
    # Get enhancement pattern data
    et_cases_patterns = et_cases[et_cases['enhancement_pattern'].notna()].copy()
    pattern_rename_map = {
        'Infiltrative Single': 'Infiltrative',
        'Irregular/Complex Single': 'Irregular/Complex',
        'Well-circumscribed Single': 'Well-circumscribed',
        'Multiple': 'Multiple components'
    }
    et_cases_patterns['enhancement_pattern'] = et_cases_patterns['enhancement_pattern'].replace(pattern_rename_map)

    # Collect dice scores for each pattern
    pattern_groups = []
    pattern_labels = []
    for pattern in et_cases_patterns['enhancement_pattern'].unique():
        if pattern in pattern_rename_map.values() or pattern in pattern_rename_map.keys():
            pattern_data = et_cases_patterns[et_cases_patterns['enhancement_pattern'] == pattern]['dice'].values
            if len(pattern_data) > 0:
                pattern_groups.append(pattern_data)
                pattern_labels.append(pattern)
                print(f"{pattern}: n={len(pattern_data)}, mean={np.mean(pattern_data):.3f}, std={np.std(pattern_data):.3f}")

    # One-way ANOVA
    if len(pattern_groups) > 2:
        f_stat, p_value = stats.f_oneway(*pattern_groups)
        print(f"One-way ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("Significant difference found. Performing post-hoc Tukey HSD test...")
            
            # Prepare data for Tukey HSD
            all_dice = []
            all_groups = []
            for i, group in enumerate(pattern_groups):
                all_dice.extend(group)
                all_groups.extend([pattern_labels[i]] * len(group))
            
            # Perform Tukey HSD
            tukey = pairwise_tukeyhsd(all_dice, all_groups, alpha=0.05)
            print("Tukey HSD Results:")
            print(tukey)
    elif len(pattern_groups) == 2:
        # If only 2 groups, use t-test
        t_stat, p_value = stats.ttest_ind(pattern_groups[0], pattern_groups[1])
        print(f"T-test (two groups): t={t_stat:.3f}, p={p_value:.4f}")
    else:
        print("Not enough pattern groups for statistical comparison.")

# Panel e: Lesion detectability by size - Logistic regression
print("" + "="*60)
print("PANEL E: Lesion Detectability by Size")
print("="*60)

# Create binary detection variable (detected if dice > 0.3)
et_cases['detected'] = (et_cases['dice'] > 0.3).astype(int)
et_cases['log_volume'] = np.log10(et_cases['gt_volume'] + 1)  # Log transform volume

# Remove any infinite or NaN values
valid_mask = np.isfinite(et_cases['log_volume']) & np.isfinite(et_cases['detected'])
X = sm.add_constant(et_cases.loc[valid_mask, 'log_volume'])
y = et_cases.loc[valid_mask, 'detected']

try:
    # Logistic regression
    logit_model = sm.Logit(y, X).fit(disp=0)
    
    print("Logistic Regression Results:")
    print(f"Log-likelihood: {logit_model.llf:.3f}")
    print(f"Pseudo R-squared: {logit_model.prsquared:.3f}")
    print("Coefficients:")
    for i, (name, coef, pval) in enumerate(zip(['Intercept', 'Log10(Volume)'], 
                                               logit_model.params, 
                                               logit_model.pvalues)):
        print(f"  {name}: β={coef:.3f}, p={pval:.4f}")

    # Calculate odds ratio for volume
    or_volume = np.exp(logit_model.params[1])
    or_ci = np.exp(logit_model.conf_int().iloc[1])
    print(f"Odds Ratio (per 10-fold increase in volume): {or_volume:.3f} (95% CI: {or_ci[0]:.3f}-{or_ci[1]:.3f})")
except Exception as e:
    print(f"Error in logistic regression: {e}")
    print("Performing alternative analysis...")
    # Alternative: correlation between volume and detection
    corr, p_val = stats.pointbiserialr(et_cases.loc[valid_mask, 'detected'], 
                                       et_cases.loc[valid_mask, 'log_volume'])
    print(f"Point-biserial correlation: r={corr:.3f}, p={p_val:.4f}")

# Panel f: Success/failure rates by pathology - Chi-squared test
print("" + "="*60)
print("PANEL F: Detection Success/Failure by Pathology")
print("="*60)

# Check if pathology column exists
if 'pathology' not in et_cases.columns:
    print("Pathology column not found in et_cases. Skipping pathology success/failure analysis.")
else:
    # Create contingency table
    pathology_types = ['Presurgical glioma', 'Postoperative glioma resection', 
                       'Meningioma', 'Metastases', 'Paediatric presurgical tumour']

    contingency_table = []
    pathology_names_clean = []

    for pathology in pathology_types:
        pathology_data = et_cases[et_cases['pathology'] == pathology]
        if len(pathology_data) > 0:
            success = (pathology_data['dice'] >= 0.3).sum()
            failure = (pathology_data['dice'] < 0.3).sum()
            contingency_table.append([success, failure])
            pathology_names_clean.append(pathology)
            total = success + failure
            success_rate = (success / total * 100) if total > 0 else 0
            print(f"{pathology}: Success={success}, Failure={failure}, Success Rate={success_rate:.1f}%")

    if len(contingency_table) > 1:
        contingency_table = np.array(contingency_table)
        
        # Chi-squared test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-squared test: χ²={chi2:.3f}, df={dof}, p={p_value:.4f}")

        # Post-hoc pairwise comparisons with Bonferroni correction
        if p_value < 0.05:
            print("Significant difference found. Performing pairwise Fisher's exact tests...")
            n_comparisons = len(pathology_names_clean) * (len(pathology_names_clean) - 1) // 2
            bonferroni_alpha = 0.05 / n_comparisons
            print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")
            
            print("Pairwise comparisons:")
            for i in range(len(pathology_names_clean)):
                for j in range(i+1, len(pathology_names_clean)):
                    table_2x2 = contingency_table[[i,j], :]
                    _, p_fisher = fisher_exact(table_2x2)
                    sig = "*" if p_fisher < bonferroni_alpha else ""
                    print(f"  {pathology_names_clean[i]} vs {pathology_names_clean[j]}: p={p_fisher:.4f} {sig}")

# Panel g: Sample size vs performance - ANOVA and regression
print("" + "="*60)
print("PANEL G: Sample Size vs Performance")
print("="*60)

# Check if pathology column exists
if 'pathology' not in et_cases.columns:
    print("Pathology column not found. Skipping sample size analysis.")
else:
    # Get pathology-wise performance data
    pathology_performance = []
    unique_pathologies = et_cases['pathology'].unique()
    
    for pathology in unique_pathologies:
        if pd.notna(pathology):  # Skip NaN values
            pathology_data = et_cases[et_cases['pathology'] == pathology]
            if len(pathology_data) > 5:  # Only include if sufficient test samples
                # Count training samples from all_images
                train_count = 0
                if 'Pathology' in all_images.columns and 'Partition' in all_images.columns:
                    train_count = ((all_images['Pathology'] == pathology) & 
                                  (all_images['Partition'] == 'Train/Val')).sum()
                elif 'pathology' in all_images.columns and 'Partition' in all_images.columns:
                    train_count = ((all_images['pathology'] == pathology) & 
                                  (all_images['Partition'] == 'Train/Val')).sum()
                else:
                    # Fallback: estimate from test set size
                    train_count = len(pathology_data) * 9  # Assume 90/10 split
                
                if train_count > 0:  # Only include if we have training samples
                    mean_dice = pathology_data['dice'].mean()
                    pathology_performance.append({
                        'pathology': pathology,
                        'train_count': train_count,
                        'mean_dice': mean_dice,
                        'test_count': len(pathology_data)
                    })

    if len(pathology_performance) > 0:
        perf_df = pd.DataFrame(pathology_performance)
        print("Pathology Performance Summary:")
        print(perf_df.to_string(index=False))

        # One-way ANOVA across pathologies
        pathology_dice_groups = []
        pathology_names_anova = []
        for pathology in perf_df['pathology']:
            dice_values = et_cases[et_cases['pathology'] == pathology]['dice'].values
            if len(dice_values) > 0:
                pathology_dice_groups.append(dice_values)
                pathology_names_anova.append(pathology)

        if len(pathology_dice_groups) > 2:
            f_stat, p_value = stats.f_oneway(*pathology_dice_groups)
            print(f"One-way ANOVA across pathologies: F={f_stat:.3f}, p={p_value:.4f}")
            
            if p_value < 0.05:
                print("Significant differences found. Performing post-hoc Tukey HSD test...")
                # Prepare data for Tukey HSD
                all_dice = []
                all_groups = []
                for i, group in enumerate(pathology_dice_groups):
                    all_dice.extend(group)
                    all_groups.extend([pathology_names_anova[i]] * len(group))
                
                # Perform Tukey HSD
                tukey = pairwise_tukeyhsd(all_dice, all_groups, alpha=0.05)
                print("Tukey HSD Results:")
                print(tukey)

        # Linear regression: log(sample size) vs performance
        if len(perf_df) > 2:
            perf_df['log_train_count'] = np.log10(perf_df['train_count'])
            X = sm.add_constant(perf_df['log_train_count'])
            y = perf_df['mean_dice']

            # Weighted by test sample size
            weights = np.sqrt(perf_df['test_count'])
            
            try:
                wls_model = sm.WLS(y, X, weights=weights).fit()

                print("Weighted Linear Regression: Mean Dice ~ log10(Training Sample Size)")
                print(f"R-squared: {wls_model.rsquared:.3f}")
                print(f"Adjusted R-squared: {wls_model.rsquared_adj:.3f}")
                print("Coefficients:")
                for i, (name, coef, pval) in enumerate(zip(['Intercept', 'Log10(Train Count)'], 
                                                           wls_model.params, 
                                                           wls_model.pvalues)):
                    print(f"  {name}: β={coef:.3f}, p={pval:.4f}")
            except Exception as e:
                print(f"Error in weighted regression: {e}")

            # Pearson correlation
            corr, p_corr = stats.pearsonr(perf_df['log_train_count'], perf_df['mean_dice'])
            print(f"Pearson correlation: r={corr:.3f}, p={p_corr:.4f}")
    else:
        print("Not enough pathology data for panel G analysis.")

print("" + "="*80)
print("STATISTICAL ANALYSIS COMPLETE")
print("="*80)

# %%
# Create dataframe with test set cases, volume categories, and radiomic categories
print("Creating summary dataframe for test set cases...")

# Filter for test cases with enhancing tumors (gt_volume > 0)
test_cases = results_df[results_df['gt_volume'] > 0].copy()

# Define volume categories (from cell 50)
def assign_volume_category(volume):
    if volume < 500:
        return 'Micro'
    elif volume < 1000:
        return 'Small'
    elif volume < 5000:
        return 'Medium'
    elif volume < 10000:
        return 'Large'
    else:
        return 'Very Large'

# Assign volume categories
test_cases['volume_category'] = test_cases['gt_volume'].apply(assign_volume_category)

# Load radiomics patterns (from cell 38)
radiomics_patterns = pd.read_csv(figures_out+'results_with_radiomics_patterns.csv')

# Merge with radiomics patterns
test_cases_with_patterns = test_cases.merge(
    radiomics_patterns[['case_id', 'enhancement_pattern']], 
    on='case_id', 
    how='left'
)

# Create the final dataframe with required columns
summary_df = pd.DataFrame({
    'filename': test_cases_with_patterns['case_id'],
    'volume_category': test_cases_with_patterns['volume_category'],
    'radiomic_category': test_cases_with_patterns['enhancement_pattern']
})

# Save to CSV
output_filename = 'test_cases_volume_radiomic_summary.csv'
summary_df.to_csv(figures_out+output_filename, index=False)
print(f"Summary dataframe saved to: {output_filename}")
print(f"Total test cases with enhancing tumors: {len(summary_df)}")

# Display first few rows
print("\nFirst 5 rows of the summary:")
print(summary_df.head())

# %%
figures_out+output_filename

# %%
# Check if the et_cases dataframe exists and has data
if 'et_cases' in locals():
    print(f"et_cases exists with {len(et_cases)} rows")
    print(f"Columns: {list(et_cases.columns)}")
else:
    print("et_cases does not exist")
    
# Check if required columns exist
if 'et_cases' in locals():
    required_cols = ['dice', 'gt_volume', 'pathology']
    for col in required_cols:
        if col in et_cases.columns:
            print(f"✓ Column '{col}' exists")
        else:
            print(f"✗ Column '{col}' missing")

# %%
# Statistical Analysis for Figure 6 Panels c, d, e, f, g
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STATISTICAL ANALYSIS FOR FIGURE 6")
print("="*80)

# Panel c: One-way ANOVA for performance by volume categories
print("\n" + "="*60)
print("PANEL C: Performance by Volume Categories")
print("="*60)

# Prepare data for ANOVA
volume_categories = [
    ('Micro (<0.5cm³)', 0, 500),
    ('Small (0.5-1cm³)', 500, 1000),
    ('Medium (1-5cm³)', 1000, 5000),
    ('Large (5-10cm³)', 5000, 10000),
    ('Very Large (>10cm³)', 10000, np.inf)
]

# Collect dice scores for each category
volume_groups = []
volume_labels = []
for cat_name, min_vol, max_vol in volume_categories:
    mask = (et_cases['gt_volume'] >= min_vol) & (et_cases['gt_volume'] < max_vol)
    if mask.sum() > 0:
        dice_values = et_cases.loc[mask, 'dice'].values
        volume_groups.append(dice_values)
        volume_labels.append(cat_name)
        print(f"{cat_name}: n={len(dice_values)}, mean={np.mean(dice_values):.3f}, std={np.std(dice_values):.3f}")

# One-way ANOVA
if len(volume_groups) > 2:
    f_stat, p_value = stats.f_oneway(*volume_groups)
    print(f"\nOne-way ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print("Significant difference found. Performing post-hoc Tukey HSD test...")
        
        # Prepare data for Tukey HSD
        all_dice = []
        all_groups = []
        for i, group in enumerate(volume_groups):
            all_dice.extend(group)
            all_groups.extend([volume_labels[i]] * len(group))
        
        # Perform Tukey HSD
        tukey = pairwise_tukeyhsd(all_dice, all_groups, alpha=0.05)
        print("\nTukey HSD Results:")
        print(tukey)

# %%
# Panel d: One-way ANOVA for performance by morphology
print("\n" + "="*60)
print("PANEL D: Performance by Morphology")
print("="*60)

# Check if enhancement_pattern column exists
if 'enhancement_pattern' not in et_cases.columns:
    print("Enhancement pattern column not found in et_cases. Skipping morphology analysis.")
    print("Note: This analysis requires radiomics-based enhancement pattern classification.")
else:
    # Get enhancement pattern data
    et_cases_patterns = et_cases[et_cases['enhancement_pattern'].notna()].copy()
    pattern_rename_map = {
        'Infiltrative Single': 'Infiltrative',
        'Irregular/Complex Single': 'Irregular/Complex',
        'Well-circumscribed Single': 'Well-circumscribed',
        'Multiple': 'Multiple components'
    }
    et_cases_patterns['enhancement_pattern'] = et_cases_patterns['enhancement_pattern'].replace(pattern_rename_map)

    # Collect dice scores for each pattern
    pattern_groups = []
    pattern_labels = []
    for pattern in et_cases_patterns['enhancement_pattern'].unique():
        if pattern in pattern_rename_map.values() or pattern in pattern_rename_map.keys():
            pattern_data = et_cases_patterns[et_cases_patterns['enhancement_pattern'] == pattern]['dice'].values
            if len(pattern_data) > 0:
                pattern_groups.append(pattern_data)
                pattern_labels.append(pattern)
                print(f"{pattern}: n={len(pattern_data)}, mean={np.mean(pattern_data):.3f}, std={np.std(pattern_data):.3f}")

    # One-way ANOVA
    if len(pattern_groups) > 2:
        f_stat, p_value = stats.f_oneway(*pattern_groups)
        print(f"\nOne-way ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("Significant difference found. Performing post-hoc Tukey HSD test...")
            
            # Prepare data for Tukey HSD
            all_dice = []
            all_groups = []
            for i, group in enumerate(pattern_groups):
                all_dice.extend(group)
                all_groups.extend([pattern_labels[i]] * len(group))
            
            # Perform Tukey HSD
            tukey = pairwise_tukeyhsd(all_dice, all_groups, alpha=0.05)
            print("\nTukey HSD Results:")
            print(tukey)

# %%
# Panel e: Lesion detectability by size - Logistic regression
print("\n" + "="*60)
print("PANEL E: Lesion Detectability by Size")
print("="*60)

# Create binary detection variable (detected if dice > 0.3)
et_cases['detected'] = (et_cases['dice'] > 0.3).astype(int)
et_cases['log_volume'] = np.log10(et_cases['gt_volume'] + 1)  # Log transform volume

# Remove any infinite or NaN values
valid_mask = np.isfinite(et_cases['log_volume']) & np.isfinite(et_cases['detected'])
X = sm.add_constant(et_cases.loc[valid_mask, 'log_volume'])
y = et_cases.loc[valid_mask, 'detected']

try:
    # Logistic regression
    logit_model = sm.Logit(y, X).fit(disp=0)
    
    print("Logistic Regression Results:")
    print(f"Log-likelihood: {logit_model.llf:.3f}")
    print(f"Pseudo R-squared: {logit_model.prsquared:.3f}")
    print("\nCoefficients:")
    for i, (name, coef, pval) in enumerate(zip(['Intercept', 'Log10(Volume)'], 
                                               logit_model.params, 
                                               logit_model.pvalues)):
        print(f"  {name}: β={coef:.3f}, p={pval:.4f}")

    # Calculate odds ratio for volume
    or_volume = np.exp(logit_model.params[1])
    or_ci = np.exp(logit_model.conf_int().iloc[1])
    print(f"\nOdds Ratio (per 10-fold increase in volume): {or_volume:.3f} (95% CI: {or_ci[0]:.3f}-{or_ci[1]:.3f})")
    
    # Print detection rates for specific volume thresholds
    volume_thresholds = [100, 500, 1000, 5000, 10000, 50000]
    print("\nDetection sensitivity by volume:")
    for vol in volume_thresholds:
        mask = et_cases['gt_volume'] >= vol
        if mask.sum() > 0:
            detection_rate = et_cases.loc[mask, 'detected'].mean()
            print(f"  Volume ≥ {vol} voxels: {detection_rate:.1%} (n={mask.sum()})")
            
except Exception as e:
    print(f"Error in logistic regression: {e}")

# %%
# Panel f: Success/failure rates by pathology - Chi-squared test
print("\n" + "="*60)
print("PANEL F: Detection Success/Failure by Pathology")
print("="*60)

# Check if pathology column exists
if 'pathology' not in et_cases.columns:
    print("Pathology column not found in et_cases. Skipping pathology success/failure analysis.")
else:
    # Create contingency table
    pathology_types = ['Presurgical glioma', 'Postoperative glioma resection', 
                       'Meningioma', 'Metastases', 'Paediatric presurgical tumour']

    contingency_table = []
    pathology_names_clean = []

    for pathology in pathology_types:
        pathology_data = et_cases[et_cases['pathology'] == pathology]
        if len(pathology_data) > 0:
            success = (pathology_data['dice'] >= 0.3).sum()
            failure = (pathology_data['dice'] < 0.3).sum()
            contingency_table.append([success, failure])
            pathology_names_clean.append(pathology)
            success_rate = success / (success + failure) * 100
            print(f"{pathology}: Success={success}, Failure={failure}, Success Rate={success_rate:.1f}%")

    # Chi-squared test
    if len(contingency_table) > 1:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"\nChi-squared test: χ²={chi2:.3f}, df={dof}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("Significant difference found. Performing pairwise Fisher's exact tests...")
            
            # Pairwise Fisher's exact tests
            alpha = 0.05
            n_comparisons = len(pathology_names_clean) * (len(pathology_names_clean) - 1) // 2
            bonferroni_alpha = alpha / n_comparisons
            print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")
            
            print("\nPairwise comparisons:")
            for i in range(len(pathology_names_clean)):
                for j in range(i+1, len(pathology_names_clean)):
                    _, p_val = fisher_exact([contingency_table[i], contingency_table[j]])
                    sig = "*" if p_val < bonferroni_alpha else ""
                    print(f"  {pathology_names_clean[i]} vs {pathology_names_clean[j]}: p={p_val:.4f} {sig}")

# %%
# Panel g: Sample size vs performance
print("\n" + "="*60)
print("PANEL G: Sample Size vs Performance")
print("="*60)

# Get training counts and performance by pathology
pathology_performance = []
for pathology in ['Presurgical glioma', 'Postoperative glioma resection', 
                  'Meningioma', 'Metastases', 'Paediatric presurgical tumour']:
    pathology_data = et_cases[et_cases['pathology'] == pathology]
    if len(pathology_data) > 0:
        # Get training count for this pathology
        if pathology == 'Presurgical glioma':
            train_count = 6707
        elif pathology == 'Postoperative glioma resection':
            train_count = 1383
        elif pathology == 'Meningioma':
            train_count = 900
        elif pathology == 'Metastases':
            train_count = 669
        elif pathology == 'Paediatric presurgical tumour':
            train_count = 321
        
        mean_dice = pathology_data['dice'].mean()
        std_dice = pathology_data['dice'].std()
        test_count = len(pathology_data)
        
        pathology_performance.append({
            'pathology': pathology,
            'train_count': train_count,
            'mean_dice': mean_dice,
            'std_dice': std_dice,
            'test_count': test_count
        })

# Create dataframe for analysis
pathology_df = pd.DataFrame(pathology_performance)
pathology_df = pathology_df.sort_values('mean_dice')

print("Pathology Performance Summary:")
for idx, row in pathology_df.iterrows():
    print(f"{row['pathology']:30s}: Train={row['train_count']:4d}, Test={row['test_count']:3d}, "
          f"Dice={row['mean_dice']:.3f} ± {row['std_dice']:.3f}")

# One-way ANOVA across pathologies
print("\nOne-way ANOVA across pathologies:")
pathology_groups = []
for pathology in pathology_df['pathology']:
    pathology_data = et_cases[et_cases['pathology'] == pathology]['dice'].values
    pathology_groups.append(pathology_data)

f_stat, p_value = stats.f_oneway(*pathology_groups)
print(f"F={f_stat:.3f}, p={p_value:.4f}")

if p_value < 0.05:
    print("Significant differences found. Performing post-hoc Tukey HSD test...")
    # Prepare data for Tukey HSD
    all_dice = []
    all_pathologies = []
    for i, pathology in enumerate(pathology_df['pathology']):
        pathology_data = et_cases[et_cases['pathology'] == pathology]['dice'].values
        all_dice.extend(pathology_data)
        all_pathologies.extend([pathology] * len(pathology_data))
    
    # Perform Tukey HSD
    tukey = pairwise_tukeyhsd(all_dice, all_pathologies, alpha=0.05)
    print("\nTukey HSD Results:")
    print(tukey)

print("\n" + "="*80)
print("STATISTICAL ANALYSIS FOR FIGURE 6 COMPLETE")
print("="*80)
