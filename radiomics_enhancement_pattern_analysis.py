#!/usr/bin/env python3
"""
Radiomics-based Enhancement Pattern Analysis
Derives enhancement patterns from ground truth labels using shape and morphological features
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage import measure
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

def calculate_shape_features(binary_mask, voxel_spacing=(1, 1, 1)):
    """
    Calculate shape features for a binary mask
    
    Parameters:
    -----------
    binary_mask : numpy.ndarray
        3D binary mask of the enhancing tumor
    voxel_spacing : tuple
        Voxel spacing in mm (x, y, z)
    
    Returns:
    --------
    dict : Dictionary containing shape features
    """
    features = {}
    
    if np.sum(binary_mask) == 0:
        # No enhancement present
        features['volume'] = 0
        features['surface_area'] = 0
        features['sphericity'] = np.nan
        features['compactness'] = np.nan
        features['elongation'] = np.nan
        features['solidity'] = np.nan
        features['surface_volume_ratio'] = np.nan
        features['n_components'] = 0
        features['largest_component_ratio'] = np.nan
        return features
    
    # Get connected components
    labeled_mask, n_components = ndimage.label(binary_mask)
    features['n_components'] = n_components
    
    # Find largest component
    component_sizes = []
    for i in range(1, n_components + 1):
        component_size = np.sum(labeled_mask == i)
        component_sizes.append(component_size)
    
    largest_component_idx = np.argmax(component_sizes) + 1
    largest_component_mask = (labeled_mask == largest_component_idx)
    
    # Calculate volume
    voxel_volume = np.prod(voxel_spacing)
    total_volume = np.sum(binary_mask) * voxel_volume
    features['volume'] = total_volume
    
    # Ratio of largest component to total
    if n_components > 0:
        features['largest_component_ratio'] = component_sizes[largest_component_idx-1] / np.sum(binary_mask)
    else:
        features['largest_component_ratio'] = np.nan
    
    # Calculate features on largest component for consistency
    if np.sum(largest_component_mask) > 10:  # Minimum size threshold
        try:
            # Surface area using marching cubes
            verts, faces, _, _ = measure.marching_cubes(
                largest_component_mask.astype(float), 
                level=0.5, 
                spacing=voxel_spacing
            )
            
            # Calculate surface area from mesh
            surface_area = measure.mesh_surface_area(verts, faces)
            features['surface_area'] = surface_area
            
            # Surface to volume ratio
            component_volume = np.sum(largest_component_mask) * voxel_volume
            features['surface_volume_ratio'] = surface_area / component_volume if component_volume > 0 else np.nan
            
            # Sphericity (how close to a sphere)
            # Sphericity = (π^(1/3) * (6*V)^(2/3)) / A
            # where V is volume and A is surface area
            if surface_area > 0:
                sphericity = (np.pi**(1/3) * (6 * component_volume)**(2/3)) / surface_area
                features['sphericity'] = min(sphericity, 1.0)  # Cap at 1.0 (perfect sphere)
            else:
                features['sphericity'] = np.nan
            
            # Compactness (inverse of sphericity, lower = more compact)
            # Compactness = A^3 / (36 * π * V^2)
            if component_volume > 0:
                compactness = (surface_area**3) / (36 * np.pi * component_volume**2)
                features['compactness'] = compactness
            else:
                features['compactness'] = np.nan
            
            # Calculate principal component analysis for elongation
            points = np.column_stack(np.where(largest_component_mask))
            if len(points) > 3:
                # Center the points
                centered = points - np.mean(points, axis=0)
                # Compute covariance matrix
                cov_matrix = np.cov(centered.T)
                # Get eigenvalues
                eigenvalues = np.linalg.eigvalsh(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
                
                # Elongation: ratio of largest to smallest eigenvalue
                if eigenvalues[-1] > 0:
                    features['elongation'] = np.sqrt(eigenvalues[0] / eigenvalues[-1])
                else:
                    features['elongation'] = np.nan
            else:
                features['elongation'] = np.nan
            
            # Solidity: ratio of actual volume to convex hull volume
            try:
                # Get convex hull of the component
                from skimage.morphology import convex_hull_image
                hull = convex_hull_image(largest_component_mask)
                hull_volume = np.sum(hull) * voxel_volume
                features['solidity'] = component_volume / hull_volume if hull_volume > 0 else np.nan
            except Exception as e:
                print(f"Warning: Could not calculate solidity: {e}")
                features['solidity'] = np.nan
                
        except Exception as e:
            print(f"Error calculating shape features: {e}")
            features['surface_area'] = np.nan
            features['surface_volume_ratio'] = np.nan
            features['sphericity'] = np.nan
            features['compactness'] = np.nan
            features['elongation'] = np.nan
            features['solidity'] = np.nan
    else:
        # Component too small for reliable shape analysis
        features['surface_area'] = np.nan
        features['surface_volume_ratio'] = np.nan
        features['sphericity'] = np.nan
        features['compactness'] = np.nan
        features['elongation'] = np.nan
        features['solidity'] = np.nan
    
    return features


def classify_enhancement_pattern(features):
    """
    Classify enhancement pattern based on radiomics features
    
    Parameters:
    -----------
    features : dict
        Dictionary of shape features
    
    Returns:
    --------
    str : Enhancement pattern classification
    """
    # No enhancement
    if features['volume'] == 0:
        return 'No Enhancement'
    
    # Multiple distinct lesions
    if features['n_components'] >= 3:
        return 'Multiple'
    
    # Two components - check if similar size (suggests multifocal) or very different (suggests satellite)
    if features['n_components'] == 2:
        if features['largest_component_ratio'] < 0.8:  # Second component is substantial
            return 'Multiple'
    
    # Single or dominant single lesion - classify by shape
    sphericity = features.get('sphericity', np.nan)
    compactness = features.get('compactness', np.nan)
    elongation = features.get('elongation', np.nan)
    solidity = features.get('solidity', np.nan)
    surface_volume_ratio = features.get('surface_volume_ratio', np.nan)
    
    # Well-circumscribed: high sphericity, low compactness, high solidity
    if not np.isnan(sphericity) and not np.isnan(solidity):
        if sphericity > 0.7 and solidity > 0.9:
            return 'Well-circumscribed'
        # Infiltrative: low sphericity, high surface/volume ratio, low solidity
        elif sphericity < 0.5 or solidity < 0.7:
            return 'Infiltrative'
        # Irregular but not clearly infiltrative
        else:
            return 'Irregular/Complex'
    
    # Fallback classification based on available features
    if not np.isnan(surface_volume_ratio):
        if surface_volume_ratio > 0.5:  # High surface area relative to volume
            return 'Infiltrative'
        else:
            return 'Well-circumscribed'
    
    return 'Unclassified'


def process_single_case(case_info):
    """
    Process a single case to extract enhancement pattern features
    
    Parameters:
    -----------
    case_info : tuple
        (case_id, gt_path, label_value, voxel_spacing)
    
    Returns:
    --------
    dict : Features and classification for the case
    """
    case_id, gt_path, label_value, voxel_spacing = case_info
    
    try:
        # Load ground truth
        gt_nii = nib.load(os.path.join(gt_path, f"{case_id}.nii.gz"))
        gt_img = gt_nii.get_fdata()
        
        # Extract enhancing tumor mask
        et_mask = (gt_img == label_value).astype(np.uint8)
        
        # Get voxel spacing from header if not provided
        if voxel_spacing is None:
            voxel_spacing = gt_nii.header.get_zooms()[:3]
        
        # Calculate shape features
        features = calculate_shape_features(et_mask, voxel_spacing)
        
        # Classify pattern
        pattern = classify_enhancement_pattern(features)
        
        # Add case info
        features['case_id'] = case_id
        features['enhancement_pattern'] = pattern
        
        return features
        
    except Exception as e:
        print(f"Error processing {case_id}: {e}")
        return {
            'case_id': case_id,
            'enhancement_pattern': 'Error',
            'volume': np.nan,
            'n_components': np.nan,
            'sphericity': np.nan,
            'compactness': np.nan,
            'elongation': np.nan,
            'solidity': np.nan
        }


def compute_enhancement_patterns(results_df, gt_path, label_value=3, n_jobs=None):
    """
    Compute enhancement patterns for all cases in results_df
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with case information
    gt_path : str
        Path to ground truth labels
    label_value : int
        Label value for enhancing tumor (default: 3)
    n_jobs : int
        Number of parallel jobs (default: CPU count - 1)
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with enhancement pattern features
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() - 1
    
    # Prepare cases for processing
    case_infos = [
        (row['case_id'], gt_path, label_value, None)
        for _, row in results_df.iterrows()
    ]
    
    print(f"Computing enhancement patterns for {len(case_infos)} cases...")
    
    # Process in parallel
    pattern_features = Parallel(n_jobs=n_jobs)(
        delayed(process_single_case)(case_info)
        for case_info in tqdm(case_infos, desc="Processing cases")
    )
    
    # Convert to DataFrame
    pattern_df = pd.DataFrame(pattern_features)
    
    # Merge with original results
    results_with_patterns = results_df.merge(pattern_df, on='case_id', how='left')
    
    # Print summary
    print("\nEnhancement Pattern Distribution:")
    print(results_with_patterns['enhancement_pattern'].value_counts())
    
    # Debug: Show some examples of each pattern
    print("\nPattern Examples (first 3 of each):")
    for pattern in ['Well-circumscribed', 'Infiltrative', 'Multiple', 'Irregular/Complex']:
        pattern_cases = results_with_patterns[results_with_patterns['enhancement_pattern'] == pattern]
        if len(pattern_cases) > 0:
            print(f"\n{pattern} (n={len(pattern_cases)}):")
            for idx, row in pattern_cases.head(3).iterrows():
                print(f"  Case {row['case_id']}: sph={row['sphericity']:.3f}, sol={row['solidity']:.3f}, comp={row['n_components']:.0f}")
    
    return results_with_patterns


def create_enhanced_subgroup_analysis_panel(results_with_patterns, ax):
    """
    Create the enhanced panel c for subgroup analysis using radiomics-derived patterns
    
    Parameters:
    -----------
    results_with_patterns : pandas.DataFrame
        Results DataFrame with enhancement pattern features
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    import seaborn as sns
    
    # Filter for cases with enhancement and valid patterns
    et_cases = results_with_patterns[
        (results_with_patterns['gt_volume'] > 0) & 
        (results_with_patterns['enhancement_pattern'] != 'Error') &
        (results_with_patterns['enhancement_pattern'] != 'No Enhancement')
    ].copy()
    
    # Calculate metrics by pattern
    pattern_metrics = et_cases.groupby('enhancement_pattern')['dice'].agg(['mean', 'std', 'count'])
    pattern_metrics = pattern_metrics.sort_values('mean', ascending=False)
    
    # Create bar plot
    colors_pie = sns.color_palette('husl', n_colors=10)
    
    x_pos = np.arange(len(pattern_metrics))
    means = pattern_metrics['mean'].values
    stds = pattern_metrics['std'].values
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=[colors_pie[i % len(colors_pie)] for i in range(len(pattern_metrics))])
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pattern_metrics.index, rotation=45, ha='right')
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('c) Performance by Enhancement Pattern (Radiomics-derived)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add sample sizes and additional metrics
    for i, (idx, row) in enumerate(pattern_metrics.iterrows()):
        count = int(row['count'])
        mean = row['mean']
        
        # Get additional metrics for this pattern
        pattern_cases = et_cases[et_cases['enhancement_pattern'] == idx]
        avg_sphericity = pattern_cases['sphericity'].mean()
        avg_components = pattern_cases['n_components'].mean()
        
        # Add text with count
        ax.text(i, mean + stds[i] + 0.02, f'n={count}', 
                ha='center', va='bottom', fontsize=9)
        
        # Add pattern characteristics below x-label
        char_text = f'Sph:{avg_sphericity:.2f}\nComp:{avg_components:.1f}'
        ax.text(i, -0.15, char_text, ha='center', va='top', 
                transform=ax.get_xaxis_transform(), fontsize=8, color='gray')
    
    return ax


# Example usage in notebook:
"""
# Load necessary libraries
import matplotlib.pyplot as plt
from radiomics_enhancement_pattern_analysis import compute_enhancement_patterns, create_enhanced_subgroup_analysis_panel

# Compute enhancement patterns
results_with_patterns = compute_enhancement_patterns(
    results_df, 
    gt_labels_path,
    label_value=3,  # Enhancing tumor label
    n_jobs=-1
)

# Save the enhanced results
results_with_patterns.to_csv(os.path.join(figures_out, 'results_with_radiomics_patterns.csv'), index=False)

# In your subgroup_analysis function, replace panel c with:
ax3 = fig.add_subplot(gs[0, 3])
create_enhanced_subgroup_analysis_panel(results_with_patterns, ax3)
"""