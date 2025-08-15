#!/usr/bin/env python3
"""
Radiomics-based Enhancement Pattern Analysis - Improved Version
Derives enhancement patterns from ground truth labels using shape and morphological features

Enhancement Pattern Definitions:
1. Well-circumscribed: Single, round/spherical lesion with smooth borders
   - High sphericity (>0.7) - close to perfect sphere
   - High solidity (>0.9) - fills its convex hull
   - Single component or dominant single component
   
2. Infiltrative: Irregular, spreading pattern with fuzzy borders
   - Low sphericity (<0.5) - far from spherical
   - Low solidity (<0.7) - doesn't fill convex hull
   - High surface-to-volume ratio - complex borders
   
3. Multiple: Several distinct lesions (changed from Multiple/Discrete)
   - 3 or more separate components
   - OR 2 components where neither dominates (both substantial)
   
4. Irregular/Complex: Single lesion that's neither well-circumscribed nor infiltrative
   - Medium sphericity (0.5-0.7)
   - Medium solidity (0.7-0.9)
   - Complex shape but contained
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
            
            # Solidity: ratio of convex hull volume to actual volume
            try:
                # Get convex hull of the component
                hull = measure.convex_hull_image(largest_component_mask)
                hull_volume = np.sum(hull) * voxel_volume
                features['solidity'] = component_volume / hull_volume if hull_volume > 0 else np.nan
            except:
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
    
    Classification logic:
    1. Multiple: ≥3 components OR 2 substantial components
    2. Well-circumscribed: Single round lesion (high sphericity & solidity)
    3. Infiltrative: Irregular spreading pattern (low sphericity OR low solidity)
    4. Irregular/Complex: Between well-circumscribed and infiltrative
    5. Unclassified: When features cannot be calculated
    
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
    
    # Multiple distinct lesions (≥3 components)
    if features['n_components'] >= 3:
        return 'Multiple'
    
    # Two substantial components (neither dominates)
    if features['n_components'] == 2:
        if features['largest_component_ratio'] < 0.85:  # Second component is at least 15% of total
            return 'Multiple'
    
    # Single or dominant single lesion - classify by shape
    sphericity = features.get('sphericity', np.nan)
    solidity = features.get('solidity', np.nan)
    elongation = features.get('elongation', np.nan)
    surface_volume_ratio = features.get('surface_volume_ratio', np.nan)
    
    # Classification based on shape metrics
    if not np.isnan(sphericity) and not np.isnan(solidity):
        # Well-circumscribed: round, smooth borders
        if sphericity > 0.7 and solidity > 0.9:
            return 'Well-circumscribed'
        
        # Infiltrative: irregular, spreading pattern
        elif sphericity < 0.5 or solidity < 0.7:
            return 'Infiltrative'
        
        # Irregular/Complex: in between
        else:
            return 'Irregular/Complex'
    
    # Fallback classification based on surface-to-volume ratio
    if not np.isnan(surface_volume_ratio):
        if surface_volume_ratio > 0.5:  # High surface area = complex borders
            return 'Infiltrative'
        else:
            return 'Well-circumscribed'
    
    # Cannot classify due to missing features
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
    
    # Print pattern characteristics
    print("\nPattern Characteristics (mean values):")
    for pattern in ['Well-circumscribed', 'Infiltrative', 'Multiple', 'Irregular/Complex']:
        pattern_data = results_with_patterns[results_with_patterns['enhancement_pattern'] == pattern]
        if len(pattern_data) > 0:
            print(f"\n{pattern}:")
            print(f"  Sphericity: {pattern_data['sphericity'].mean():.3f} ± {pattern_data['sphericity'].std():.3f}")
            print(f"  Solidity: {pattern_data['solidity'].mean():.3f} ± {pattern_data['solidity'].std():.3f}")
            print(f"  Components: {pattern_data['n_components'].mean():.1f} ± {pattern_data['n_components'].std():.1f}")
    
    return results_with_patterns