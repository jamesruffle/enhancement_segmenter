#!/usr/bin/env python3
"""
Radiomics-based Enhancement Pattern Analysis with Distance-Based Component Merging
Components must be more than 20 voxels apart to be considered separate
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.spatial import distance_matrix
from skimage import measure
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


def merge_close_components(labeled_mask, min_distance_voxels=20):
    """
    Merge components that are closer than min_distance_voxels.
    
    Parameters:
    -----------
    labeled_mask : numpy.ndarray
        Labeled array from ndimage.label
    min_distance_voxels : int
        Minimum distance in voxels between components to be considered separate
        
    Returns:
    --------
    merged_labels : numpy.ndarray
        Labeled array with merged components
    n_merged : int
        Number of components after merging
    """
    n_initial = labeled_mask.max()
    
    if n_initial <= 1:
        return labeled_mask, n_initial
    
    # Get component information
    components = []
    for i in range(1, n_initial + 1):
        coords = np.argwhere(labeled_mask == i)
        if len(coords) > 0:
            components.append({
                'label': i,
                'coords': coords,
                'size': len(coords),
                'bbox_min': coords.min(axis=0),
                'bbox_max': coords.max(axis=0)
            })
    
    n_components = len(components)
    if n_components <= 1:
        return labeled_mask, n_components
    
    # Initialize merge groups (each component starts in its own group)
    merge_groups = list(range(n_components))
    
    # Check distances between all pairs of components
    for i in range(n_components):
        for j in range(i + 1, n_components):
            # Quick check using bounding boxes
            bbox_distance = np.max([
                components[i]['bbox_min'] - components[j]['bbox_max'],
                components[j]['bbox_min'] - components[i]['bbox_max']
            ])
            
            if bbox_distance > min_distance_voxels:
                continue  # Bounding boxes are far apart
            
            # For efficiency, sample points if components are large
            coords1 = components[i]['coords']
            coords2 = components[j]['coords']
            
            # Subsample for large components to speed up distance calculation
            max_samples = 200
            if len(coords1) > max_samples:
                idx1 = np.random.choice(len(coords1), max_samples, replace=False)
                coords1 = coords1[idx1]
            if len(coords2) > max_samples:
                idx2 = np.random.choice(len(coords2), max_samples, replace=False)
                coords2 = coords2[idx2]
            
            # Calculate minimum distance between components
            distances = distance_matrix(coords1, coords2)
            min_distance = distances.min()
            
            if min_distance <= min_distance_voxels:
                # Merge these components
                group_i = merge_groups[i]
                group_j = merge_groups[j]
                
                if group_i != group_j:
                    # Merge all components in group_j into group_i
                    new_group = min(group_i, group_j)
                    old_group = max(group_i, group_j)
                    
                    for k in range(n_components):
                        if merge_groups[k] == old_group:
                            merge_groups[k] = new_group
    
    # Create merged label array
    merged_labels = np.zeros_like(labeled_mask)
    unique_groups = sorted(set(merge_groups))
    label_map = {group: idx + 1 for idx, group in enumerate(unique_groups)}
    
    for comp, group in zip(components, merge_groups):
        merged_labels[labeled_mask == comp['label']] = label_map[group]
    
    return merged_labels, len(unique_groups)


def calculate_shape_features(binary_mask, voxel_spacing=(1, 1, 1), min_distance_voxels=20):
    """Calculate shape features for a binary mask with distance-based component merging"""
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
        features['n_components_original'] = 0
        features['largest_component_ratio'] = np.nan
        return features
    
    # Get original connected components
    labeled_mask_original, n_components_original = ndimage.label(binary_mask)
    features['n_components_original'] = n_components_original
    
    # Apply distance-based merging
    labeled_mask, n_components = merge_close_components(labeled_mask_original, min_distance_voxels)
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
            if surface_area > 0:
                sphericity = (np.pi**(1/3) * (6 * component_volume)**(2/3)) / surface_area
                features['sphericity'] = min(sphericity, 1.0)  # Cap at 1.0 (perfect sphere)
            else:
                features['sphericity'] = np.nan
            
            # Compactness
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
            
            # Solidity - often fails for 3D, so we catch exception
            try:
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
    Uses distance-merged component count
    """
    # No enhancement
    if features['volume'] == 0:
        return 'No Enhancement'
    
    # Multiple distinct lesions (≥3 components after distance merging)
    if features['n_components'] >= 3:
        return 'Multiple'
    
    # Two substantial components (neither dominates)
    if features['n_components'] == 2:
        if features['largest_component_ratio'] < 0.85:  # Second component is at least 15% of total
            return 'Multiple'
    
    # Single or dominant single lesion - classify by shape
    sphericity = features.get('sphericity', np.nan)
    solidity = features.get('solidity', np.nan)
    surface_volume_ratio = features.get('surface_volume_ratio', np.nan)
    
    # If both sphericity and solidity are available, use both
    if not np.isnan(sphericity) and not np.isnan(solidity):
        # Well-circumscribed: round, smooth borders
        if sphericity > 0.7 and solidity > 0.9:
            return 'Well-circumscribed Single'
        
        # Infiltrative: irregular, spreading pattern
        elif sphericity < 0.5 or solidity < 0.7:
            return 'Infiltrative Single'
        
        # Irregular/Complex: in between
        else:
            return 'Irregular/Complex Single'
    
    # If only sphericity is available, use it with adjusted thresholds
    elif not np.isnan(sphericity):
        if sphericity > 0.75:  # Very round
            return 'Well-circumscribed Single'
        elif sphericity < 0.45:  # Very irregular
            return 'Infiltrative Single'
        else:  # Medium sphericity (0.45-0.75)
            return 'Irregular/Complex Single'
    
    # Fallback: use surface-to-volume ratio if available
    elif not np.isnan(surface_volume_ratio):
        if surface_volume_ratio < 0.4:  # Low surface area = smooth
            return 'Well-circumscribed Single'
        elif surface_volume_ratio > 0.6:  # High surface area = complex borders
            return 'Infiltrative Single'
        else:
            return 'Irregular/Complex Single'
    
    # Cannot classify due to missing features but is a single lesion
    return 'Single Lesion (Unclassified)'


def process_single_case(case_info):
    """Process a single case to extract enhancement pattern features"""
    case_id, gt_path, label_value, voxel_spacing, min_distance_voxels = case_info
    
    try:
        # Load ground truth
        gt_nii = nib.load(os.path.join(gt_path, f"{case_id}.nii.gz"))
        gt_img = gt_nii.get_fdata()
        
        # Extract enhancing tumor mask
        et_mask = (gt_img == label_value).astype(np.uint8)
        
        # Get voxel spacing from header if not provided
        if voxel_spacing is None:
            voxel_spacing = gt_nii.header.get_zooms()[:3]
        
        # Calculate shape features with distance-based merging
        features = calculate_shape_features(et_mask, voxel_spacing, min_distance_voxels)
        
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
            'n_components_original': np.nan,
            'sphericity': np.nan,
            'compactness': np.nan,
            'elongation': np.nan,
            'solidity': np.nan
        }


def compute_enhancement_patterns(results_df, gt_path, label_value=3, n_jobs=None, min_distance_voxels=20):
    """Compute enhancement patterns for all cases in results_df with distance-based merging"""
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() - 1
    
    # Prepare cases for processing
    case_infos = [
        (row['case_id'], gt_path, label_value, None, min_distance_voxels)
        for _, row in results_df.iterrows()
    ]
    
    print(f"Computing enhancement patterns for {len(case_infos)} cases...")
    print(f"Using distance threshold: {min_distance_voxels} voxels")
    
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
    
    # Print merging statistics
    merge_stats = results_with_patterns[['n_components_original', 'n_components']].describe()
    print("\nComponent Merging Statistics:")
    print(f"Cases where components were merged: {(results_with_patterns['n_components'] < results_with_patterns['n_components_original']).sum()}")
    print(f"Average reduction in component count: {(results_with_patterns['n_components_original'] - results_with_patterns['n_components']).mean():.2f}")
    
    # Print pattern characteristics
    print("\nPattern Characteristics (mean values):")
    patterns = ['Multiple', 'Well-circumscribed Single', 'Infiltrative Single', 
                'Irregular/Complex Single', 'Single Lesion (Unclassified)']
    for pattern in patterns:
        pattern_data = results_with_patterns[results_with_patterns['enhancement_pattern'] == pattern]
        if len(pattern_data) > 0:
            print(f"\n{pattern}:")
            print(f"  Sphericity: {pattern_data['sphericity'].mean():.3f} ± {pattern_data['sphericity'].std():.3f}")
            if pattern_data['solidity'].notna().any():
                print(f"  Solidity: {pattern_data['solidity'].mean():.3f} ± {pattern_data['solidity'].std():.3f}")
            print(f"  Components (after merging): {pattern_data['n_components'].mean():.1f} ± {pattern_data['n_components'].std():.1f}")
            print(f"  Components (original): {pattern_data['n_components_original'].mean():.1f} ± {pattern_data['n_components_original'].std():.1f}")
    
    return results_with_patterns