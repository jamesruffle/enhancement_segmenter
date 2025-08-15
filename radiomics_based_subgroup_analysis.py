#!/usr/bin/env python3
"""
Improved subgroup analysis using radiomics features to programmatically determine:
- Well-circumscribed vs infiltrative
- Single vs multiple lesions
- Compactness/sphericity metrics
"""

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure
import pandas as pd

def calculate_radiomics_features(label_img, voxel_spacing=(1, 1, 1)):
    """
    Calculate radiomics features for a given label image
    
    Parameters:
    -----------
    label_img : numpy array
        3D binary image of the enhancing tumor
    voxel_spacing : tuple
        Voxel spacing in mm for x, y, z dimensions
    
    Returns:
    --------
    dict : Dictionary of radiomics features
    """
    features = {}
    
    # Binary mask of enhancing tumor
    mask = (label_img > 0).astype(int)
    
    if np.sum(mask) == 0:
        # Return default values for empty mask
        return {
            'volume': 0,
            'surface_area': 0,
            'sphericity': 0,
            'compactness': 0,
            'elongation': 0,
            'n_components': 0,
            'largest_component_ratio': 0,
            'solidity': 0,
            'extent': 0
        }
    
    # 1. Number of connected components (for multiple lesions)
    labeled_array, n_components = ndimage.label(mask)
    features['n_components'] = n_components
    
    # 2. Volume of each component
    component_volumes = []
    for i in range(1, n_components + 1):
        component_mask = (labeled_array == i)
        volume = np.sum(component_mask) * np.prod(voxel_spacing)
        component_volumes.append(volume)
    
    if component_volumes:
        features['largest_component_ratio'] = max(component_volumes) / sum(component_volumes)
    else:
        features['largest_component_ratio'] = 0
    
    # For remaining features, use the largest connected component
    if n_components > 0:
        largest_component_id = np.argmax(component_volumes) + 1
        largest_mask = (labeled_array == largest_component_id)
    else:
        largest_mask = mask
    
    # 3. Calculate surface area using marching cubes
    try:
        verts, faces, _, _ = measure.marching_cubes(largest_mask.astype(float), level=0.5, spacing=voxel_spacing)
        surface_area = measure.mesh_surface_area(verts, faces)
        features['surface_area'] = surface_area
    except:
        features['surface_area'] = 0
    
    # 4. Volume
    volume = np.sum(largest_mask) * np.prod(voxel_spacing)
    features['volume'] = volume
    
    # 5. Sphericity (how sphere-like the shape is)
    # Sphericity = (pi^(1/3) * (6*V)^(2/3)) / A
    # Where V is volume and A is surface area
    if features['surface_area'] > 0:
        sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / features['surface_area']
        features['sphericity'] = min(sphericity, 1.0)  # Cap at 1.0 due to discretization errors
    else:
        features['sphericity'] = 0
    
    # 6. Compactness (alternative measure)
    # Compactness = V / (A^(3/2) / (6 * sqrt(pi)))
    if features['surface_area'] > 0:
        compactness = volume / (features['surface_area']**(3/2) / (6 * np.sqrt(np.pi)))
        features['compactness'] = min(compactness, 1.0)
    else:
        features['compactness'] = 0
    
    # 7. Calculate principal component analysis for elongation
    if np.sum(largest_mask) > 10:  # Need sufficient voxels
        # Get coordinates of all voxels in the mask
        coords = np.column_stack(np.where(largest_mask))
        
        # Apply voxel spacing
        coords_scaled = coords * np.array(voxel_spacing)
        
        # Center the coordinates
        coords_centered = coords_scaled - np.mean(coords_scaled, axis=0)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(coords_centered.T)
        
        # Get eigenvalues (sorted in descending order)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Elongation = ratio of largest to smallest eigenvalue
        if eigenvalues[-1] > 0:
            features['elongation'] = np.sqrt(eigenvalues[0] / eigenvalues[-1])
        else:
            features['elongation'] = 1.0
    else:
        features['elongation'] = 1.0
    
    # 8. Solidity (ratio of volume to convex hull volume)
    try:
        props = measure.regionprops(largest_mask.astype(int))[0]
        features['solidity'] = props.solidity
        features['extent'] = props.extent  # Ratio of pixels to bounding box
    except:
        features['solidity'] = 0
        features['extent'] = 0
    
    return features


def classify_tumor_pattern(features):
    """
    Classify tumor pattern based on radiomics features
    
    Returns:
    --------
    str : Classification of tumor pattern
    """
    # Multiple lesions
    if features['n_components'] > 2:
        return 'Multiple lesions'
    
    # Well-circumscribed vs infiltrative based on shape metrics
    if features['sphericity'] > 0.7 and features['compactness'] > 0.7 and features['solidity'] > 0.8:
        return 'Well-circumscribed'
    elif features['sphericity'] < 0.5 or features['compactness'] < 0.5 or features['solidity'] < 0.6:
        return 'Infiltrative'
    else:
        return 'Intermediate'


def analyze_subgroups_with_radiomics(results_df, gt_path, voxel_spacing=(1, 1, 1)):
    """
    Analyze subgroups using radiomics features from ground truth labels
    
    Parameters:
    -----------
    results_df : DataFrame
        Results dataframe with case information
    gt_path : str
        Path to ground truth label files
    voxel_spacing : tuple
        Voxel spacing in mm
    
    Returns:
    --------
    DataFrame : Updated results_df with radiomics features and pattern classification
    """
    import os
    
    # Initialize new columns
    radiomics_features = []
    pattern_classifications = []
    
    print("Calculating radiomics features for tumor pattern classification...")
    
    for idx, row in results_df.iterrows():
        case_id = row['case_id']
        
        # Skip cases without enhancing tumor
        if row['gt_volume'] == 0:
            radiomics_features.append({})
            pattern_classifications.append('No enhancing tumor')
            continue
        
        try:
            # Load ground truth
            gt_file = os.path.join(gt_path, f"{case_id}.nii.gz")
            if os.path.exists(gt_file):
                gt_img = nib.load(gt_file).get_fdata()
                
                # Extract enhancing tumor (label 3)
                et_mask = (gt_img == 3)
                
                # Calculate features
                features = calculate_radiomics_features(et_mask, voxel_spacing)
                radiomics_features.append(features)
                
                # Classify pattern
                pattern = classify_tumor_pattern(features)
                
                # Special cases
                if 'Paediatric' in row.get('Pathology', ''):
                    pattern = 'Pediatric'
                elif 'postoperative' in row.get('Pathology', '').lower():
                    pattern = 'Post-treatment'
                
                pattern_classifications.append(pattern)
                
                if idx % 100 == 0:
                    print(f"  Processed {idx}/{len(results_df)} cases...")
                
            else:
                radiomics_features.append({})
                pattern_classifications.append('File not found')
                
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            radiomics_features.append({})
            pattern_classifications.append('Error')
    
    # Add features to dataframe
    for feature_name in ['sphericity', 'compactness', 'n_components', 'elongation', 'solidity']:
        results_df[f'radiomics_{feature_name}'] = [f.get(feature_name, np.nan) for f in radiomics_features]
    
    results_df['tumor_pattern'] = pattern_classifications
    
    print("\nTumor pattern distribution:")
    print(results_df['tumor_pattern'].value_counts())
    
    return results_df


# Example usage for creating improved subgroup analysis
def create_improved_subgroup_analysis(results_df, gt_path):
    """
    Create improved subgroup analysis panel using radiomics-based classification
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Analyze radiomics features
    results_df = analyze_subgroups_with_radiomics(results_df, gt_path)
    
    # Filter for cases with enhancing tumor
    et_cases = results_df[results_df['gt_volume'] > 0].copy()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors_pie = sns.color_palette('husl', n_colors=10)
    
    # Panel A: Performance by radiomics-based pattern
    ax = axes[0, 0]
    pattern_metrics = et_cases.groupby('tumor_pattern')['dice'].agg(['mean', 'std', 'count'])
    pattern_metrics = pattern_metrics[pattern_metrics['count'] >= 5]  # At least 5 cases
    pattern_metrics = pattern_metrics.sort_values('mean', ascending=False)
    
    if len(pattern_metrics) > 0:
        x_pos = np.arange(len(pattern_metrics))
        means = pattern_metrics['mean'].values
        stds = pattern_metrics['std'].values
        counts = pattern_metrics['count'].values
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_pie[0])
        
        # Add sample size on bars
        for i, (mean, count) in enumerate(zip(means, counts)):
            ax.text(i, mean + 0.02, f'n={count}', ha='center', va='bottom')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pattern_metrics.index, rotation=45, ha='right')
        ax.set_ylabel('Dice Score')
        ax.set_title('a) Performance by Tumor Pattern (Radiomics-based)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Sphericity distribution by pattern
    ax = axes[0, 1]
    patterns_to_plot = ['Well-circumscribed', 'Infiltrative', 'Multiple lesions']
    for i, pattern in enumerate(patterns_to_plot):
        data = et_cases[et_cases['tumor_pattern'] == pattern]['radiomics_sphericity'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=20, alpha=0.6, label=f'{pattern} (n={len(data)})', 
                   color=colors_pie[i+1], density=True)
    
    ax.set_xlabel('Sphericity')
    ax.set_ylabel('Density')
    ax.set_title('b) Sphericity Distribution by Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel C: Compactness vs Dice score
    ax = axes[0, 2]
    scatter_data = et_cases[et_cases['radiomics_compactness'].notna()]
    scatter = ax.scatter(scatter_data['radiomics_compactness'], scatter_data['dice'],
                        c=scatter_data['radiomics_n_components'], cmap='viridis',
                        alpha=0.6, s=50)
    ax.set_xlabel('Compactness')
    ax.set_ylabel('Dice Score')
    ax.set_title('c) Compactness vs Performance')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Components')
    
    # Panel D: Number of components distribution
    ax = axes[1, 0]
    component_counts = et_cases['radiomics_n_components'].value_counts().sort_index()
    ax.bar(component_counts.index, component_counts.values, alpha=0.7, color=colors_pie[3])
    ax.set_xlabel('Number of Tumor Components')
    ax.set_ylabel('Number of Cases')
    ax.set_title('d) Distribution of Tumor Multiplicity')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel E: Performance vs tumor multiplicity
    ax = axes[1, 1]
    # Group by single vs multiple
    et_cases['is_multiple'] = et_cases['radiomics_n_components'] > 1
    mult_metrics = et_cases.groupby('is_multiple')['dice'].agg(['mean', 'std', 'count'])
    
    labels = ['Single lesion', 'Multiple lesions']
    means = mult_metrics['mean'].values
    stds = mult_metrics['std'].values
    counts = mult_metrics['count'].values
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors_pie[4])
    
    for i, (mean, count) in enumerate(zip(means, counts)):
        ax.text(i, mean + 0.02, f'n={count}', ha='center', va='bottom')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Dice Score')
    ax.set_title('e) Performance: Single vs Multiple Lesions')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel F: Shape metrics correlation
    ax = axes[1, 2]
    shape_data = et_cases[['radiomics_sphericity', 'radiomics_compactness', 
                          'radiomics_solidity', 'dice']].dropna()
    if len(shape_data) > 0:
        corr_matrix = shape_data.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('f) Shape Metrics Correlation')
    
    plt.suptitle('Radiomics-based Subgroup Analysis', fontsize=16)
    plt.tight_layout()
    
    return fig, results_df


print("=" * 80)
print("RADIOMICS-BASED SUBGROUP ANALYSIS")
print("=" * 80)
print()
print("This script provides functions to:")
print("1. Calculate radiomics features from ground truth labels:")
print("   - Sphericity: How sphere-like the tumor is (0-1)")
print("   - Compactness: Volume to surface area relationship")
print("   - Number of components: Identifies multiple lesions")
print("   - Elongation: Ratio of principal axes")
print("   - Solidity: Ratio to convex hull")
print()
print("2. Classify tumors based on these features:")
print("   - Well-circumscribed: High sphericity, compactness, solidity")
print("   - Infiltrative: Low shape metrics")
print("   - Multiple lesions: More than 2 connected components")
print("   - Post-treatment: Based on pathology (can't determine from imaging alone)")
print("   - Pediatric: Based on pathology")
print()
print("3. Create improved subgroup analysis visualizations")
print()
print("This approach is more objective than using pathology labels alone,")
print("as it directly measures the imaging characteristics of interest.")
print("=" * 80)