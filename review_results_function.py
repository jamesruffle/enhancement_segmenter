# %%
from joblib import Parallel, delayed
import random
import time
import pandas as pd
import json
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import subprocess
try:
    from IPython.display import clear_output
except ImportError:
    # Define a fallback clear_output function if not in IPython environment
    def clear_output(wait=False):
        os.system('cls' if os.name == 'nt' else 'clear')

def find_matching_prediction(base_name, predictions_dir, predictions_prefix=''):
    """Find matching prediction file for a given base name"""
    if not predictions_dir.endswith('/'):
        predictions_dir += '/'
    
    # Try common prediction file patterns
    possible_names = [
        f"{predictions_prefix}{base_name}.nii.gz",
        f"{predictions_prefix}{base_name}.nii"
    ]
    
    for name in possible_names:
        pred_path = predictions_dir + name
        if os.path.exists(pred_path):
            return pred_path
    
    return None

def process_single_case(flair_file, structural_data_dir, predictions_dir, ground_truth_dir, predictions_prefix, enhancing_label):
    """Process a single case - helper function for parallelization"""
    try:
        # Extract base name (remove _0000.nii.gz suffix)
        base_name = os.path.basename(flair_file)[:-12]  # Remove '_0000.nii.gz'
        
        # Check if T1 and T2 files exist
        t1_file = os.path.join(structural_data_dir, f"{base_name}_0001.nii.gz")
        t2_file = os.path.join(structural_data_dir, f"{base_name}_0002.nii.gz")
        
        if os.path.exists(t1_file) and os.path.exists(t2_file):
            # Find matching prediction
            pred_path = find_matching_prediction(base_name, predictions_dir, predictions_prefix)
            
            # Find matching ground truth
            gt_path = find_matching_prediction(base_name, ground_truth_dir, '')
            
            if pred_path and gt_path:
                # Load ground truth to get sum (for stratified sampling based on GT)
                try:
                    gt_img = nib.load(gt_path).get_fdata()
                    
                    # Filter for specific enhancing label if specified
                    if enhancing_label is not None:
                        gt_filtered = (gt_img == enhancing_label).astype(gt_img.dtype)
                        gt_sum = float(gt_filtered.sum())
                    else:
                        gt_sum = float(gt_img.sum())
                        
                except Exception as e:
                    gt_sum = 0.0
                    return None, f"Warning: Could not load ground truth for {base_name}: {e}"
                
                return {
                    'base_name': base_name,
                    'flair_file': flair_file,
                    't1_file': t1_file,
                    't2_file': t2_file,
                    'model_prediction': pred_path,
                    'ground_truth': gt_path,
                    'ground_truth_sum': gt_sum,
                    'enhancing_label': enhancing_label
                }, None
            else:
                warnings = []
                if not pred_path:
                    warnings.append(f"Warning: No matching prediction found for {base_name}")
                if not gt_path:
                    warnings.append(f"Warning: No matching ground truth found for {base_name}")
                return None, "; ".join(warnings)
        else:
            return None, f"Warning: Missing T1 or T2 file for {base_name}"
            
    except Exception as e:
        return None, f"Error processing {flair_file}: {e}"


def load_data(structural_data_dir, predictions_dir, ground_truth_dir, predictions_prefix='', enhancing_label=None):
    """Load structural data files and find matching predictions and ground truth"""
    import glob
    
    if not structural_data_dir.endswith('/'):
        structural_data_dir += '/'
    if not ground_truth_dir.endswith('/'):
        ground_truth_dir += '/'
    
    # Find all FLAIR files (_0000.nii.gz) as the base for each case
    flair_pattern = os.path.join(structural_data_dir, '*_0000.nii.gz')
    flair_files = glob.glob(flair_pattern)
    
    print(f"Processing {len(flair_files)} cases in parallel...")
    
    # Process cases in parallel
    results = Parallel(n_jobs=os.cpu_count())(
        delayed(process_single_case)(
            flair_file, structural_data_dir, predictions_dir, ground_truth_dir, 
            predictions_prefix, enhancing_label
        ) for flair_file in tqdm(flair_files, desc="Processing cases")
    )
    
    # Collect successful samples and warnings
    samples = []
    for result, warning in results:
        if result is not None:
            samples.append(result)
        elif warning:
            print(warning)
    
    print(f"Found {len(samples)} cases with matching predictions and ground truth")
    if enhancing_label is not None:
        print(f"Filtering for enhancing label: {enhancing_label}")
    return samples

def load_or_create_results_file(results_file):
    """Load existing results file or create empty structure"""
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded existing results file: {results_file}")
            return data
        else:
            print(f"Creating new results file: {results_file}")
            return {
                'session_info': {
                    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_samples': 0,
                    'sample_order': []
                },
                'results_without_seg': [],
                'results_with_seg': []
            }
    except Exception as e:
        print(f"Error loading results file {results_file}: {e}")
        return None

def get_completed_cases(results_data, load_seg):
    """Get list of base_names that have already been completed for the current mode"""
    key = 'results_with_seg' if load_seg else 'results_without_seg'
    return [result['sample']['base_name'] for result in results_data.get(key, [])]

def get_remaining_samples(samples, results_data, load_seg):
    """Get samples that still need to be completed for the current mode"""
    completed_cases = get_completed_cases(results_data, load_seg)
    
    # If we have a sample order from previous session, use it
    sample_order = results_data.get('session_info', {}).get('sample_order', [])
    
    if sample_order:
        # Follow the original sample order, skip completed cases
        remaining_samples = []
        for base_name in sample_order:
            if base_name not in completed_cases:
                # Find the sample with this base_name
                sample = next((s for s in samples if s['base_name'] == base_name), None)
                if sample:
                    remaining_samples.append(sample)
        print(f"Resuming from previous session. {len(remaining_samples)} samples remaining.")
        return remaining_samples
    else:
        # First time running, return all samples
        return samples


def visualize_and_annotate_N_samples(N, samples, cache_dir=None, reporter=None, years_experience=None, debug=False, outpath=None, load_seg=False, results_data=None, viewer='itksnap'):
    '''
    Visualize N random validation samples and collect user annotations.
    Args:
        N (int): Number of samples to review
        samples (list): List of sample dictionaries
        cache_dir (str): Directory for cache files
        reporter (str): Name of person doing the reporting
        years_experience (int): Years of experience of the reporter
        debug (bool): If True, print additional debug information
        outpath (str): Path to save results
        load_seg (bool): If True, load segmentation overlay in viewer
        results_data (dict): Existing results data structure
        viewer (str): Viewer to use ('itksnap' or 'fsleyes')
    Returns:
        results_data (dict): Updated results data structure
    '''
    # Get remaining samples for this mode (with/without segmentation)
    remaining_samples = get_remaining_samples(samples, results_data, load_seg)
    
    # If this is a new session, establish sample order and do stratified sampling
    if not results_data.get('session_info', {}).get('sample_order'):
        # Do stratified random sampling for new sessions
        zero_gt = [s for s in samples if s.get('ground_truth_sum', None) == 0]
        nonzero_gt = [s for s in samples if s.get('ground_truth_sum', None) and s['ground_truth_sum'] > 0]

        # Ensure exactly 50/50 split when possible
        n_half = N // 2
        available_zero = len(zero_gt)
        available_nonzero = len(nonzero_gt)
        
        if available_zero >= n_half and available_nonzero >= n_half:
            n_zero = n_half
            n_nonzero = n_half
        elif available_zero < n_half:
            n_zero = available_zero
            n_nonzero = min(N - n_zero, available_nonzero)
        else:
            n_nonzero = available_nonzero
            n_zero = min(N - n_nonzero, available_zero)

        selected_samples = random.sample(zero_gt, n_zero) + random.sample(nonzero_gt, n_nonzero)
        random.shuffle(selected_samples)
        
        # Store the sample order for consistency
        results_data['session_info']['sample_order'] = [s['base_name'] for s in selected_samples]
        results_data['session_info']['total_samples'] = len(selected_samples)
        
        print(f"New session: Reviewing {len(selected_samples)} samples ({n_zero} with no ground truth enhancement, {n_nonzero} with ground truth enhancement)")
        remaining_samples = selected_samples
    
    # Limit to N samples if specified
    if N and len(remaining_samples) > N:
        remaining_samples = remaining_samples[:N]
    
    print(f"Processing {len(remaining_samples)} samples ({'with' if load_seg else 'without'} segmentation overlay)")
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = '/home/jruffle/Downloads/cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    for idx, sample in enumerate(remaining_samples, 1):
        base_name = sample['base_name']
        flair_path = sample['flair_file']
        t1_path = sample['t1_file']
        t2_path = sample['t2_file']
        seg_path = sample['model_prediction']
        
        if debug:
            print(f'\nSample {idx}/{len(remaining_samples)}: {base_name}')
            print(f'Prediction path: {seg_path}')
            print(f'Files exist check:')
            print(f'  FLAIR: {os.path.exists(flair_path)}')
            print(f'  T1: {os.path.exists(t1_path)}')
            print(f'  T2: {os.path.exists(t2_path)}')
            print(f'  Prediction: {os.path.exists(seg_path)}')
            
        # Load the separate 3D volumes
        flair_img = nib.load(flair_path)
        t1_img = nib.load(t1_path)
        t2_img = nib.load(t2_path)
        
        # Save original images to cache directory (no intensity clipping)
        nib.save(flair_img, os.path.join(cache_dir, 'flair.nii.gz'))
        nib.save(t1_img, os.path.join(cache_dir, 't1.nii.gz'))
        nib.save(t2_img, os.path.join(cache_dir, 't2.nii.gz'))

        if load_seg:
            seg_img = nib.load(seg_path)
            seg_data = seg_img.get_fdata()
            
            if debug:
                print(f"Original segmentation shape: {seg_data.shape}")
                print(f"Original segmentation unique values: {np.unique(seg_data)}")
                print(f"Original segmentation sum: {seg_data.sum()}")
            
            # Filter for specific enhancing label if specified
            enhancing_label = sample.get('enhancing_label')
            if enhancing_label is not None:
                seg_filtered = (seg_data == enhancing_label).astype(np.uint8)
                if debug:
                    print(f"Filtering for label {enhancing_label}")
                    print(f"Filtered segmentation unique values: {np.unique(seg_filtered)}")
                    print(f"Filtered segmentation sum: {seg_filtered.sum()}")
                
                # If segmentation is all zeros, create a small visible marker for debugging
                if seg_filtered.sum() == 0:
                    print("Warning: No voxels found for the specified enhancing label!")
                    print("This case has no enhancing regions according to the prediction.")
                
                seg_filtered_img = nib.Nifti1Image(seg_filtered, seg_img.affine, seg_img.header)
                nib.save(seg_filtered_img, os.path.join(cache_dir, 'seg.nii.gz'))
            else:
                nib.save(seg_img, os.path.join(cache_dir, 'seg.nii.gz'))
            
            # Verify the saved file
            saved_seg_path = os.path.join(cache_dir, 'seg.nii.gz')
            if os.path.exists(saved_seg_path):
                if debug:
                    saved_seg = nib.load(saved_seg_path).get_fdata()
                    print(f"Saved segmentation file exists: {saved_seg_path}")
                    print(f"Saved segmentation sum: {saved_seg.sum()}")
                    print(f"Saved segmentation unique values: {np.unique(saved_seg)}")
            else:
                print(f"Warning: Segmentation file not created at {saved_seg_path}")
        
        # Create ITK-SNAP workspace file if loading segmentation
        def create_itksnap_workspace(cache_dir, include_segmentation=True):
            """Create an ITK-SNAP workspace file using your template with updated file paths"""
            template_path = '/home/jruffle/Downloads/workspace.itksnap'
            
            # Read your template workspace file
            try:
                with open(template_path, 'r') as f:
                    workspace_content = f.read()
                
                # Replace the file paths in the template with current cache directory paths
                # Use more specific replacements to avoid double cache/cache paths
                workspace_content = workspace_content.replace(
                    '/home/jruffle/Downloads/cache/flair.nii.gz',
                    os.path.join(cache_dir, 'flair.nii.gz')
                )
                workspace_content = workspace_content.replace(
                    '/home/jruffle/Downloads/cache/t1.nii.gz',
                    os.path.join(cache_dir, 't1.nii.gz')
                )
                workspace_content = workspace_content.replace(
                    '/home/jruffle/Downloads/cache/t2.nii.gz',
                    os.path.join(cache_dir, 't2.nii.gz')
                )
                workspace_content = workspace_content.replace(
                    '/home/jruffle/Downloads/cache/seg.nii.gz',
                    os.path.join(cache_dir, 'seg.nii.gz')
                )
                # Update SaveLocation to parent directory of cache_dir to avoid double paths
                parent_dir = os.path.dirname(cache_dir)
                workspace_content = workspace_content.replace(
                    'value="/home/jruffle/Downloads"',
                    f'value="{parent_dir}"'
                )
                
                # If not including segmentation, create a simplified workspace without segmentation
                if not include_segmentation:
                    # Create a minimal workspace with just the anatomical images
                    simple_workspace = f"""<?xml version="1.0" encoding="UTF-8" ?>
<!--ITK-SNAP (itksnap.org) Project File-->
<!DOCTYPE registry [
<!ELEMENT registry (entry*,folder*)>
<!ELEMENT folder (entry*,folder*)>
<!ELEMENT entry EMPTY>
<!ATTLIST folder key CDATA #REQUIRED>
<!ATTLIST entry key CDATA #REQUIRED>
<!ATTLIST entry value CDATA #REQUIRED>
]>
<registry>
  <entry key="SaveLocation" value="{parent_dir}" />
  <entry key="Version" value="20241202" />
  <folder key="Layers" >
    <folder key="Layer[000]" >
      <entry key="AbsolutePath" value="{os.path.join(cache_dir, 'flair.nii.gz')}" />
      <entry key="Role" value="MainRole" />
      <entry key="Tags" value="" />
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="255" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="0" />
        <entry key="Tags" value="" />
      </folder>
      <folder key="ProjectMetaData" >
        <folder key="IOHistory" >
          <folder key="AnatomicImage" >
            <entry key="ArraySize" value="2" />
            <entry key="Element[0]" value="{os.path.join(cache_dir, 't1.nii.gz')}" />
            <entry key="Element[1]" value="{os.path.join(cache_dir, 't2.nii.gz')}" />
          </folder>
        </folder>
      </folder>
    </folder>
    <folder key="Layer[001]" >
      <entry key="AbsolutePath" value="{os.path.join(cache_dir, 't1.nii.gz')}" />
      <entry key="Role" value="OverlayRole" />
      <entry key="Tags" value="" />
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="0.5" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="0" />
        <entry key="Tags" value="" />
      </folder>
    </folder>
    <folder key="Layer[002]" >
      <entry key="AbsolutePath" value="{os.path.join(cache_dir, 't2.nii.gz')}" />
      <entry key="Role" value="OverlayRole" />
      <entry key="Tags" value="" />
      <folder key="LayerMetaData" >
        <entry key="Alpha" value="0.5" />
        <entry key="CustomNickName" value="" />
        <entry key="Sticky" value="0" />
        <entry key="Tags" value="" />
      </folder>
    </folder>
  </folder>
</registry>"""
                    workspace_content = simple_workspace
                
                # Save the modified workspace file
                workspace_path = os.path.join(cache_dir, 'workspace.itksnap')
                with open(workspace_path, 'w') as f:
                    f.write(workspace_content)
                
                return workspace_path
                
            except Exception as e:
                print(f"Error reading template workspace file: {e}")
                print("Falling back to simple ITK-SNAP command")
                return None

        # Create a function to launch FSLeyes
        def launch_fsleyes(cache_dir, load_seg=load_seg):
            """Launch FSLeyes with anatomical images and optional segmentation"""
            cmd = [
                'fsleyes',
                os.path.join(cache_dir, 't2.nii.gz'),
                os.path.join(cache_dir, 't1.nii.gz'),
                os.path.join(cache_dir, 'flair.nii.gz')
            ]
            
            # Add segmentation overlay if requested
            if load_seg:
                seg_file = os.path.join(cache_dir, 'seg.nii.gz')
                if os.path.exists(seg_file):
                    cmd.extend([
                        seg_file,
                        '--alpha', '30',  # 30% opacity
                        '--cmap', 'hot'   # Hot colormap for segmentation
                    ])
            
            try:
                if debug:
                    print(f"FSLeyes command: {' '.join(cmd)}")
                # Launch FSLeyes as a subprocess in a new process group for easier termination
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                      preexec_fn=os.setsid)
                return proc
            except FileNotFoundError:
                raise RuntimeError("FSLeyes not found. Please ensure it's installed and in your PATH.")
            except Exception as e:
                raise RuntimeError(f"Failed to launch FSLeyes: {e}")

        # Create a function to launch ITK-SNAP
        def launch_itksnap(cache_dir, load_seg=load_seg):
            # Check if required files exist
            required_files = [os.path.join(cache_dir, 'flair.nii.gz'),
                            os.path.join(cache_dir, 't1.nii.gz'),
                            os.path.join(cache_dir, 't2.nii.gz')]
            
            if load_seg:
                required_files.append(os.path.join(cache_dir, 'seg.nii.gz'))
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required file not found: {file_path}")
            
            # Always create workspace file for consistent window sizing
            workspace_path = create_itksnap_workspace(cache_dir, include_segmentation=load_seg)
            
            if workspace_path:
                # Command to launch ITK-SNAP with workspace file and window geometry
                cmd = [
                    'itksnap',
                    '-w', workspace_path,
                    '--geometry', '1200x800+100+100'  # width x height + x_offset + y_offset
                ]
            else:
                # Fallback to simple command if workspace creation failed
                if not load_seg:
                    cmd = [
                        'itksnap',
                        '-g', os.path.join(cache_dir, 'flair.nii.gz'),
                        '-o', os.path.join(cache_dir, 't1.nii.gz'), 
                        os.path.join(cache_dir, 't2.nii.gz'),
                        '--geometry', '1200x800+100+100'
                    ]
                else:
                    cmd = [
                        'itksnap',
                        '-g', os.path.join(cache_dir, 'flair.nii.gz'),
                        '-o', os.path.join(cache_dir, 't1.nii.gz'), 
                        os.path.join(cache_dir, 't2.nii.gz'),
                        '-s', os.path.join(cache_dir, 'seg.nii.gz'),
                        '--geometry', '1200x800+100+100'
                    ]
            
            try:
                if debug:
                    print(f"ITK-SNAP command: {' '.join(cmd)}")
                # Launch ITK-SNAP as a subprocess
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return proc
            except FileNotFoundError:
                raise RuntimeError("ITK-SNAP not found. Please ensure it's installed and in your PATH.")
            except Exception as e:
                raise RuntimeError(f"Failed to launch ITK-SNAP: {e}")
        

        # Launch the selected viewer for the current sample
        viewer_proc = None
        try:
            if viewer == 'fsleyes':
                viewer_proc = launch_fsleyes(cache_dir)
                print(f"FSLeyes launched successfully (PID: {viewer_proc.pid})")
                if load_seg:
                    print("Segmentation loaded with 30% opacity and hot colormap")
            else:  # itksnap
                viewer_proc = launch_itksnap(cache_dir)
                print(f"ITK-SNAP launched successfully (PID: {viewer_proc.pid})")
                if load_seg:
                    print("NOTE: If segmentation is not visible, adjust opacity in ITK-SNAP:")
                    print("      Go to Segmentation panel → Opacity slider → Set to ~30%")
                else:
                    print("NOTE: To adjust image contrast/brightness in ITK-SNAP:")
                    print("      - Right-click on any image layer in the Layer Inspector")
                    print("      - Select 'Adjust Contrast' or use the intensity curve controls")
                    print("      - Or use Window/Level adjustment tools in the toolbar")
            
            start_time = time.time()
            resp1 = input('Do you think there will be an enhancing abnormality in this case? (Y/N): ').strip().upper()
            while resp1 not in ['Y', 'N']:
                resp1 = input('Please enter Y or N: ').strip().upper()
            resp2 = input('How confident are you? (1-10) [1=very unconfident, 10=very confident]: ').strip()
            while resp2 not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                resp2 = input('Please enter a number from 1 (very unconfident) to 10 (very confident): ').strip()
            resp3 = input('What is your assessment of image quality (1-10)? [1=worst, 10=best]: ').strip()
            while resp3 not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                resp3 = input('Please enter a number from 1 (worst quality) to 10 (best quality): ').strip()
            elapsed = time.time() - start_time
            
        except Exception as e:
            print(f"Error with {viewer}: {e}")
            print(f"Continuing without {viewer} visualization...")
            
            start_time = time.time()
            resp1 = input('Do you think there will be an enhancing abnormality in this case? (Y/N): ').strip().upper()
            while resp1 not in ['Y', 'N']:
                resp1 = input('Please enter Y or N: ').strip().upper()
            resp2 = input('How confident are you? (1-10) [1=very unconfident, 10=very confident]: ').strip()
            while resp2 not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                resp2 = input('Please enter a number from 1 (very unconfident) to 10 (very confident): ').strip()
            resp3 = input('What is your assessment of image quality (1-10)? [1=worst, 10=best]: ').strip()
            while resp3 not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                resp3 = input('Please enter a number from 1 (worst quality) to 10 (best quality): ').strip()
            elapsed = time.time() - start_time
        
        finally:
            # Terminate viewer process if it was launched
            if viewer_proc and viewer_proc.poll() is None:
                try:
                    if viewer == 'fsleyes':
                        # FSLeyes may spawn child processes, so we need to kill the process group
                        import signal
                        try:
                            os.killpg(os.getpgid(viewer_proc.pid), signal.SIGTERM)
                            viewer_proc.wait(timeout=3)
                            print(f"FSLeyes process group terminated")
                        except (ProcessLookupError, subprocess.TimeoutExpired):
                            try:
                                os.killpg(os.getpgid(viewer_proc.pid), signal.SIGKILL)
                                print(f"FSLeyes process group forcefully killed")
                            except ProcessLookupError:
                                pass
                    else:
                        # ITK-SNAP termination
                        viewer_proc.terminate()
                        viewer_proc.wait(timeout=5)
                        print(f"{viewer} process {viewer_proc.pid} terminated")
                except subprocess.TimeoutExpired:
                    viewer_proc.kill()
                    print(f"{viewer} process {viewer_proc.pid} forcefully killed")
                except Exception as e:
                    print(f"Error terminating {viewer}: {e}")
        
        # Add result to appropriate list in results_data
        result_entry = {
            'sample': sample,
            'abnormality': resp1,
            'confidence': int(resp2),
            'image_quality': int(resp3),
            'response_time': elapsed,
            'reporter': reporter,
            'years_experience': years_experience,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'segmentation_shown': load_seg
        }
        
        key = 'results_with_seg' if load_seg else 'results_without_seg'
        results_data[key].append(result_entry)
        
        # Save updated results to JSON file after each sample
        if outpath:
            # Only create directory if outpath has a directory component
            outpath_dir = os.path.dirname(outpath)
            if outpath_dir:  # Only if directory is not empty string
                os.makedirs(outpath_dir, exist_ok=True)
            # We need to handle numpy types for JSON serialization
            try:
                with open(outpath, 'w') as f:
                    json.dump(results_data, f, indent=4, default=lambda x: str(x) if isinstance(x, (np.integer, np.floating, np.ndarray)) else x)
                
                completed_without = len(results_data.get('results_without_seg', []))
                completed_with = len(results_data.get('results_with_seg', []))
                total_samples = results_data.get('session_info', {}).get('total_samples', 0)
                
                print(f"Progress saved: {completed_without}/{total_samples} without seg, {completed_with}/{total_samples} with seg")
            except Exception as e:
                print(f"Error saving results: {e}")
        
        # Clean up temporary files for next sample
        try:
            for temp_file in ['flair.nii.gz', 't1.nii.gz', 't2.nii.gz', 'seg.nii.gz']:
                temp_path = os.path.join(cache_dir, temp_file)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")
            
        plt.close('all')
        clear_output(wait=True)
            
    return results_data
    

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Review medical images and collect annotations for enhancement prediction validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--structural_data_dir', type=str, 
                       default='/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset002_enhance_and_abnormality/imagesTs/',
                       help='Directory containing structural MRI data (T1, T2, FLAIR) with nnUNet naming')
    parser.add_argument('--predictions_dir', type=str, 
                       default='/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset002_enhance_and_abnormality/predictionsTs_PP/',
                       help='Directory containing model predictions')
    parser.add_argument('--ground_truth_dir', type=str, 
                       default='/media/jruffle/DATA/nnUNet/nnUNet_raw/Dataset002_enhance_and_abnormality/labelsTs/',
                       help='Directory containing ground truth segmentations for stratified sampling')
    parser.add_argument('--enhancing_label', type=int, default=3,
                       help='Specific label value for enhancing tumor (filters out all other labels)')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of samples to review')
    parser.add_argument('--reporter', type=str, required=True,
                       help='Name of person doing the reporting')
    parser.add_argument('--years_experience', type=int, required=True,
                       help='Years of experience of the reporter')
    parser.add_argument('--outpath', type=str, required=True,
                       help='Path to save results JSON file')
    parser.add_argument('--predictions_prefix', type=str, default='',
                       help='Prefix for prediction filenames (default: no prefix)')
    parser.add_argument('--cache_dir', type=str,
                       help='Directory for temporary cache files (default: /home/jruffle/Downloads/cache)')
    parser.add_argument('--load_seg', action='store_true',
                       help='Load segmentation overlay')
    parser.add_argument('--viewer', type=str, choices=['itksnap', 'fsleyes'], default='itksnap',
                       help='Medical image viewer to use (default: itksnap)')
    parser.add_argument('--debug', action='store_true',
                       help='Print additional debug information')
    
    return parser.parse_args()


def main():
    """Main function to run the review process"""
    args = parse_args()
    
    print(f"Loading structural data from: {args.structural_data_dir}")
    print(f"Loading predictions from: {args.predictions_dir}")
    print(f"Loading ground truth from: {args.ground_truth_dir}")
    if args.predictions_prefix:
        print(f"Using predictions prefix: '{args.predictions_prefix}'")
    if args.enhancing_label is not None:
        print(f"Filtering for enhancing label: {args.enhancing_label}")
    
    try:
        # Load or create results file
        results_data = load_or_create_results_file(args.outpath)
        if results_data is None:
            print("Failed to load or create results file.")
            return 1
            
        # Check if trying to run with segmentation before completing without
        if args.load_seg:
            if not os.path.exists(args.outpath):
                print("\n❌ ERROR: Cannot run with --load_seg before running without segmentation.")
                print("   Please first run without --load_seg to establish baseline results.")
                return 1
            
            total_samples = results_data.get('session_info', {}).get('total_samples', 0)
            completed_without = len(results_data.get('results_without_seg', []))
            
            if total_samples == 0 or completed_without < total_samples:
                print("\n❌ ERROR: Cannot run with --load_seg before completing all cases without segmentation.")
                print(f"   Progress: {completed_without}/{total_samples} cases completed without segmentation.")
                print("   Please complete all cases without --load_seg first.")
                return 1
                
        # Load sample data
        samples = load_data(args.structural_data_dir, args.predictions_dir, args.ground_truth_dir, 
                           args.predictions_prefix, args.enhancing_label)
        print(f"Loaded {len(samples)} samples with matching predictions and ground truth")
        if len(samples) == 0:
            print("No samples found! Check your data, prediction, and ground truth directories.")
            return 1
        
        # Show stratification info
        zero_gt = [s for s in samples if s.get('ground_truth_sum', None) == 0]
        nonzero_gt = [s for s in samples if s.get('ground_truth_sum', None) and s['ground_truth_sum'] > 0]
        if args.enhancing_label is not None:
            print(f"Available samples for label {args.enhancing_label}: {len(zero_gt)} with no enhancement, {len(nonzero_gt)} with enhancement")
        else:
            print(f"Available samples: {len(zero_gt)} with no ground truth enhancement, {len(nonzero_gt)} with ground truth enhancement")
        
        # Show current progress
        completed_without = len(results_data.get('results_without_seg', []))
        completed_with = len(results_data.get('results_with_seg', []))
        total_samples = results_data.get('session_info', {}).get('total_samples', 0)
        
        if total_samples > 0:
            print(f"Current progress: {completed_without}/{total_samples} without seg, {completed_with}/{total_samples} with seg")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    print(f"Starting review process...")
    print(f"Reporter: {args.reporter}")
    print(f"Years of experience: {args.years_experience}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Output path: {args.outpath}")
    print(f"Cache directory: {args.cache_dir or '/home/jruffle/Downloads/cache'}")
    print(f"Viewer: {args.viewer}")
    print(f"Load segmentation: {args.load_seg}")
    
    try:
        updated_results_data = visualize_and_annotate_N_samples(
            N=args.n_samples,
            samples=samples,
            cache_dir=args.cache_dir,
            reporter=args.reporter,
            years_experience=args.years_experience,
            debug=args.debug,
            outpath=args.outpath,
            load_seg=args.load_seg,
            results_data=results_data,
            viewer=args.viewer
        )
        
        # Final progress summary
        completed_without = len(updated_results_data.get('results_without_seg', []))
        completed_with = len(updated_results_data.get('results_with_seg', []))
        total_samples = updated_results_data.get('session_info', {}).get('total_samples', 0)
        
        print(f"\nReview session completed!")
        print(f"Results saved to: {args.outpath}")
        print(f"Final progress: {completed_without}/{total_samples} without seg, {completed_with}/{total_samples} with seg")
        
        if completed_without == total_samples and completed_with == total_samples:
            print("🎉 All samples completed for both conditions!")
        elif completed_without == total_samples:
            print("✅ All samples completed without segmentation. Run with --load_seg to complete with segmentation.")
        elif completed_with == total_samples:
            print("✅ All samples completed with segmentation. Run without --load_seg to complete without segmentation.")
        else:
            remaining_without = total_samples - completed_without
            remaining_with = total_samples - completed_with
            print(f"📝 Remaining: {remaining_without} without seg, {remaining_with} with seg")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nReview interrupted by user - progress has been saved!")
        return 1
    except Exception as e:
        print(f"Error during review: {e}")
        return 1


if __name__ == '__main__':
    exit(main())


# %%
# Example usage for Jupyter notebook:
# samples = load_data('/path/to/structural/data', '/path/to/predictions', '/path/to/ground_truth', enhancing_label=1)
# results = visualize_and_annotate_N_samples(1, samples, reporter='James', years_experience=5, outpath='results.json')
