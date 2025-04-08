import numpy as np
import mne
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
import matplotlib.pyplot as plt
from config import PICK_CHANNELS
import os

def create_info():
    """Create MNE info structure with our channel configuration"""
    # Create montage
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Create info structure
    info = mne.create_info(
        ch_names=list(PICK_CHANNELS),
        sfreq=250,  # This doesn't matter for plotting
        ch_types=['eeg'] * len(PICK_CHANNELS)
    )
    
    # Set montage
    info.set_montage(montage)
    
    return info

def compute_csp_patterns_for_subject(X_subject, y_subject):
    """Compute CSP patterns for a single subject"""
    # Compute covariances
    cov = Covariances(estimator="oas")
    X_cov = cov.transform(X_subject)
    
    # Fit CSP
    csp = CSP(nfilter=4, metric='riemann', log=True)
    csp.fit(X_cov, y_subject)
    
    # Get patterns (spatial filters)
    patterns = csp.patterns_
    
    return patterns

def plot_csp_patterns(patterns, info, subject_id, output_dir):
    """Plot CSP patterns for a subject using MNE topomap"""
    n_components = patterns.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
    if n_components == 1:
        axes = [axes]
    
    # Plot each pattern
    vmax = np.abs(patterns).max()  # Symmetrical scaling
    for idx, ax in enumerate(axes):
        pattern = patterns[idx]
        
        # Plot topomap directly
        mne.viz.plot_topomap(
            pattern, 
            info, 
            axes=ax,
            show=False,
            contours=6,
            sensors=True,
            names=list(PICK_CHANNELS),  # Use our channel names
            # vmin=-vmax,
            # vmax=vmax
        )
        ax.set_title(f'CSP Pattern {idx+1}')
    
    plt.suptitle(f'Subject {subject_id} - CSP Patterns')
    
    # Save figure
    output_path = os.path.join(output_dir, f'{subject_id}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Create output directory
    output_dir = 'csp_patterns'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X = np.load('X.npy')
    y = np.load('y.npy')
    subject_ids = np.load('subject_ids.npy')
    
    # Remove class 2 (as in training.py)
    X = X[y != 2]
    subject_ids = subject_ids[y != 2]
    y = y[y != 2]
    
    # Create MNE info structure
    info = create_info()
    
    # Process each subject
    for subject_id in np.unique(subject_ids):
        print(f"Processing subject {subject_id}...")
        
        # Get subject data
        subject_mask = subject_ids == subject_id
        X_subject = X[subject_mask]
        y_subject = y[subject_mask]
        
        # Compute CSP patterns
        patterns = compute_csp_patterns_for_subject(X_subject, y_subject)
        
        # Plot and save patterns
        plot_csp_patterns(patterns, info, subject_id, output_dir)
    
    print(f"\nCSP pattern plots have been saved to {output_dir}/")

if __name__ == "__main__":
    main()

