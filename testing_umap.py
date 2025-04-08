import umap
import numpy as np
import matplotlib.pyplot as plt
from archive.dsygvd import riemann_metric, riemann_metric2
from src.brainbuilding.training import compute_normalized_augmented_covariances
from pyriemann.utils.distance import distance_riemann
from sklearn.manifold import MDS
import os
from matplotlib.colors import Normalize
from tqdm import tqdm

X = np.load('X.npy')
y = np.load('y.npy')
subjects = np.load('subject_ids.npy')
sample_weights = np.load('sample_weights.npy')

X = X[y != 2]
X = compute_normalized_augmented_covariances(X)
subjects = subjects[y != 2]
sample_weights = sample_weights[y != 2]
y = y[y != 2]

distances_arr = []
# Create directory if it doesn't exist
os.makedirs('umap-plots', exist_ok=True)

for subject in tqdm(np.sort(np.unique(subjects))):
    mask = (subjects == subject)
    samples = X[mask]
    # samples = samples.reshape(samples.shape[0], -1)
    distances = np.zeros((samples.shape[0], samples.shape[0]))
    for i in range(samples.shape[0]):
        for j in range(i+1, samples.shape[0]):
            distances[i, j] = distance_riemann(samples[i], samples[j])
    # mirror the matrix preserving the diagonal
    distances = distances + distances.T - np.diag(np.diag(distances))
    # print(f'{np.any(np.isinf(distances))=}')
    distances_arr.append(distances)
    trans = umap.UMAP(n_neighbors=1000, metric='precomputed', n_epochs=200).fit(distances).embedding_
    
    plt.figure(figsize=(10, 10))
    
    # Split data into class 0 and 1
    y_mask = (y[mask] == 0)
    sample_weights_subject = sample_weights[mask]
    
    # Plot class 0 with pastel green
    plt.scatter(trans[y_mask, 0], trans[y_mask, 1], 
                s=5, c='#98FB98', label='Class 0')
    
    # Plot class 1 with plasma colormap using sample weights
    sc = plt.scatter(trans[~y_mask, 0], trans[~y_mask, 1], 
                     s=5, 
                     c=sample_weights_subject[~y_mask], 
                     cmap='plasma', 
                     norm=Normalize(vmin=0, 
                                   vmax=1),
                     label='Class 1')
    
    plt.colorbar(sc, label='Sample Weight (Class 1)')
    plt.title(f'{subject} Embedded samples colored by class/weights', fontsize=18)
    plt.legend()
    
    # Save as PDF
    plt.savefig(f'umap-plots/{subject}.pdf', bbox_inches='tight', format='pdf')
    plt.close()  # Important to prevent memory leaks

np.save('distances_arr.npy', distances_arr)