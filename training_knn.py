import warnings
from scipy.linalg import eigvalsh
from tqdm import tqdm
import pandas as pd
import pyneurostim as ns
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import os
from scipy import interpolate
from pyriemann.estimation import Covariances
from sklearn.neighbors import KNeighborsClassifier
from pyriemann.utils.distance import distance_riemann
from scipy.spatial.distance import pdist, cdist, squareform
import umap

from typing import TypedDict

# # Bayesian optimization
# from skopt import BayesSearchCV
# from skopt.space import Real, Integer, Categorical

import matplotlib.pyplot as plt
import pickle
from eye_removal import EyeRemoval, create_standard_eog_channels
from config import PICK_CHANNELS, ND_CHANNELS_MASK, ORDER, STANDARD_EVENT_NAME_TO_ID_MAPPING, CSP_METRIC

import glob
import pathlib
from multiprocessing import Pool, cpu_count
from itertools import combinations
from joblib import Parallel, delayed

warnings.filterwarnings('ignore') 

class SubjectRaw(TypedDict):
    subject_id: int
    raw_data: mne.io.Raw

def load_training_data() -> list[SubjectRaw]:
    """Load training data from training_data directory"""
    training_files = glob.glob('training_data/*.fif')
    result = []
    
    for file_path in training_files:
        # Extract subject_id from filename
        subject_id = int(pathlib.Path(file_path).stem)
        # Load raw data
        raw = mne.io.read_raw_fif(file_path, preload=True)
        result.append(SubjectRaw(subject_id=subject_id, raw_data=raw))
    
    return result

def load_validation_data() -> list[SubjectRaw]:
    """Load validation data from validation_data directory"""
    validation_files = glob.glob('validation_data/*.fif')
    result = []
    
    for file_path in validation_files:
        # Extract subject_id from filename
        subject_id = int(pathlib.Path(file_path).stem)
        # Load raw data
        raw = mne.io.read_raw_fif(file_path, preload=True)
        result.append(SubjectRaw(subject_id=subject_id, raw_data=raw))
    
    return result

class AugmentedDataset(BaseEstimator, TransformerMixin):
    """This transformation creates an embedding version of the current dataset.

    The implementation and the application is described in [1]_.
    """
    def __init__(self, order=1, lag=1):
        self.order = order
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.order == 1:
            return X

        X_new = np.concatenate(
            [
                X[:, :, p * self.lag: -(self.order - p) * self.lag]
                for p in range(0, self.order)
            ],
            axis=1,
        )

        return X_new

def extract_epochs_from_raw(raw: mne.io.Raw):
    """Extract epochs for Rest and Animation events using sliding windows"""
    epochs_data = []
    labels = []
    data = raw.get_data()
    zero_event_names = []
    one_event_names = []
    
    sfreq = raw.info['sfreq']
    print(f'SFREQ: {sfreq}')
    sample_delay = int(1. * sfreq)  # Initial delay after event
    sample_window = int(1. * sfreq)   # Window size
    sample_step = int(0.8 * sfreq)   # Step size

    events, event_ids = mne.events_from_annotations(raw, event_id=STANDARD_EVENT_NAME_TO_ID_MAPPING, verbose=False)
    for k, v in event_ids.items():
        if v == 0:
            zero_event_names.append(k)
        elif v == 1:
            one_event_names.append(k)
    print(f'Zero event names: {sorted(list(set(zero_event_names)))}')
    print(f'One event names: {sorted(list(set(one_event_names)))}')
    # print(f'Event IDs: {event_ids}')
    for ind, (ts_idx, _, event_id) in enumerate(events[:-1]):  # Exclude last event
        if event_id not in [0, 1]:
            continue
        
        start_idx = ts_idx + sample_delay
        
        # search for the next event with different event_id
        next_event = next((i for i in events[ind + 1:] if i[2] != event_id), None)
        if next_event is None:
            end_idx = len(data)
        else:
            end_idx = next_event[0]
        
        
        # Calculate number of possible windows
        # n_windows = (end_idx - start_idx - sample_window) // sample_step + 1
        
        chunk_start = start_idx
        chunk_end = chunk_start + sample_window
        while chunk_end <= end_idx:
            chunk = data[:, chunk_start:chunk_end]
            epochs_data.append(chunk)
            labels.append(event_id)
            chunk_start = chunk_start + sample_step
            chunk_end = chunk_start + sample_window
    print(f'{np.unique([len(i[0]) for i in epochs_data])}')

    return np.array(epochs_data), np.array(labels)

def riemann_metric(x, y):
    """Custom metric for KNN using Riemannian distance between covariance matrices"""
    # Use constant number of channels from PICK_CHANNELS 
    # n = len(PICK_CHANNELS)
    # A = x.reshape(n, n)
    # B = y.reshape(n, n)
    # Since eigvalsh is from scipy.linalg and not supported by numba,
    # we need to use numpy's version instead
    return (np.log(eigvalsh(x, y))**2).sum(axis=-1)

from pyriemann.classification import FgMDM

def create_pipelines():
    """Create pipeline with KNN classifier using Riemannian distance"""
    steps = [
        ("knn", FgMDM(
            n_jobs=-1, tsupdate=True))
    ]
    
    return Pipeline(steps=steps)

def compute_distance_chunk(i, j, matrices):
    """Compute single distance between two matrices"""
    return i, j, riemann_metric(matrices[i], matrices[j])

def parallel_distance_matrix(matrices, n_jobs=-1):
    """Compute distance matrix in parallel using joblib"""
    n = len(matrices)
    # Generate upper triangle indices
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Compute distances in parallel with progress bar
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(compute_distance_chunk)(i, j, matrices) 
        for i, j in pairs
    )
    
    # Fill the distance matrix
    distances = np.zeros((n, n))
    for i, j, d in results:
        distances[i, j] = d
        distances[j, i] = d
    
    return distances

def main():
    # Load processed data
    raws_train = load_training_data()
    raws_val = load_validation_data()
    
    # TODO: Extract epochs from all training subjects
    X = []
    y = []
    subject_ids = []
    # subject_names = {}
    for raw in raws_train:
        X_train, y_train = extract_epochs_from_raw(raw['raw_data'])
        print(f'{X_train.shape=}')
        X.extend(X_train)
        y.extend(y_train)
        subject_ids.extend([raw['subject_id']] * len(y_train))
        # subject_names[raw['subject_id']] = raw
    print(f'Subject IDs: {len(subject_ids)}')
    
    # TODO: Extract epochs from all validation subjects
    X_val = []
    y_val = []
    subject_ids_val = []
    for raw in raws_val:
        X_val_train, y_val_train = extract_epochs_from_raw(raw['raw_data'])
        X_val.extend(X_val_train)
        y_val.extend(y_val_train)
        subject_ids_val.extend([raw['subject_id']] * len(y_val_train))

    X = np.array(X)
    y = np.array(y)
    subject_ids = np.array(subject_ids)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    subject_ids_val = np.array(subject_ids_val)
    
    # Separate background samples
    train_mask = y != 2
    
    X = X[train_mask]
    y = y[train_mask]
    subject_ids = subject_ids[train_mask]
    
    # Do the same for validation data
    val_train_mask = y_val != 2
    
    X_val = X_val[val_train_mask]
    y_val = y_val[val_train_mask]
    subject_ids_val = subject_ids_val[val_train_mask]
    
    # Compute covariance matrices
    print("\nComputing covariances...")
    cov = Covariances(estimator="oas")
    augmenter = AugmentedDataset(order=4, lag=8)
    X_covs = cov.transform(augmenter.transform(X))
    
    # Flatten the covariance matrices for KNN
    n_samples, n_channels, _ = X_covs.shape
    X_covs_flat = X_covs.reshape(n_samples, -1)
    
    # Create simplified pipeline with KNN
    pipeline = create_pipelines()
    
    # Use flattened covariance matrices directly as features
    features = X_covs_flat
    
    # Print data shapes and distributions
    print('Verification:')
    print(f'Number of samples in X: {len(features)}')
    print(f'Number of samples in y: {len(y)}')
    print(f'Number of groups: {len(np.unique(subject_ids))}')
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique groups: {np.unique(subject_ids)}")
    
    # Define scorers
    scorers = {
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'accuracy': make_scorer(accuracy_score),
        'auc': make_scorer(roc_auc_score)
    }
    
    print("\nEvaluating model configuration:")
    
    cv = GroupKFold(n_splits=len(np.unique(subject_ids)))
    # Calculate metrics for each fold
    metrics = {name: [] for name in scorers}
    
    # Dictionary to store cross-validation predictions
    cv_predictions = {
        'fold_indices': [],
        'sample_indices': [],
        'predicted_labels': [],
        'predicted_probas': [],
    }

    
    np.save('y.npy', y)
    print("\nComputing distance matrix...")
    # distances = parallel_distance_matrix(X_covs)
    # np.save('distances.npy', distances)
    # distances = np.load('distances.npy')
            
    # trans = umap.UMAP(n_neighbors=3, metric='precomputed', random_state=42, ).fit(distances)
    # plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=y, cmap='Spectral')
    # plt.title('Embedding of the training set by UMAP', fontsize=24)
    # plt.show()
    # exit()
    # from sklearn.manifold import Isomap
    # trans = Isomap(n_components=2, n_neighbors=15, metric=riemann_metric)
    # features_iso = trans.fit_transform(features)
    # plt.scatter(features_iso[:, 0], features_iso[:, 1], s= 5, c=y, cmap='Spectral')
    # plt.title('Embedding of the training set by Isomap', fontsize=24)
    # plt.show()
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_covs, y, subject_ids)):
        X_train, X_test = X_covs[train_idx], X_covs[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_pipeline = create_pipelines()
        fold_pipeline.fit(X_train, y_train)
        y_pred = fold_pipeline.predict(X_test)
        y_pred_proba = fold_pipeline.predict_proba(X_test)
        
        # Store predictions
        cv_predictions['fold_indices'].extend([fold_idx] * len(test_idx))
        cv_predictions['sample_indices'].extend(test_idx)
        cv_predictions['predicted_labels'].extend(y_pred)
        cv_predictions['predicted_probas'].extend(y_pred_proba)
        
        # Calculate metrics
        print(f"\nFold {fold_idx + 1} (subject {np.unique(subject_ids[test_idx])}):")
        for metric_name, scorer in scorers.items():
            if metric_name == 'auc':
                score = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                score = scorer._score_func(y_test, y_pred)
            metrics[metric_name].append(score)
            print(f"{metric_name}: {score:.3f}")
    
    # Print summary statistics
    print("\nCross-validation summary statistics:")
    for metric_name, scores in metrics.items():
        scores_array = np.array(scores)
        print(f"\n{metric_name.upper()}:")
        print(f"Mean: {np.mean(scores_array):.3f}")
        print(f"Std:  {np.std(scores_array):.3f}")
        print(f"Min:  {np.min(scores_array):.3f}")
        print(f"Max:  {np.max(scores_array):.3f}")
    
    # Train final model on all data
    pipeline.fit(features, y)
    
    # Save the trained pipeline
    print("\nSaving pipeline to trained_pipeline.pickle...")
    with open('trained_pipeline.pickle', 'wb') as f:
        pickle.dump({'pipeline': pipeline}, f)
    print("Pipeline saved successfully")

if __name__ == "__main__":
    main()

# Kernel: rbf
# Best parameters: OrderedDict([('augment__lag', 8), ('augment__order', 4), ('ada_svm__base_estimator__C', 0.6964903244678404), ('ada_svm__base_estimator__class_weight', None), ('ada_svm__base_estimator__gamma', 0.03482855648142016), ('ada_svm__learning_rate', 0.1), ('ada_svm__n_estimators', 100)])
# f_beta: 0.733 (+/- 0.212)
# precision: 0.745 (+/- 0.231)
# recall: 0.711 (+/- 0.254)
# accuracy: 0.725 (+/- 0.204)