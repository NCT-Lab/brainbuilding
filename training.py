import warnings
import pyneurostim as ns
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GroupKFold
import os
from scipy import interpolate
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from mne.preprocessing import ICA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.decomposition import KernelPCA

from typing import TypedDict

# # Bayesian optimization
# from skopt import BayesSearchCV
# from skopt.space import Real, Integer, Categorical

import matplotlib.pyplot as plt
import pickle
from eye_removal import EyeRemoval, create_standard_eog_channels
from config import PICK_CHANNELS, ND_CHANNELS_MASK, ORDER, STANDARD_EVENT_NAME_TO_ID_MAPPING

import glob
import pathlib

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

class CSPWithChannelSelection(CSP):
    """CSP that prioritizes filters with maximum response in specified channels"""
    def __init__(self, nfilter=4, metric='euclid', log=True, channels_mask=None, order=1):
        super().__init__(nfilter=nfilter, metric=metric, log=log)
        self.channels_mask = channels_mask
        self.order = order
        
    def fit(self, X, y):
        super().fit(X, y)
        
        # if self.channels_mask is not None:
        #     # Adjust channels mask for augmented data
        #     expanded_mask = np.tile(self.channels_mask, self.order)
            
        #     # Calculate response for each filter in the specified channels
        #     filter_responses = np.abs(self.filters_[:, expanded_mask]).sum(axis=1)
            
        #     # Get indices of filters with highest response in specified channels
        #     top_filter_indices = np.argsort(filter_responses)[-self.nfilter:]
            
        #     # Keep only the selected filters
        #     self.filters_ = self.filters_[top_filter_indices]
            
        return self

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
        end_idx = [i for i in events[ind + 1:] if i[2] != event_id][0][0]
        
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

def compute_normalized_augmented_covariances(X, groups, background_data, order=4):
    """Compute normalized augmented covariances for all samples using vectorized operations"""
    augmentation = AugmentedDataset(order=order, lag=8)
    cov = Covariances(estimator="oas")
    
    X_cov = cov.transform(augmentation.transform(X))
    return X_cov


def create_pipelines():
    """Create simplified pipeline that works with pre-computed covariances"""
    steps = [
        ("csp_channels", CSPWithChannelSelection(nfilter=10, 
                                               metric="euclid",
                                               channels_mask=ND_CHANNELS_MASK,
                                               order=ORDER)),
        ("svm", SVC(kernel='rbf',
                   probability=True,
                   C=0.696,
                   gamma=0.035,
                   class_weight=None))
    ]
    
    return Pipeline(steps=steps)

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
    
    # Compute normalized augmented covariances
    print("\nComputing normalized augmented covariances...")
    X_covs = compute_normalized_augmented_covariances(
        X, 
        subject_ids,
        None, 
        order=ORDER
    )
    print(f"Covariances shape: {X_covs.shape}")
    print(f"{np.any(np.isnan(X_covs))}")
    
    # Create pipeline for classification
    pipeline = create_pipelines()
    

    # include_mask = groups_all == 3
    X = X_covs[:]
    y = y[:]
    groups = subject_ids[:]
    
    # Print data shapes and distributions
    print('Verification:')
    print(f'Number of samples in X: {len(X)}')
    print(f'Number of samples in y: {len(y)}')
    print(f'Number of groups: {len(np.unique(groups))}')
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique groups: {np.unique(groups)}")
    
    # Create and evaluate best model
    pipeline = create_pipelines()
    
    # Define scorers
    scorers = {
        # 'f_beta': make_scorer(fbeta_score, beta=0.5),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'accuracy': make_scorer(accuracy_score)
    }
    
    print("\nEvaluating best model configuration:")
    
    cv = GroupKFold(n_splits=len(np.unique(groups)))
    # Calculate metrics for each fold
    metrics = {name: [] for name in scorers}
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        # Train and evaluate model for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone and fit pipeline
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        y_pred = fold_pipeline.predict(X_test)
        
        groups_test = groups[test_idx]
        # Calculate all metrics for this fold
        print(f"\nFold {fold_idx + 1} (subject {np.unique(groups_test)}):")
        for metric_name, scorer in scorers.items():
            score = scorer._score_func(y_test, y_pred)
            metrics[metric_name].append(score)
            print(f"{metric_name}: {score:.3f}")
    
    # Print summary statistics
    print("\nOverall metrics:")
    for metric_name in metrics:
        scores = metrics[metric_name]
        print(f"{metric_name}:")
        print(f"  mean: {np.mean(scores):.3f} (+/- {np.std(scores) * 2:.3f})")
        print(f"  min: {np.min(scores):.3f}")
        print(f"  max: {np.max(scores):.3f}")
    
    # Train final pipeline on all data
    print("\nTraining final pipeline on all data...")
    pipeline = create_pipelines()
    X_val_covs = compute_normalized_augmented_covariances(
        X_val,
        subject_ids_val,
        None,
        order=ORDER
    )
    
    pipeline.fit(X_val_covs, y_val)
    print("Pipeline training completed")
    
    # After training the final pipeline, evaluate on validation data
    print("\nEvaluating model on validation data...")
    
    # Make predictions on validation data
    y_val_pred = pipeline.predict(X_val_covs)
    
    # Calculate and print validation metrics
    print("\nValidation metrics:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.3f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.3f}")
    print(f"Recall: {recall_score(y_val, y_val_pred):.3f}")
    print(f"F-beta score: {fbeta_score(y_val, y_val_pred, beta=0.5):.3f}")
    
    # Save the trained pipeline
    print("\nSaving pipeline to trained_pipeline.pickle...")
    with open('trained_pipeline.pickle', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Pipeline saved successfully")

if __name__ == "__main__":
    main()

# Kernel: rbf
# Best parameters: OrderedDict([('augment__lag', 8), ('augment__order', 4), ('ada_svm__base_estimator__C', 0.6964903244678404), ('ada_svm__base_estimator__class_weight', None), ('ada_svm__base_estimator__gamma', 0.03482855648142016), ('ada_svm__learning_rate', 0.1), ('ada_svm__n_estimators', 100)])
# f_beta: 0.733 (+/- 0.212)
# precision: 0.745 (+/- 0.231)
# recall: 0.711 (+/- 0.254)
# accuracy: 0.725 (+/- 0.204)