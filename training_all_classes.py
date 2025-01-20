import warnings
import pandas as pd
import pyneurostim as ns
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score, accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, train_test_split
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
from config import PICK_CHANNELS, ND_CHANNELS_MASK, ORDER, STANDARD_EVENT_NAME_TO_ID_MAPPING, CSP_METRIC

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
        next_event = next((i for i in events[ind + 1:] if i[2] != event_id), None)
        if next_event is None:
            end_idx = len(data)
        else:
            end_idx = next_event[0]
        
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

def compute_csp_features(X, csp_transformer):
    """Compute CSP features for given samples using pre-fitted transformer"""
    return csp_transformer.transform(X)

def compute_background_medians(X_background, csp_transformer, subject_ids):
    """Compute median CSP features for background samples per subject"""
    features = csp_transformer.transform(X_background)
    unique_subjects = np.unique(subject_ids)
    
    background_medians = {}
    for subject in unique_subjects:
        subject_mask = subject_ids == subject
        if np.any(subject_mask):
            subject_features = features[subject_mask]
            background_medians[subject] = np.median(subject_features, axis=0)
    
    return background_medians

def normalize_features(features, background_medians, subject_ids):
    """Normalize features using background medians per subject"""
    normalized_features = np.zeros_like(features)
    
    for subject in background_medians:
        subject_mask = subject_ids == subject
        if np.any(subject_mask):
            # Percent normalization relative to background median
            # normalized_features[subject_mask] = (features[subject_mask] - background_medians[subject][None, :]) / background_medians[subject][None, :]
            normalized_features[subject_mask] = features[subject_mask]
            
    return normalized_features

def create_pipelines():
    """Create pipeline with AdaBoosted SVM"""
    base_svm = SVC(
        kernel='rbf',
        probability=True,
        C=0.696,
        gamma=0.035,
        class_weight='balanced'  # Add class weights to handle imbalance
    )
    
    steps = [
        ("svm", AdaBoostClassifier(
            estimator=base_svm,
            n_estimators=50,
            learning_rate=0.1,
            algorithm='SAMME.R'
        ))
    ]
    
    return Pipeline(steps=steps)

def main():
    # Load processed data
    raws_train = load_training_data()
    raws_val = load_validation_data()
    
    X = []
    y = []
    subject_ids = []
    for raw in raws_train:
        X_train, y_train = extract_epochs_from_raw(raw['raw_data'])
        X.extend(X_train)
        y.extend(y_train)
        subject_ids.extend([raw['subject_id']] * len(y_train))
    print(f'Subject IDs: {len(subject_ids)}')
    
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
    X = compute_normalized_augmented_covariances(
        X, 
        subject_ids,
        None, 
        order=ORDER
    )
    
    # Create and fit CSP transformer outside the pipeline
    csp = CSP(nfilter=10, 
              metric=CSP_METRIC,
              log=True)
    csp.fit(X, y)
    
    # Compute features and background medians
    X = csp.transform(X)
    
    # Create simplified pipeline (without CSP)
    pipeline = create_pipelines()
    
    # Print data shapes and distributions
    print('Verification:')
    print(f'Number of samples in X: {len(X)}')
    print(f'Number of samples in y: {len(y)}')
    print(f'Number of groups: {len(np.unique(subject_ids))}')
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique groups: {np.unique(subject_ids)}")
    
    # Define scorers
    scorers = {
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'accuracy': make_scorer(balanced_accuracy_score),  # Use balanced accuracy
        'auc': make_scorer(roc_auc_score, average='weighted', multi_class='ovr'),
        'f1': make_scorer(fbeta_score, beta=1, average='weighted')
    }
    
    print("\nEvaluating best model configuration:")
    
    cv = GroupKFold(n_splits=len(np.unique(subject_ids)))
    # Calculate metrics for each fold
    metrics = {name: [] for name in scorers}
    
    # Perform cross-validation and store predictions
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, subject_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        y_pred = fold_pipeline.predict(X_test)
        y_pred_proba = fold_pipeline.predict_proba(X_test)
        
        # Calculate metrics for monitoring
        print(f"\nFold {fold_idx + 1} (subject {np.unique(subject_ids[test_idx])}):")
        # Print class distribution for this fold
        unique, counts = np.unique(y_test, return_counts=True)
        print(f"Class distribution in test set: {dict(zip(unique, counts))}")
        
        for metric_name, scorer in scorers.items():
            if metric_name == 'auc':
                score = roc_auc_score(y_test, y_pred_proba[:, 1], average='weighted')
            elif metric_name == 'accuracy':
                score = scorer._score_func(y_test, y_pred)
            elif metric_name == 'f1':
                score = fbeta_score(y_test, y_pred, beta=1, average='weighted')
            else:
                score = scorer._score_func(y_test, y_pred, average='weighted')
            metrics[metric_name].append(score)
            print(f"{metric_name}: {score:.3f}")
    
    # Print summary statistics for all metrics
    print("\nCross-validation summary statistics:")
    for metric_name, scores in metrics.items():
        scores_array = np.array(scores)
        print(f"\n{metric_name.upper()}:")
        print(f"Mean: {np.mean(scores_array):.3f}")
        print(f"Std:  {np.std(scores_array):.3f}")
        print(f"Min:  {np.min(scores_array):.3f}")
        print(f"Max:  {np.max(scores_array):.3f}")
    
    # Save the trained pipeline and CSP transformer
    print("\nSaving pipeline and CSP transformer to trained_pipeline.pickle...")
    model_dict = {
        'pipeline': pipeline,
        'csp': csp,
        # 'background_medians': background_medians  # It might be useful to save this too
    }
    with open('trained_pipeline.pickle', 'wb') as f:
        pickle.dump(model_dict, f)
    print("Pipeline and CSP transformer saved successfully")

if __name__ == "__main__":
    main()

# Kernel: rbf
# Best parameters: OrderedDict([('augment__lag', 8), ('augment__order', 4), ('ada_svm__base_estimator__C', 0.6964903244678404), ('ada_svm__base_estimator__class_weight', None), ('ada_svm__base_estimator__gamma', 0.03482855648142016), ('ada_svm__learning_rate', 0.1), ('ada_svm__n_estimators', 100)])
# f_beta: 0.733 (+/- 0.212)
# precision: 0.745 (+/- 0.231)
# recall: 0.711 (+/- 0.254)
# accuracy: 0.725 (+/- 0.204)