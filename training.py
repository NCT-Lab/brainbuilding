import warnings
import pandas as pd
import pyneurostim as ns
import numpy as np
import mne
from scipy.signal import butter, filtfilt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score, accuracy_score, roc_auc_score
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
from config import ORDER, STANDARD_EVENT_NAME_TO_ID_MAPPING, CSP_METRIC

import glob
import pathlib

warnings.filterwarnings('ignore') 

class SubjectRaw(TypedDict):
    subject_id: int
    raw_data: mne.io.Raw

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

def compute_normalized_augmented_covariances(X, order=4):
    """Compute normalized augmented covariances for all samples using vectorized operations"""
    augmentation = AugmentedDataset(order=order, lag=8)
    cov = Covariances(estimator="oas")
    
    X_cov = cov.transform(augmentation.transform(X))
    return X_cov

def compute_csp_features(X, csp_transformer):
    """Compute CSP features for given samples using pre-fitted transformer"""
    return csp_transformer.transform(X)

def create_pipelines():
    """Create simplified pipeline that works with pre-computed normalized features"""
    steps = [
        ("csp", CSP(nfilter=10, metric='riemann')),
        ("svm", SVC(kernel='rbf',
                   probability=True,
                   C=0.696,
                   gamma=0.035,
                   class_weight=None))
    ]
    
    return Pipeline(steps=steps)

def main():
    # Load processed data
    X = np.load('X.npy')
    y = np.load('y.npy')
    subject_ids = np.load('subject_ids.npy')
    X = X[y != 2]
    subject_ids = subject_ids[y != 2]
    y = y[y != 2]
    print(f"{np.unique(y)=}")
    
    
    # Compute normalized augmented covariances
    print("\nComputing normalized augmented covariances...")
    X = compute_normalized_augmented_covariances(
        X, 
        order=ORDER
    )
    
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
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'accuracy': make_scorer(accuracy_score),
        'auc': make_scorer(roc_auc_score)
    }
    
    print("\nEvaluating best model configuration:")
    
    cv = GroupKFold(n_splits=len(np.unique(subject_ids)))
    # Calculate metrics for each fold
    metrics = {name: [] for name in scorers}
    
    # Dictionary to store cross-validation predictions
    cv_predictions = {
        'fold_indices': [],
        'sample_indices': [],  # To match predictions with original samples
        'predicted_labels': [],
        'predicted_probas': [],
    }
    
    # Perform cross-validation and store predictions
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, subject_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_pipeline = create_pipelines()
        fold_pipeline.fit(X_train, y_train)
        y_pred = fold_pipeline.predict(X_test)
        y_pred_proba = fold_pipeline.predict_proba(X_test)
        
        # Store predictions with sample indices
        cv_predictions['fold_indices'].extend([fold_idx] * len(test_idx))
        cv_predictions['sample_indices'].extend(test_idx)
        cv_predictions['predicted_labels'].extend(y_pred)
        cv_predictions['predicted_probas'].extend(y_pred_proba)
        
        # Calculate metrics for monitoring
        print(f"\nFold {fold_idx + 1} (subject {np.unique(subject_ids[test_idx])}):")
        for metric_name, scorer in scorers.items():
            if metric_name == 'auc':
                # For AUC we need probabilities of the positive class
                score = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                score = scorer._score_func(y_test, y_pred)
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
    
    exit() 
    # Train final model on all data and get predictions
    pipeline.fit(normalized_features, y)
    full_predictions = pipeline.predict(normalized_features)
    full_probas = pipeline.predict_proba(normalized_features)
    
    # Add cross-validation predictions to both DataFrames
    for df in [log_features_df, full_features_df]:
        # Initialize prediction columns with NaN
        df['cv_predicted_label'] = np.nan
        df['cv_proba_class_0'] = np.nan
        df['cv_proba_class_1'] = np.nan
        df['cv_fold'] = np.nan
        
        # Fill in CV predictions
        for idx, sample_idx in enumerate(cv_predictions['sample_indices']):
            df.loc[sample_idx, 'cv_predicted_label'] = cv_predictions['predicted_labels'][idx]
            df.loc[sample_idx, 'cv_proba_class_0'] = cv_predictions['predicted_probas'][idx][0]
            df.loc[sample_idx, 'cv_proba_class_1'] = cv_predictions['predicted_probas'][idx][1]
            df.loc[sample_idx, 'cv_fold'] = cv_predictions['fold_indices'][idx]
        
        # Add full dataset predictions
        df['full_predicted_label'] = full_predictions
        df['full_proba_class_0'] = [p[0] for p in full_probas]
        df['full_proba_class_1'] = [p[1] for p in full_probas]
    
    # Save both feature tables
    log_features_df.to_csv('features_log_with_predictions.csv', index=False)
    full_features_df.to_csv('features_full_with_predictions.csv', index=False)
    
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