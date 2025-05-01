
import warnings
import pandas as pd
from brainbuilding.transformers import AugmentedDataset
import pyneurostim as ns
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
from pyriemann.tangentspace import TangentSpace
from sklearn.decomposition import KernelPCA
import numba

from typing import TypedDict

# # Bayesian optimization
# from skopt import BayesSearchCV
# from skopt.space import Real, Integer, Categorical

import matplotlib.pyplot as plt
import pickle
from config import ORDER, STANDARD_EVENT_NAME_TO_ID_MAPPING, CSP_METRIC, LAG

import glob
import pathlib

warnings.filterwarnings('ignore') 

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

def create_pipeline():
    """Create simplified pipeline that works with pre-computed normalized features"""
    steps = [
        ("augmentation", AugmentedDataset(order=ORDER, lag=LAG)),
        ("csp", CSP(nfilter=10, metric=CSP_METRIC)),
        ("svm", SVC(kernel='rbf',
                   probability=True,
                   C=0.696,
                   gamma=0.035,
                   class_weight=None))
    ]
    
    return Pipeline(steps=steps)