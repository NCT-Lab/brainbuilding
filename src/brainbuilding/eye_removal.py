import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA
from scipy import stats

from .config import PICK_CHANNELS

class EyeRemoval(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, veog_data, heog_data, remove_veog=True, remove_heog=True, random_state=42):
        self.ica = FastICA(n_components=n_components, random_state=random_state)
        self.veog_data = veog_data
        self.heog_data = heog_data
        self.remove_veog = remove_veog
        self.remove_heog = remove_heog
        
    def fit(self, X, y=None):
        X_transformed = self.ica.fit_transform(X.T).T
        veog_correlations = np.array([stats.pearsonr(source, self.veog_data)[0] for source in X_transformed])
        heog_correlations = np.array([stats.pearsonr(source, self.heog_data)[0] for source in X_transformed])
        self.veog_idx = np.argmax(np.abs(veog_correlations))
        self.heog_idx = np.argmax(np.abs(heog_correlations))
        if self.veog_idx == self.heog_idx:
            indices = np.arange(len(veog_correlations))
            mask = indices != self.veog_idx
            self.heog_idx = np.argmax(np.abs(heog_correlations[mask]))
        return self
    
    def transform(self, X):
        X_transformed = self.ica.transform(X.T)
        if self.remove_veog:
            X_transformed[:, self.veog_idx] = 0
        if self.remove_heog:
            X_transformed[:, self.heog_idx] = 0
        X_transformed = self.ica.inverse_transform(X_transformed).T
        return X_transformed
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_sources(self, X):
        """Get ICA sources for benchmarking"""
        return self.ica.transform(X.T).T
    
    def score_sources(self, X, target, start=None, stop=None):
        """Score ICA components against a target channel"""

        sources = self.get_sources(X)
        if isinstance(target, (int, np.integer)):
            target = X[target, start:stop]
        else:
            target = target[start:stop]
            
        correlations = np.array([stats.pearsonr(source, target)[0] for source in sources])
        return correlations

def create_standard_eog_channels(data):
    """Create vEOG and hEOG channels from EEG data"""
    assert data.shape[0] == len(PICK_CHANNELS)
    # Get indices for EOG computation
    fp1 = data[PICK_CHANNELS == 'Fp1'][0]
    fp2 = data[PICK_CHANNELS == 'Fp2'][0]
    f7 = data[PICK_CHANNELS == 'F7'][0]
    f8 = data[PICK_CHANNELS == 'F8'][0]
    
    # Compute EOG channels
    veog = (fp2 + fp1) / 2
    heog = f8 - f7
    
    return veog, heog