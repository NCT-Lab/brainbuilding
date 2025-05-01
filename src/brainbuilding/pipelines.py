from typing import List, Dict, Any
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from pyriemann.classification import KNearestNeighbor
from pyriemann.spatialfilters import CSP
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier

from .transformers import TangentSpaceProjector, ParallelTransportTransformer, ColumnSelector, StructuredCSP, AugmentedDataset, Covariances, SubjectWhiteningTransformer, StructuredColumnTransformer, BackgroundFilterTransformer, ReferenceKNN, reference_weighted_euclidean_distance

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from typing import List, Tuple, Any, Optional
import numpy as np

N_NEIGHBORS = 25
N_FILTERS = 10

# Parallel Transport -> Tangent Space -> KNN
# PT_TANGENT_KNN_STEPS = [
#     ('pt', ParallelTransportTransformer(include_means=True)),
#     ('project-tangent', TangentSpaceProjector()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('knn', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance'))
# ]

# # Parallel Transport -> Tangent Space -> KernelPCA -> KNN
# PT_TANGENT_KPCA_KNN_STEPS = [
#     ('pt', ParallelTransportTransformer(include_means=True)),
#     ('project-tangent', TangentSpaceProjector()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('kpca', KernelPCA(n_components=N_FILTERS, kernel='rbf', gamma=0.035)),
#     ('knn', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance'))
# ]

# # Parallel Transport -> CSP (no log) -> Riemannian KNN
# PT_CSP_KNN_STEPS = [
#     ('pt', ParallelTransportTransformer(include_means=True)),
#     ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=False, metric='riemann')),
#     ('tangent-space', TangentSpaceProjector()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('knn', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance'))
# ]

# # Parallel Transport -> CSP (with log) -> SVC
# PT_CSP_SVC_STEPS = [
#     ('pt', ParallelTransportTransformer()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('csp', CSP(nfilter=N_FILTERS, log=True, metric='riemann')),
#     ('svc', SVC(kernel='rbf', C=0.696, gamma=0.035, class_weight=None, probability=True))
# ]

# # CSP -> Parallel Transport -> Tangent Space -> SVC
# CSP_PT_TANGENT_SVC_STEPS = [
#     ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=False, metric='riemann')),
#     ('pt', ParallelTransportTransformer(include_means=True)),
#     ('tangent-space', TangentSpaceProjector()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('svc', SVC(kernel='rbf', C=0.696, gamma=0.035, class_weight=None, probability=True))
# ]

# # CSP -> Parallel Transport -> Tangent Space -> KNN
# CSP_PT_TANGENT_KNN_STEPS = [
#     ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=False, metric='riemann')),
#     ('pt', ParallelTransportTransformer(include_means=True)),
#     ('tangent-space', TangentSpaceProjector()),
#     ('selector', ColumnSelector(fields='sample')),
#     ('knn', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance'))
# ]

# # CSP -> SVC
# CSP_SVC_STEPS = [
#     ('selector', ColumnSelector(fields='sample')),
#     ('csp', CSP(nfilter=N_FILTERS, log=True, metric='riemann')),
#     ('svc', SVC(kernel='rbf', C=0.696, gamma=0.035, class_weight=None, probability=True))
# ]


SIMPLE_CSP = [
    ('selector', ColumnSelector(fields='sample')),
    ('csp', CSP(nfilter=N_FILTERS, log=True, metric='riemann')),
]

PT_CSP_TANGENT = [
    ('pt', ParallelTransportTransformer(include_means=True)),
    ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=False, metric='riemann')),
    ('tangent-space', TangentSpaceProjector()),
    ('selector', ColumnSelector(fields='sample')),
]

PT_CSP_LOG = [
    ('pt', ParallelTransportTransformer()),
    ('selector', ColumnSelector(fields='sample')),
    ('csp', CSP(nfilter=N_FILTERS, log=True, metric='riemann')),
]

CSP_PT_TANGENT = [
    ('csp', StructuredCSP(field='sample', nfilter=N_FILTERS, log=False, metric='riemann')),
    ('pt', ParallelTransportTransformer(include_means=True)),
    ('tangent-space', TangentSpaceProjector()),
    ('selector', ColumnSelector(fields='sample')),
]

KPCA_REDUCTION = [
    ('kpca', KernelPCA(n_components=N_FILTERS, kernel='rbf', gamma=0.035)),
]

SVC_CLASSIFICATION = [
    ('svc', SVC(kernel='rbf', C=0.696, gamma=0.035, class_weight=None, probability=True))
]

KNN_NO_DISTANCE_CLASSIFICATION = [
    ('knn-no-distance', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='uniform'))
]

KNN_DISTANCE_CLASSIFICATION = [
    ('knn-distance', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance'))
]


OAS_COV = [
    ('covariances', StructuredColumnTransformer(column='sample', transformer=Covariances(estimator='oas'))),
]

AUGMENTED_OAS_COV = [
    ('augmented_dataset', StructuredColumnTransformer(column='sample', transformer=AugmentedDataset(order=4, lag=8))),
    ('covariances', StructuredColumnTransformer(column='sample', transformer=Covariances(estimator='oas'))),
]

WHITENING = [
    ('whitening', SubjectWhiteningTransformer()),
]

BACKGROUND_FILTER = [
    ('background_filter', BackgroundFilterTransformer()),
]

GENERAL_PIPELINE_STEPS = [
    [
        WHITENING, []
    ],
    [
        BACKGROUND_FILTER
    ],
    [
        AUGMENTED_OAS_COV, OAS_COV
    ],
    [
        PT_CSP_TANGENT, SIMPLE_CSP, PT_CSP_LOG, CSP_PT_TANGENT
    ],
    [
        KPCA_REDUCTION, []
    ],
    [
        SVC_CLASSIFICATION, KNN_NO_DISTANCE_CLASSIFICATION, KNN_DISTANCE_CLASSIFICATION
    ]
]


class OnlinePipeline(Pipeline):
    """
    A pipeline that supports partial_fit for online learning.
    
    This class extends scikit-learn's Pipeline to support partial_fit,
    which is useful for online learning scenarios. It will call partial_fit
    on all steps that implement it, while maintaining the regular fit behavior
    for steps that don't support partial_fit.
    
    Parameters
    ----------
    steps : list of (str, estimator) tuples
        List of (name, estimator) tuples that are chained in sequence.
    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline.
    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.
    """
    
    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> 'OnlinePipeline':
        """
        Fit the pipeline on a batch of data.
        
        This method will call partial_fit on all steps that implement it,
        while maintaining the regular fit behavior for steps that don't support
        partial_fit.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.
        **fit_params : dict of string -> object
            Parameters passed to the partial_fit method of each step.
            
        Returns
        -------
        self : OnlinePipeline
            This estimator.
        """
        Xt = X
        for name, step in self.steps:
            if hasattr(step, 'partial_fit'):
                # If step supports partial_fit, use it
                step.partial_fit(Xt, y, **fit_params)

            # Transform the data for the next step
            if hasattr(step, 'transform'):
                Xt = step.transform(Xt)
                
        return self