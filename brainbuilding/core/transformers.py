from typing import Dict, Optional
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from pyriemann.utils.mean import mean_riemann  # type: ignore
from scipy.linalg import sqrtm  # type: ignore
from pyriemann.utils.tangentspace import tangent_space  # type: ignore
from pyriemann.estimation import Covariances  # type: ignore
from pyriemann.utils.base import invsqrtm  # type: ignore
from pyriemann.spatialfilters import CSP

from brainbuilding.core.config import PICK_CHANNELS
from sklearn.decomposition import FastICA

from brainbuilding.core.utils import happend  # type: ignore
from ..train.context import _IN_PREDICTION_PHASE
from pyriemann.utils.tangentspace import upper, unupper  # type: ignore
from functools import partial
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore


# @njit(nopython=True)
def reference_weighted_euclidean_distance(X, Y, reference):
    diff = X - Y
    a = upper(reference @ unupper(diff))
    return np.sqrt(np.sum(a * a))


class ParallelTransportTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that applies parallel transport to covariance matrices.
    Supports online updates of mean estimates.

    Parameters
    ----------
    mock_fit : bool, default=False
        If True, the transformer will not fit any data and just pass it through
    include_means : bool, default=False
        If True, the transformer will include the subject mean and general mean
        matrices in the output structured array

    Attributes
    ----------
    subject_means_ : dict
        Dictionary mapping subject IDs to their mean covariance matrices
    general_mean_ : ndarray
        General mean covariance matrix across all subjects
    subject_counts_ : dict
        Dictionary mapping subject IDs to the number of samples used for their mean estimation
    total_count_ : int
        Total number of samples used for general mean estimation
    """

    def __init__(
        self,
        mock_fit: bool = False,
        include_means: bool = False,
        prevent_subject_drift: bool = True,
        subject_min_samples_for_transform: int = 1,
    ):
        self.subject_means_: Dict[int, np.ndarray] = {}
        self.general_mean_: Optional[np.ndarray] = None
        self.subject_counts_: Dict[int, int] = {}
        self.total_count_: int = 0
        self.mock_fit = mock_fit
        self.include_means = include_means
        self.prevent_subject_drift: bool = prevent_subject_drift
        self.subject_min_samples_for_transform: int = (
            subject_min_samples_for_transform
        )
        self.max_samples_for_subject: int = 0

    def get_subject_means(self) -> Dict[int, np.ndarray]:
        """
        Get the mean covariance matrices for all subjects.

        Returns
        -------
        dict
            Dictionary mapping subject IDs to their mean covariance matrices
        """
        return self.subject_means_

    def get_subject_counts(self) -> Dict[int, int]:
        """
        Get the sample counts for all subjects.

        Returns
        -------
        dict
            Dictionary mapping subject IDs to their sample counts
        """
        return self.subject_counts_

    def set_subject_means_and_counts(
        self,
        subject_means: Dict[int, np.ndarray],
        subject_counts: Dict[int, int],
    ) -> None:
        """
        Set subject means and counts, and update the general mean and total count.

        Parameters
        ----------
        subject_means : dict
            Dictionary mapping subject IDs to their mean covariance matrices
        subject_counts : dict
            Dictionary mapping subject IDs to their sample counts

        Raises
        ------
        ValueError
            If subject_means and subject_counts have different keys
        """
        if self.mock_fit:
            return

        if set(subject_means.keys()) != set(subject_counts.keys()):
            raise ValueError(
                "subject_means and subject_counts must have the same keys"
            )

        self.subject_means_ = subject_means
        self.subject_counts_ = subject_counts
        self.total_count_ = sum(subject_counts.values())

        # Update general mean using weighted Riemannian mean
        if self.total_count_ > 0:
            self.general_mean_ = mean_riemann(
                np.array(list(self.subject_means_.values())),
                sample_weight=np.array(
                    list(self.subject_counts_.values()), dtype=np.float64
                ),
            )

    def _extract_covariances(self, X: np.ndarray) -> np.ndarray:
        """Extract covariance matrices from structured array"""
        return X["sample"]

    # TODO: we should use configurable field name
    def _extract_subject_ids(self, X: np.ndarray) -> np.ndarray:
        """Extract per-session IDs from structured array.

        Prefer 'session_id' if present; fall back to 'subject_id' for backward
        compatibility.
        """
        if "session_id" in X.dtype.names:
            return X["session_id"]
        return X["subject_id"]

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "ParallelTransportTransformer":
        """
        Fit the transformer by computing subject means and general mean.

        Parameters
        ----------
        X : structured array
            Input data with 'covariance' and 'subject_id' fields
        y : None
            Not used, present for scikit-learn compatibility

        Returns
        -------
        self : ParallelTransportTransformer
            The fitted transformer
        """
        if self.mock_fit:
            return self

        self.subject_means_ = {}
        self.general_mean_ = None
        self.subject_counts_ = {}
        self.total_count_ = 0

        # Extract data from structured array
        covariances = self._extract_covariances(X)
        subject_ids = self._extract_subject_ids(X)

        # Compute means for each subject
        unique_subjects = np.unique(subject_ids)
        for subject in unique_subjects:
            if subject in self.subject_means_:
                # Skip subjects that have already been seen (use partial_fit for online updates)
                continue

            mask = subject_ids == subject
            subject_covs = covariances[mask]
            self.subject_means_[subject] = mean_riemann(subject_covs)
            self.subject_counts_[subject] = len(subject_covs)

        # Compute general mean
        self.general_mean_ = mean_riemann(
            np.array(list(self.subject_means_.values())),
            sample_weight=np.array(
                list(self.subject_counts_.values()), dtype=np.float64
            ),
        )
        self.general_mean_inv_ = np.linalg.inv(self.general_mean_)
        self.total_count_ = len(covariances)
        self.max_samples_for_subject = int(
            np.mean(list(self.subject_counts_.values()))
        )

        return self

    def partial_fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        custom_weights: Optional[np.ndarray] = None,
    ) -> "ParallelTransportTransformer":
        """
        Update mean estimates with new data.

        Parameters
        ----------
        X : structured array
            New data with 'covariance' and 'subject_id' fields
        y : None
            Not used, present for scikit-learn compatibility

        Returns
        -------
        self : ParallelTransportTransformer
            The updated transformer
        """
        if self.mock_fit:
            return self

        # Extract data from structured array
        covariances = self._extract_covariances(X)
        subject_ids = self._extract_subject_ids(X)

        # If this is the first call, initialize with fit
        if self.general_mean_ is None:
            return self.fit(X, y)

        # Update subjects with capping to avoid exceeding max_samples_for_subject
        unique_subjects = np.unique(subject_ids)
        accepted_covariances_list: list[np.ndarray] = []
        accepted_weights_list: Optional[list[float]] = (
            [] if custom_weights is not None else None
        )
        accepted_increment = 0
        for subject in unique_subjects:
            mask = subject_ids == subject
            subject_covs = covariances[mask]

            # global mean drift prevention logic
            current_count = self.subject_counts_.get(subject, 0)
            if self.prevent_subject_drift:
                allow_n = max(0, self.max_samples_for_subject - current_count)
                allow_n = min(allow_n, len(subject_covs))
            else:
                allow_n = len(subject_covs)
            if allow_n == 0:
                continue

            selected_covs = subject_covs[:allow_n]
            accepted_covariances_list.extend(list(selected_covs))
            if custom_weights is None:
                increment = allow_n
                subj_weights = [1] * allow_n
            else:
                subj_weights_arr = np.asarray(custom_weights)[mask][:allow_n]
                increment = int(np.sum(subj_weights_arr))
                subj_weights = list(subj_weights_arr)
                if accepted_weights_list is not None:
                    accepted_weights_list.extend(subj_weights)

            if subject in self.subject_means_:
                old_mean = self.subject_means_[subject]
                old_count = self.subject_counts_[subject]
                weights = [old_count] + subj_weights
                self.subject_means_[subject] = mean_riemann(
                    np.array([old_mean] + list(selected_covs)),
                    sample_weight=np.array(weights, dtype=np.float64),
                )
                self.subject_counts_[subject] = old_count + increment
            else:
                self.subject_means_[subject] = mean_riemann(selected_covs)
                self.subject_counts_[subject] = increment
            accepted_increment += increment

        # If nothing accepted, return unchanged
        if len(accepted_covariances_list) == 0:
            return self

        # Update general mean using accepted samples only
        if custom_weights is None:
            general_weights = [float(self.total_count_)] + [1.0] * len(
                accepted_covariances_list
            )
        else:
            assert accepted_weights_list is not None
            general_weights = [float(self.total_count_)] + [
                float(w) for w in accepted_weights_list
            ]
        self.general_mean_ = mean_riemann(
            np.array([self.general_mean_] + accepted_covariances_list),
            sample_weight=np.array(general_weights, dtype=np.float64),
        )
        assert self.general_mean_ is not None
        self.general_mean_inv_ = np.linalg.inv(self.general_mean_)
        self.total_count_ += accepted_increment

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform covariance matrices using parallel transport.

        Parameters
        ----------
        X : structured array
            Input data with 'covariance' and 'subject_id' fields

        Returns
        -------
        ndarray
            Transformed covariance matrices
        """
        if self.mock_fit:
            return self._extract_covariances(X)

        covariances = self._extract_covariances(X)
        subject_ids = self._extract_subject_ids(X)

        X_transformed = np.zeros_like(covariances)

        if self.general_mean_ is None:
            raise ValueError(
                "Transformer not fitted: general mean is not available"
            )
        general_means_inv = (
            np.zeros_like(covariances) if self.include_means else None
        )
        general_means = (
            np.zeros_like(covariances) if self.include_means else None
        )

        # Apply parallel transport to each sample
        for i, (cov, subject) in enumerate(zip(covariances, subject_ids)):
            if subject not in self.subject_means_:
                raise ValueError(f"Subject {subject} not seen during fitting")

            # Get subject mean
            M = self.subject_means_[subject]

            if self.include_means:
                assert (
                    general_means_inv is not None and general_means is not None
                )
                general_means_inv[i] = self.general_mean_inv_
                general_means[i] = self.general_mean_

            if (
                self.subject_counts_.get(subject, 0)
                < self.subject_min_samples_for_transform
            ):
                raise ValueError(
                    "Transformer not tuned for this subject: not enough samples"
                )
            else:
                # Compute transformation matrix E = (GM^-1)^1/2
                GM_inv = np.dot(self.general_mean_, np.linalg.inv(M))
                E = sqrtm(GM_inv)

                # Apply transformation: ESE^T
                X_transformed[i] = np.dot(np.dot(E, cov), E.T)

        # Create structured array with transformed data
        X_transformed_structured = happend(X, X_transformed, "sample")

        # Add subject and general means if requested
        if self.include_means:
            X_transformed_structured = happend(
                X_transformed_structured, general_means_inv, "general_mean_inv"
            )
            X_transformed_structured = happend(
                X_transformed_structured, general_means, "general_mean"
            )

        return X_transformed_structured

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit the transformer and transform the data.

        Parameters
        ----------
        X : structured array
            Input data with 'covariance' and 'subject_id' fields
        y : None
            Not used, present for scikit-learn compatibility

        Returns
        -------
        ndarray
            Transformed covariance matrices
        """
        return self.fit(X, y).transform(X)


class SimpleWhiteningTransformer(BaseEstimator, TransformerMixin):
    """
    Whitening that operates on plain arrays without any assumptions about labels.

    Expects X with shape (n_samples, n_channels, n_times). Computes a single
    whitening matrix from all provided samples and applies it per sample.
    """

    def __init__(self, n_channels: int):
        self.W_: Optional[np.ndarray] = None
        self.n_channels = n_channels

    def fit(self, X, y=None):
        if X.ndim == 3:
            n_samples, n_channels, _ = X.shape
            assert n_channels == self.n_channels
            background_concat = np.concatenate(
                [X[i] for i in range(n_samples)], axis=1
            )
        elif X.ndim == 2:
            n_channels, _ = X.shape
            assert n_channels == self.n_channels
            background_concat = X
        else:
            raise ValueError(
                "SimpleWhiteningTransformer expects X of shape (n_samples, n_channels, n_times) or (n_channels, n_times)"
            )

        cov = Covariances(estimator="oas").transform(
            background_concat[np.newaxis, :, :]
        )[0]
        self.W_ = invsqrtm(cov)
        return self

    def transform(self, X):
        if self.W_ is None:
            raise ValueError("Transformer not fitted")
        Xw = np.zeros_like(X)
        if X.ndim == 3:
            n_samples, n_channels, _ = X.shape
            assert n_channels == self.n_channels
            for i in range(n_samples):
                Xw[i] = self.W_ @ X[i]
        elif X.ndim == 2:
            n_channels, _ = X.shape
            assert n_channels == self.n_channels
            Xw = self.W_ @ X
        return Xw


class StructuredArrayBuilder(BaseEstimator, TransformerMixin):
    """
    Utility to convert plain arrays into structured arrays required by
    structured transformers (e.g., ParallelTransportTransformer).

    Parameters
    ----------
    subject_id : int, default=0
        Subject identifier to assign to all samples.
    sample_field : str, default='sample'
        Field name to store the per-sample matrices.
    """

    def __init__(self, subject_id: int = 0, sample_field: str = "sample"):
        self.subject_id = int(subject_id)
        self.sample_field = sample_field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim != 3:
            raise ValueError(
                "StructuredArrayBuilder expects X of shape (n_samples, n_channels, n_times)"
            )
        n = X.shape[0]
        dtype = [
            (self.sample_field, X.dtype, X.shape[1:]),
            ("subject_id", np.int64),
        ]
        out = np.empty(n, dtype=dtype)
        out[self.sample_field] = X
        out["subject_id"] = self.subject_id
        return out


class StructuredToArray(BaseEstimator, TransformerMixin):
    """
    Extract a field from a structured array into a plain ndarray for
    downstream transformers that don't support structured arrays.
    """

    def __init__(self, field: str = "sample"):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not hasattr(X, "dtype") or X.dtype.names is None:
            raise ValueError(
                "StructuredToArray expects a structured array input"
            )
        if self.field not in X.dtype.names:
            raise ValueError(f"Field '{self.field}' not found in input")
        return X[self.field]


class TangentSpaceProjector(BaseEstimator, TransformerMixin):
    """
    A transformer that projects matrices to tangent space using the general mean matrices.

    Parameters
    ----------
    sample_col : str, default='sample'
        Name of the column in structured array containing the sample matrices
    general_mean_col : str, default='general_mean'
        Name of the column in structured array containing the general mean matrices

    Attributes
    ----------
    sample_col_ : str
        Name of the column containing sample matrices
    general_mean_col_ : str
        Name of the column containing general mean matrices
    """

    def __init__(self, sample_col="sample", general_mean_col="general_mean"):
        self.sample_col = sample_col
        self.general_mean_col = general_mean_col

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X : structured array
            Input data with sample matrices and general mean matrices
        y : None
            Not used, present for scikit-learn compatibility

        Returns
        -------
        self : TangentSpaceParallelTransportTransformer
            The fitted transformer
        """
        return self

    def transform(self, X):
        """
        Transform matrices by projecting to tangent space around general mean.

        Parameters
        ----------
        X : structured array
            Input data with sample matrices and general mean matrices

        Returns
        -------
        ndarray
            Structured array with tangent space vectors
        """
        # Check if general_mean_col exists in X
        if self.general_mean_col not in X.dtype.names:
            raise ValueError(
                f"Column '{self.general_mean_col}' not found in input data. "
                f"Available columns: {X.dtype.names}"
            )

        # Extract samples and general means
        samples = X[self.sample_col]
        general_means = X[self.general_mean_col]

        # Project each sample to tangent space around its corresponding general mean
        tangent_vectors = np.array(
            [
                tangent_space(sample[np.newaxis, :, :], general_mean)[0]
                for sample, general_mean in zip(samples, general_means)
            ]
        )

        # Create structured array with tangent vectors
        X_tangent_structured = happend(X, tangent_vectors, self.sample_col)

        return X_tangent_structured

    def fit_transform(self, X, y=None):
        """
        Fit the transformer and transform the data.

        Parameters
        ----------
        X : structured array
            Input data with sample matrices and general mean matrices
        y : None
            Not used, present for scikit-learn compatibility

        Returns
        -------
        ndarray
            Structured array with tangent space vectors
        """
        return self.fit(X, y).transform(X)


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
                X[:, :, p * self.lag : -(self.order - p) * self.lag]
                for p in range(0, self.order)
            ],
            axis=1,
        )

        return X_new


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that selects specific columns from structured arrays.

    Parameters
    ----------
    fields : list of str
        List of field names to select from the structured array

    Attributes
    ----------
    fields_ : list of str
        The field names to select
    """

    def __init__(self, fields):
        self.fields = fields
        self.fields_ = fields

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.fields]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class StructuredCSP(CSP):
    def __init__(
        self, field="sample", nfilter=10, log=True, metric="riemann", **kwargs
    ):
        super().__init__(nfilter=nfilter, log=log, metric=metric, **kwargs)
        self.field = field

    def fit(self, X, y=None):
        super().fit(X[self.field], y)
        return self

    def transform(self, X):
        transformed_data = super().transform(X[self.field])
        X_new = happend(X, transformed_data, self.field)
        if "general_mean" in X.dtype.names:
            X_new = happend(
                X_new, super().transform(X["general_mean"]), "general_mean"
            )
        return X_new


class SubjectWhiteningTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that applies subject-specific whitening to signals.
    Computes whitening matrices using background samples for each subject.

    Parameters
    ----------
    None

    Attributes
    ----------
    subject_whitening_matrices_ : dict
        Dictionary mapping subject IDs to their whitening matrices
    """

    def __init__(self):
        self.subject_whitening_matrices_: Dict[int, np.ndarray] = {}

    def _extract_signals(self, X: np.ndarray) -> np.ndarray:
        """Extract signals from structured array"""
        return X["sample"]

    def _extract_subject_ids(self, X: np.ndarray) -> np.ndarray:
        """Extract subject IDs from structured array"""
        return X["subject_id"]

    def _extract_is_background(self, X: np.ndarray) -> np.ndarray:
        """Extract background flags from structured array"""
        return X["is_background"]

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "SubjectWhiteningTransformer":
        """
        Fit the transformer by computing whitening matrices for each subject using their background samples.

        Parameters
        ----------
        X : structured array
            Input data with fields:
            - sample: (n_channels, n_times) array
            - subject_id: int
            - is_background: int (0 or 1)
        y : None
            Not used, present for scikit-learn compatibility

        Returns
        -------
        self : SubjectWhiteningTransformer
            The fitted transformer
        """
        # Extract data from structured array
        signals = self._extract_signals(X)
        subject_ids = self._extract_subject_ids(X)
        is_background = self._extract_is_background(X)

        # Compute whitening matrices for each subject
        unique_subjects = np.unique(subject_ids)
        for subject in unique_subjects:
            # Get background samples for this subject
            mask = (subject_ids == subject) & (is_background == 1)
            if not np.any(mask):
                raise ValueError(
                    f"No background samples found for subject {subject}"
                )

            subject_background = signals[mask]

            # Concatenate all background samples
            background_concat = np.concatenate(subject_background, axis=1)

            # Compute covariance matrix
            cov = Covariances(estimator="oas").transform(
                background_concat[np.newaxis, :, :]
            )[0]

            # Compute whitening matrix
            self.subject_whitening_matrices_[subject] = invsqrtm(cov)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform signals using subject-specific whitening matrices.

        Parameters
        ----------
        X : structured array
            Input data with fields:
            - sample: (n_channels, n_times) array
            - subject_id: int
            - is_background: int (0 or 1)

        Returns
        -------
        ndarray
            Whitened signals
        """
        # Extract data from structured array
        signals = self._extract_signals(X)
        subject_ids = self._extract_subject_ids(X)

        # Initialize output array
        X_transformed = np.zeros_like(signals)
        X_structured_transformed = X.copy()

        # Apply whitening to each sample
        for i, (signal, subject) in enumerate(zip(signals, subject_ids)):
            if subject not in self.subject_whitening_matrices_:
                raise ValueError(f"Subject {subject} not seen during fitting")

            # Get whitening matrix
            W = self.subject_whitening_matrices_[subject]

            # Apply whitening: W @ signal
            X_transformed[i] = np.dot(W, signal)

        X_structured_transformed["sample"] = X_transformed

        return X_structured_transformed

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit the transformer and transform the data.

        Parameters
        ----------
        X : structured array
            Input data with fields:
            - sample: (n_channels, n_times) array
            - subject_id: int
            - is_background: int (0 or 1)
        y : None
            Not used, present for scikit-learn compatibility

        Returns
        -------
        ndarray
            Whitened signals
        """
        return self.fit(X, y).transform(X)


class PredictionPhaseTransformer(BaseEstimator, TransformerMixin):
    """
    A wrapper transformer that only applies its inner transformer during prediction phase.

    Parameters
    ----------
    transformer : object
        The transformer to apply during prediction phase.
        Must implement fit, transform, and fit_transform methods.

    Attributes
    ----------
    transformer_ : object
        The transformer to apply during prediction phase
    """

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not _IN_PREDICTION_PHASE:
            return X
        return self.transformer.transform(X)

    def fit_transform(self, X, y=None):
        if not _IN_PREDICTION_PHASE:
            return X
        return self.transformer.fit_transform(X, y)


class Splitter(BaseEstimator, TransformerMixin):
    """
    A transformer that splits samples into multiple segments of equal length.

    Parameters
    ----------
    n_splits : int, default=3
        Number of segments to split each sample into.
        Each segment will have length = original_length / n_splits.
    sample_field : str, default='sample'
        Name of the field in the structured array that contains the data to split.

    Attributes
    ----------
    n_splits_ : int
        The number of splits to perform
    sample_field_ : str
        The name of the field containing the data to split
    """

    def __init__(self, n_splits=3, sample_field="sample"):
        self.n_splits = n_splits
        self.sample_field = sample_field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Repeat the entire structured array n_splits times
        X_expanded = np.repeat(X, self.n_splits)

        # Split the data into n_splits segments
        original_samples = X[self.sample_field]
        segment_length = original_samples.shape[2] // self.n_splits
        segments = np.array(
            [
                original_samples[
                    :, :, i * segment_length : (i + 1) * segment_length
                ]
                for i in range(self.n_splits)
            ]
        )

        # Reshape to get the desired order: (n_samples*n_splits, n_channels, segment_length)
        X_expanded[self.sample_field] = segments.transpose(1, 0, 2, 3).reshape(
            len(X_expanded), original_samples.shape[1], segment_length
        )

        return X_expanded

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class StructuredColumnTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a single transformer to a specific column of a structured array.

    Parameters
    ----------
    column : str
        Name of the column to transform.
    transformer : object
        Transformer to apply to the specified column.
        Must implement fit, transform, and fit_transform methods.

    Attributes
    ----------
    column_ : str
        The name of the column to transform
    transformer_ : object
        The transformer to apply
    """

    def __init__(self, column, transformer):
        self.column = column
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X[self.column], y)
        return self

    def transform(self, X):
        # Apply transformer to the specified column
        transformed = self.transformer.transform(X[self.column])
        X_transformed = happend(X, transformed, self.column)

        return X_transformed


class BackgroundFilterTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that filters out all samples where is_background == True (or 1).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assumes 'is_background' is 1 for True, 0 for False
        mask = X["is_background"] == 0
        return X[mask]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class ReferenceKNN(KNeighborsClassifier):
    def fit(self, X, y):
        # we are assuming that the first sample is the reference
        # this is a hack to make the metric work.
        # TODO: fix this. metadata routing?
        reference = X["general_mean_inv"][0]
        self.metric = partial(
            reference_weighted_euclidean_distance, reference=reference
        )
        X = X["sample"]
        return super().fit(X, y)

    def predict(self, X):
        reference = X["general_mean_inv"][0]
        self.metric = partial(
            reference_weighted_euclidean_distance, reference=reference
        )
        X = X["sample"]

        return super().predict(X)

    def predict_proba(self, X):
        reference = X["general_mean_inv"][0]
        self.metric = partial(
            reference_weighted_euclidean_distance, reference=reference
        )
        X = X["sample"]
        return super().predict_proba(X)


class EyeRemoval(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=None,
        remove_veog=True,
        remove_heog=True,
        random_state=42,
    ):
        self.n_components = n_components
        self.remove_veog = remove_veog
        self.remove_heog = remove_heog
        self.random_state = random_state
        self.ica = None

    def fit(self, X, y=None):
        if X.ndim == 3:
            X = np.squeeze(X)
        n_components = (
            self.n_components if self.n_components is not None else X.shape[0]
        )
        self.ica = FastICA(
            n_components=n_components, random_state=self.random_state
        )

        X_transformed = self.ica.fit_transform(X.T).T

        veog, heog = self._create_standard_eog_channels(X)

        veog_correlations = np.array(
            [stats.pearsonr(source, veog)[0] for source in X_transformed]
        )
        heog_correlations = np.array(
            [stats.pearsonr(source, heog)[0] for source in X_transformed]
        )
        self.veog_idx = np.argmax(np.abs(veog_correlations))
        self.heog_idx = np.argmax(np.abs(heog_correlations))
        if self.veog_idx == self.heog_idx:
            indices = np.arange(len(veog_correlations))
            mask = indices != self.veog_idx
            self.heog_idx = np.argmax(np.abs(heog_correlations[mask]))
        return self

    def transform(self, X):
        if X.ndim == 3:
            X_temp = np.squeeze(X)
        else:
            X_temp = X

        X_transformed = self.ica.transform(X_temp.T)
        if self.remove_veog:
            X_transformed[:, self.veog_idx] = 0
        if self.remove_heog:
            X_transformed[:, self.heog_idx] = 0
        X_transformed = self.ica.inverse_transform(X_transformed).T
        if X.ndim == 3:
            X_transformed = X_transformed[None, :, :]
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

        correlations = np.array(
            [stats.pearsonr(source, target)[0] for source in sources]
        )
        return correlations

    # TODO: remove dependency on PICK_CHANNELS
    @classmethod
    def _create_standard_eog_channels(cls, data):
        """Create vEOG and hEOG channels from EEG data"""
        assert data.shape[0] == len(PICK_CHANNELS)
        # Get indices for EOG computation
        fp1 = data[PICK_CHANNELS == "Fp1"][0]
        fp2 = data[PICK_CHANNELS == "Fp2"][0]
        f7 = data[PICK_CHANNELS == "F7"][0]
        f8 = data[PICK_CHANNELS == "F8"][0]

        # Compute EOG channels
        veog = (fp2 + fp1) / 2
        heog = f8 - f7

        return veog, heog


class CustomSVC(SVC):  # type: ignore[misc]
    """SVC subclass with a convenience method returning
    [[predicted_label, predicted_probability]].
    """

    def predict_label_and_confidence(self, data: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(data)
        if proba.ndim == 2 and proba.shape[0] >= 1:
            pred_idx = int(np.argmax(proba[0]))
            pred = (
                self.classes_[pred_idx]
                if hasattr(self, "classes_")
                else pred_idx
            )
            conf = float(np.max(proba[0]))
            return np.array([[pred, conf]])

        pred_arr = self.predict(data)
        pred = int(pred_arr[0]) if hasattr(pred_arr, "__len__") else int(pred_arr)
        return np.array([[pred, 1.0]])