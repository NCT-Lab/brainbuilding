import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ThirdOrderMomentTensorTransformer(BaseEstimator, TransformerMixin):
    """
    Calculates unique values of the 3rd order central moment tensor
    for each sample in a 3D array, with OAS regularization.

    Input X is expected to have shape: (n_samples, n_variables, m_variable_samples)
    Output Y will have shape: (n_samples, n_unique_moments)
    where n_unique_moments = n_variables * (n_variables + 1) * (n_variables + 2) // 6.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        This transformer is stateless, so fit does nothing.
        X shape: (n_samples, n_variables, m_variable_samples)
        """
        return self

    def _calculate_oas_tensor_for_sample(self, sample_data_matrix):
        """
        Calculates the OAS regularized 3rd order central moment tensor for a single sample's data.
        sample_data_matrix shape: (n_variables, m_variable_samples)
        """
        n_variables, m_variable_samples = sample_data_matrix.shape

        # Transpose for einsum: (m_variable_samples, n_variables)
        X_sample_proc = sample_data_matrix.T 

        mu = np.mean(X_sample_proc, axis=0) # Will warn/NaN on m_variable_samples=0
        Xc = X_sample_proc - mu
        # Potential DivisionByZero if m_variable_samples is 0
        hat_M = np.einsum('aj,ak,al->jkl', Xc, Xc, Xc) / m_variable_samples
        
        N_oas = m_variable_samples
        D_oas = n_variables

        norm_hat_M_sq = np.sum(hat_M * hat_M)
        sum_diag_hat_M = np.einsum('iii->', hat_M)
        sum_diag_hat_M_sq = sum_diag_hat_M * sum_diag_hat_M

        # Direct calculation, may lead to ZeroDivisionError or Inf/NaN
        # if D_oas is 0 or specific value combinations occur.
        denominator_val = (N_oas + 1 - 2/D_oas) * norm_hat_M_sq + \
                          (1 - N_oas/D_oas) * sum_diag_hat_M_sq
        
        numerator_val = (1 - 2/D_oas) * norm_hat_M_sq + sum_diag_hat_M_sq
        rho_3rd_oas = numerator_val / denominator_val
        
        # These constraints are part of the OAS algorithm definition.
        rho_3rd_oas = np.minimum(rho_3rd_oas, 1.0)
        rho_3rd_oas = np.maximum(rho_3rd_oas, 0.0)

        central_moments_3rd_order_OAS = (1 - rho_3rd_oas) * hat_M
        return central_moments_3rd_order_OAS

    def _extract_unique_elements(self, tensor_3d):
        """
        Extracts unique elements from a symmetric 3rd order tensor M_jkl (j <= k <= l).
        tensor_3d shape: (D, D, D)
        """
        D = tensor_3d.shape[0]
        # If D=0, loops won't run, returns empty list, which is fine.
        unique_elements = []
        for j in range(D):
            for k in range(j, D): 
                for l in range(k, D):
                    unique_elements.append(tensor_3d[j, k, l])
        return np.array(unique_elements)

    def transform(self, X, y=None):
        """
        Transforms the input data.
        X shape: (n_samples, n_variables, m_variable_samples)
        Returns: np.ndarray of shape (n_samples, n_unique_moments)
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        if X.ndim != 3:
            raise ValueError(f"Input X must be a 3D array. Got {X.ndim} dimensions. Shape: {X.shape}")

        n_samples, n_variables, m_variable_samples = X.shape
        
        # num_unique_output_features will be 0 if n_variables is 0.
        num_unique_output_features = n_variables * (n_variables + 1) * (n_variables + 2) // 6
        
        all_samples_unique_moments = np.empty((n_samples, num_unique_output_features))

        for i in range(n_samples):
            sample_data = X[i, :, :]
            oas_tensor = self._calculate_oas_tensor_for_sample(sample_data)
            unique_moments = self._extract_unique_elements(oas_tensor)
            all_samples_unique_moments[i, :] = unique_moments
        
        return all_samples_unique_moments

# Example usage:
if __name__ == '__main__':
    N_s_test = 10
    D_v_test = 3 
    M_vs_test = 50

    X_test_data = np.random.rand(N_s_test, D_v_test, M_vs_test) 
    X_test_data = (X_test_data - 0.5) * 2 
    X_test_data = X_test_data + (np.random.chisquare(df=2, size=(N_s_test, D_v_test, M_vs_test)) - 2) / 5 

    transformer = ThirdOrderMomentTensorTransformer()
    transformed_features = transformer.transform(X_test_data)

    print(f"Original data shape: {X_test_data.shape}")
    print(f"Transformed features shape: {transformed_features.shape}") 
    
    if N_s_test > 0 and D_v_test > 0 and transformed_features.size > 0:
        print("Transformed features (first sample's unique moments):")
        print(transformed_features[0])
