import numpy as np


def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix

    # Implementation that regularizes a covariance matrix    
    n, m = covariance.shape
    regularized_cov = covariance + epsilon * np.eye(n, m)

    # matrix is symmetric upto 1e-15 decimal
    regularized_cov = (regularized_cov + regularized_cov.conj().transpose()) / 2


    return regularized_cov
