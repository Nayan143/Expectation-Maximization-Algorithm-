# Expectation-Maximization-Algorithm
EM algorithm implementation to solve a computer vision task of image segmentation. In particular, EM algorithm to fit one Gaussian Mixture Model (GMM) to samples of skin pixels (sdata) and one to samples of non-skin pixels (ndata). Each sample is an RGB color value. These trained GMMs used to segment an image into skin color and non-skin color regions.

# apply.py : 
- Expectation Maximization Algorithm for GMMs
- test getLogLikelihood
- test EStep
- test MStep
- test regularization
- compute GMM on all 3 datasets
- for different number of modes k plot the log likelihood for data3
- plot result
- skin detection

# getLogLikelihood.py
Implement the function getLogLikelihood that computes the log-likelihood of a mixture of Gaussian distributions with the signature (Log Likelihood estimation) 

# EStep.py :
Implement the function EStep that performs the expectation step of the EM algorithm
- Expectation step of the EM Algorithm

# MStep.py :
Implement the function MStep that performs the maximization step of the EM algorithm
- Maximization step of the EM Algorithm

# regularize_cov.py :
It is often necessary to introduce a regularization for EM to work robustly. One possibility is to add a small value to the diagonal entries of all covariance matrices: Σ_reg = Σ + ƐI. This ensures that the covariance matrix has a low condition number, which makes the computation of the inverse more stable.
- Implementation that regularizes a covariance matrix
- regularize a covariance matrix, by enforcing a minimum value on its singular values

# estGaussMixEM.py :
Implement the function estGaussMixEM for performing EM for estimating Gaussian Mixture Models of D-dimensional data.
- Initialize the weights uniformly 
- means and the covariances using the K-Means algorithm)
- EM algorithm for estimation gaussian mixture mode

# skinDetection.py
- Datasets of RGB pixel values for skin (sdata) and non-skin (ndata) regions for a skin detection experiment. First, train a Gaussian Mixture Model for each dataset. Based on these two Mixture Models, the pixels is classified in the provided image(faces.png) according to the Likelihood Ratio.
- Used different θ and compare your results to the groundtruth (faces_groundtruth.png). Which parameters θ, K, etc.
- calculate ration and threshold
