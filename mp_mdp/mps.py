import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# weights, means, and covariances are your provided data
weights = np.array([0.1, 0.4, 0.5])  # example weights
means = np.array([1, 2, 3])  # example means
covariances = np.array([0.1, 0.2, 0.3])  # example variances (GMM needs standard deviations)

gmm = GaussianMixture(n_components=3)
gmm.weights_ = weights
gmm.means_ = means.reshape(-1, 1)  # needs to be column vector
gmm.covariances_ = covariances.reshape(-1, 1, 1)  # needs to be matrix
gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

# Create a grid of points where the PDF will be evaluated
x = np.linspace(-2, 6, 1000).reshape(-1, 1)

# Compute the PDF of the GMM at each point on the grid
pdf = np.exp(gmm.score_samples(x))

# Plot the PDF
plt.plot(x, pdf, label='GMM')

# Generate some samples and plot them
samples = gmm.sample(10000)[0]
plt.hist(samples, bins=50, density=True, alpha=0.5, label='Samples')

plt.legend()
plt.show()

