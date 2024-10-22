## imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


#########################################################################
###----------------------1.GAUSSIAN PROCESS---------------------------###
#########################################################################

###-------------------------------TASK 1-------------------------------##
# Loads training data from a file and extracts input features (X_train) and observations (Y_train).
def load_data(path):
    # Load training data
    training_data = np.loadtxt(path)

    # Extract inputs (X) and noisy observations (Y)
    X_train = training_data[:, 0]  # Selecting the first column for x values ; input features
    Y_train = training_data[:, 1] # observation

    # Reshape X to have only one column
    X_train = X_train.reshape(-1, 1)

    return X_train, Y_train


# Defines a rational quadratic kernel function used in Gaussian Process Regression
def modified_rational_quadratic_kernel(x1, x2, alpha, length_scale):
    # cdist calculates the squared Euclidean distance matrix between each pair of rows in x1 and x2.
    # The distances are scaled by length_scale to control the influence of each dimension.
    dist_matrix = cdist(x1 / length_scale, x2 / length_scale, metric='sqeuclidean')

    # The rational quadratic kernel formula is applied element-wise to the distance matrix
    return alpha * (1 + dist_matrix / (2 * length_scale ** 2)) ** (-alpha)


# Gaussian Process Regression function
def modified_gaussian_process_regression(X_train, Y_train, X_pred, kernel_params):
    # Extracts the kernel parameters (alpha, length_scale, beta) from the provided kernel_params
    alpha, length_scale, beta = kernel_params

    # Computes the rational quadratic kernel matrix for the training data X_train
    # Adds a regularization term (np.eye(len(X_train)) / beta) to the diagonal of the kernel matrix to improve numerical stability
    K = modified_rational_quadratic_kernel(X_train, X_train, alpha, length_scale) + np.eye(len(X_train)) / beta
    # Calculates the inverse of the kernel matrix K
    # This is used for efficient computation during prediction
    K_inv = np.linalg.inv(K)

    # Computes the rational quadratic kernel matrix between the training data X_train and the prediction data X_pred
    k_star = modified_rational_quadratic_kernel(X_train, X_pred, alpha, length_scale)

    # Calculates the mean prediction using the transpose of k_star multiplied by the inverse of the kernel matrix K_inv
    # and then multiplied by the training observations Y_train
    mean_pred = k_star.T @ K_inv @ Y_train
    # Computes the covariance matrix of predictions using the rational quadratic kernel for the prediction data X_pred
    # Includes a regularization term (1 / beta) to improve numerical stability
    cov_pred = modified_rational_quadratic_kernel(X_pred, X_pred, alpha, length_scale) + 1 / beta - k_star.T @ K_inv @ k_star

    # Returns the calculated mean prediction (mean_pred) and covariance matrix of predictions (cov_pred).
    return mean_pred, cov_pred


# Visualizes the results of Gaussian Process Regression,
# including training data, mean of f, and a 95% confidence interval.
def visualize_gp_result(X_train, Y_train, X_pred, mean_pred, cov_pred):
    # Plot training data
    plt.scatter(X_train[:, 0], Y_train, c='purple', marker='o', label='Training Data')

    # Plot mean of f
    plt.plot(X_pred[:, 0], mean_pred, label='Mean of f', color='purple')

    # Plot 95% confidence interval of f
    # Calculates the uncertainty (1.96 times the square root of the diagonal of cov_pred) to represent a 95% confidence interval
    uncertainty = 1.96 * np.sqrt(np.diag(cov_pred))
    # Fills the area between mean_pred - uncertainty and mean_pred
    # + uncertainty with an amazing purple color and 20% transparency
    # The shaded region represents the 95% confidence interval
    plt.fill_between(X_pred[:, 0], mean_pred - uncertainty, mean_pred + uncertainty, color='purple', alpha=0.2, label='95% Confidence Interval')

    # Set axis labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Show the plot
    plt.show()


###-------------------------------TASK 2-------------------------------##

# Negative marginal log-likelihood function to minimize for optimization
def negative_log_likelihood(kernel_params, X_train, Y_train):
    # Extracts the kernel parameters (alpha, length_scale, beta) from the provided kernel_params
    alpha, length_scale, beta = kernel_params

    # Computes the rational quadratic kernel matrix for the training data X_train
    # Adds a regularization term (np.eye(len(X_train)) / beta) to the diagonal of the kernel matrix to improve numerical stability
    K = modified_rational_quadratic_kernel(X_train, X_train, alpha, length_scale) + np.eye(len(X_train)) / beta
    # Calculates the inverse of the kernel matrix K. This is used for efficient computation during the calculation of log-likelihood
    K_inv = np.linalg.inv(K)

    # Calculates the log-likelihood using the formula for the negative marginal log-likelihood in Gaussian Process Regression
    log_likelihood = 0.5 * (Y_train.T @ K_inv @ Y_train + np.log(np.linalg.det(K)) + len(X_train) * np.log(2 * np.pi))

    # Returns the negative log-likelihood
    # Here the optimization algorithm aims to minimize a function,
    # because minimizing the negative log-likelihood is equivalent to maximizing the likelihood
    return log_likelihood


# Optimize kernel parameters by minimizing the negative log-likelihood.
def optimize_kernel_parameters(X_train, Y_train):
    # Provides an initial guess for the values of kernel parameters [alpha, length_scale, beta] to kick-start the optimization process
    initial_params = [1.0, 1.0, 1/5.0]  # Initial guess for [alpha, length_scale, beta]

    # Defines bounds for the optimization algorithm to constrain the search space
    # Ensures that the optimized values for each parameter are positive or non-negative
    bounds = ((1e-5, None), (1e-5, None), (1e-5, None))  # Ensure positive parameters

    # Utilizes the minimize function from the scipy.optimize module to find the values of kernel parameters that minimize the negative log-likelihood
    result = minimize(negative_log_likelihood, initial_params, args=(X_train, Y_train), bounds=bounds)

    # Extracts the optimized values of kernel parameters from the optimization result
    optimized_params = result.x

    # Returns the optimized values of kernel parameters
    return optimized_params