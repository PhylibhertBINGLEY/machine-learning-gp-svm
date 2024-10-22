## imports 
import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import svm_train, svm_save_model, svm_predict, svm_load_model
from scipy.spatial.distance import cdist

#########################################################################
###--------------------2. SVM on MNIST dataset------------------------###
#########################################################################
###-------------------------------TASK 1-------------------------------##
#  function loads data from a file using NumPy's loadtxt function
#  The file_path parameter is the path to the file,
#  and delimiter=',' specifies that the data is comma-separated.
def load_data_part2(file_path):
    # returns the loaded data
    return np.loadtxt(file_path, delimiter=',')

# This function preprocesses the input data (X and Y) based on a specified digit limit
# The digit_limit parameter (default is 4) is used to filter the data
def preprocess_data(X, Y, digit_limit=4):
    # creates a mask (mask) by comparing the values in Y with the digit_limit
    mask = Y <= digit_limit
    # applies this mask to both X and Y, creating subsets X_subset and Y_subset
    X_subset = X[mask]
    Y_subset = Y[mask]

    # returns these subsets
    return X_subset, Y_subset

'''
## function to test the optimal C and gamma value of task 2 from part 2
def train_svm_model(kernel, X_train, Y_train, C=1.0, gamma='auto', degree=3, coef0=0.0):
    # Use of a dictionary to map the kernel name to its corresponding libsvm type
    kernel_types = {'linear': 0, 'polynomial': 1, 'rbf': 2, 'sigmoid': 3}

    # Check if the provided kernel is valid
    if kernel not in kernel_types:
        # raises an error if not
        raise ValueError(f"Invalid kernel type: {kernel}")

    # Set the kernel type based on the provided kernel
    kernel_type = kernel_types[kernel]

    # Build the SVM parameters string based on the provided parameters
    params = f'-t {kernel_type} -c {C} -g {gamma} -d {degree} -r {coef0}'

    # Call svm_train with the specified parameters
    prob = svm_train(Y_train, X_train, params)

    # The file name includes the kernel type
    model_file = f'model_{kernel}.svm'
    # Save the trained model to a file (model_file)
    svm_save_model(model_file, prob)

    # The function returns the file name.
    return model_file
'''


# This function trains an SVM model using the LIBSVM library
# It takes the kernel type (kernel), training features (X_train), training labels (Y_train), and optional parameters (params)
def train_svm_model(kernel, X_train, Y_train, params=''):
    # Use of a dictionnary to map the kernel name to its corresponding libsvm type
    kernel_types = {'linear': 0, 'polynomial': 1, 'rbf': 2, 'sigmoid': 3}

    # Check if the provided kernel is valid
    if kernel not in kernel_types:
        # raises an error if not
        raise ValueError(f"Invalid kernel type: {kernel}")

    # Set the kernel type based on the provided kernel
    kernel_type = kernel_types[kernel]

    # The function calls svm_train with the appropriate parameters
    prob = svm_train(Y_train, X_train, f'-t {kernel_type} {params}')
    # The file name includes the kernel type
    model_file = f'model_{kernel}.svm'
    # saves the trained model to a file (model_file)
    svm_save_model(model_file, prob)

    # The function returns the file name.
    return model_file


# This function evaluates an SVM model using the LIBSVM library
# It takes the file name of the trained model (model_file), testing features (X_test), testing labels (Y_test), and optional parameters (params).
def evaluate_svm_model(model_file, X_test, Y_test, params=''):
    #  loading the model using svm_load_model
    prob = svm_load_model(model_file)

    # performing predictions using svm_predict and retrieving the accuracy
    _, accuracy, _ = svm_predict(Y_test, X_test, prob)

    # the function returns the accuracy
    return accuracy[0]


###-------------------------IMAGINATION OF NUMBERS-----------------------------##
# function to calculate the imagination of numbers from 0 to 4
# the function takes two parameters - X_test (testing features) and Y_predict (predicted labels).
def imagination_number_calculation(X_test, Y_predict):
    # his line retrieves the shape of the testing features X_test
    # where m is the number of samples and n is the number of features
    [m, n] = np.shape(X_test)

    # initializing a matrix meanPixels with zeros. It has dimensions (5, n),
    # where 5 represents the number of classes (digits) and n represents the number of features
    meanPixels = np.zeros((5, n))

    # This line initializes an array totalClass with zeros
    # It has a length of 5, representing the total count for each class.
    totalClass = np.zeros(5)

    # This line initializes a matrix imaginaryNumber with zeros. It has dimensions (5, n),
    # where 5 represents the number of classes and n represents the number of features
    imaginaryNumber = np.zeros((5, n))

    for i in range(m):
        #  extracting the predicted label for the current sample (Y_predict[i]) and converts it to an integer
        #  It subtracts 1 to adjust the label to be in the range [0, 4] (assuming classes are labeled from 1 to 5).
        c = int(Y_predict[i]) - 1
        #  incrementing the count for the class c in the totalClass array
        totalClass[c] += 1
        for j in range(n):
            # retrieves the pixel value for the current feature of the current sample
            pixel_value = X_test[i][j]
            # accumulates the pixel values for each feature in the meanPixels matrix,
            # specific to the class c
            meanPixels[c, j] += pixel_value

    #  starts another loop over the classes (digits)
    for c in range(5):
        # calculating the mean pixel value for each feature across all samples belonging to class c
        # by dividing the accumulated sum by the total count (totalClass[c])
        meanPixels[c] /= totalClass[c]
        # loop over the features
        for j in range(n):
            # checking if the mean pixel value for the current feature and class is less than 0.4
            if (meanPixels[c, j] < 0.4):
                # If the condition is true, set the corresponding entry in the imaginaryNumber matrix to 0
                imaginaryNumber[c, j] = 0
            else:
                # else set the corresponding entry in the imaginaryNumber matrix to 1
                imaginaryNumber[c, j] = 1


    # This line returns the imaginaryNumber matrix,
    # which represents the "imagination" of numbers based on the mean pixel values for each feature and class
    # Entries are 0 or 1, indicating whether the pixel value is below or above the threshold (0.4)
    return imaginaryNumber


###-------------------------------TASK 2-------------------------------##
# function which performs a grid search to find the best parameters for an SVM model with a given kernel type
def grid_search_svm(X_train, Y_train, X_test_subset, Y_test_subset, kernel_type, degree=None):
    # This line checks if the provided kernel_type is one of the supported kernel types (linear, polynomial, sigmoid, or rbf)
    if kernel_type not in ['linear', 'polynomial', 'sigmoid', 'rbf']:
        # If not, it raises a ValueError
        raise ValueError(f"Invalid kernel type: {kernel_type}")

    # defines a list of candidate values for the regularization parameter C in the SVM model
    C_values = [0.1, 10, 0.1, 10]
    # This line defines a list of candidate values for the kernel coefficient gamma in the SVM model
    gamma_values = [0.01, 0.01, 0.01, 0.01]

    #  checks if the kernel type is polynomial
    if kernel_type == 'polynomial':
        # If true, it adds the degree parameter to the grid for polynomial kernel
        # Add degree to the parameter grid for polynomial kernel
        if degree is None:
            raise ValueError("Degree must be specified for polynomial kernel.")
        param_grid = [(C, gamma, degree) for C in C_values for gamma in gamma_values]
    else:
        # If the kernel type is not polynomial, it executes the following block
        # For non-polynomial kernels, it creates a grid without the degree parameter
        param_grid = [(C, gamma) for C in C_values for gamma in gamma_values]

    #  initializes a matrix accuracies with zeros
    #  The matrix will store the accuracy values for each combination of C and gamma
    accuracies = np.zeros((len(C_values), len(gamma_values)))

    # This line starts a loop over the parameter grid,
    # where i is the index and (C, gamma, *extra_params) unpacks the current parameter combination
    for i, (C, gamma, *extra_params) in enumerate(param_grid):
        # Train SVM model
        # Checks if the kernel type is linear
        # If true, it sets the SVM parameters accordingly
        if kernel_type == 'linear':
            # Creates a string params representing the SVM parameters for a linear kernel
            params = f'-t 0 -c {C}'
        # Checks if the kernel type is polynomial
        # If true, it sets the SVM parameters accordingly
        elif kernel_type == 'polynomial':
            # Creates a string params representing the SVM parameters for a polynomial kernel
            params = f'-t 1 -c {C} -g {gamma} -d {extra_params[0]}'
        # Checks if the kernel type is sigmoid
        # If true, it sets the SVM parameters accordingly
        elif kernel_type == 'sigmoid':
            # Creates a string params representing the SVM parameters for a sigmoid kernel
            params = f'-t 3 -c {C} -g {gamma}'
        # Checks if the kernel type is RBF. If true, it sets the SVM parameters accordingly
        elif kernel_type == 'rbf':
            # Creates a string params representing the SVM parameters for an RBF kernel
            params = f'-t 2 -c {C} -g {gamma}'


        # Trains an SVM model using the specified kernel type and parameters and returns the model file name
        model_file = train_svm_model(kernel_type, X_train, Y_train, params)

        # Evaluate SVM model on the test subset and returns the accuracy
        accuracy = evaluate_svm_model(model_file, X_test_subset, Y_test_subset)

        # Store accuracy in the matrix
        # Stores the accuracy in the accuracies matrix at the corresponding position
        accuracies[i // len(gamma_values), i % len(gamma_values)] = accuracy

    # Visualize results as a heatmap
    plt.figure(figsize=(8, 6))
    # Creates a heatmap of accuracy values using Matplotlib,
    # visualizing the performance across different C and gamma values
    plt.imshow(accuracies, cmap='viridis', origin='lower',
               extent=[min(gamma_values), max(gamma_values), min(C_values), max(C_values)])
    # Adds a colorbar to the plot indicating the accuracy scale
    plt.colorbar(label='Accuracy')
    # Labels the x-axis as 'Gamma'
    plt.xlabel('Gamma')
    # Labels the y-axis as 'C'
    plt.ylabel('C')
    # Adds a title to the plot indicating the kernel type
    plt.title(f'Accuracy for different C and gamma values (Kernel: {kernel_type.capitalize()})')
    # Displays the heatmap
    plt.show()

    ## Find the best parameters
    # Finds the indices of the maximum accuracy in the accuracies matrix
    best_indices = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    # Finds the best C value corresponding to the maximum accuracy
    best_C = C_values[best_indices[0]]
    # Finds the best gamma value corresponding to the maximum accuracy
    best_gamma = gamma_values[best_indices[1]]
    # This line retrieves the best accuracy based on the indices of the maximum accuracy
    best_accuracy = accuracies[best_indices]

    # These lines print the best parameters and the corresponding best accuracy found during the grid search
    print(f'Best Parameters: C = {best_C}, gamma = {best_gamma}')
    print(f'Best Accuracy: {best_accuracy}%')




###-------------------------------TASK 3-------------------------------##
# This function loads data from a file using NumPy's loadtxt function
# the data is stored in a CSV format with a comma as the delimiter
def load_data_p2_task3(file_path):
    # The loaded data is returned
    return np.loadtxt(file_path, delimiter=',')

# Compute the linear kernel matrix for a given set of features
def compute_linear_kernel(features):
    #  Computes the dot product of the feature matrix (features) with its transpose (features.T)
    #  This operation results in a matrix where each element (i, j) represents the inner product of the feature vectors for data points i and j
    # Return : the computed linear matrix kernel
    return features @ features.T

# Compute the RBF (Gaussian) kernel matrix for a given set of features
def compute_rbf_kernel(features, gamma=0.1):
    # Calculates the squared Euclidean distances between all pairs of data points in the feature matrix
    distances_squared = cdist(features, features, 'sqeuclidean')

    # Applies the Gaussian kernel formula using the squared distances and a specified gamma parameter
    # Return : the computed RBF matrix kernel
    return np.exp(-gamma * distances_squared)

# Combine the linear and RBF kernel matrices
def combine_kernels(linear_kernel, rbf_kernel):
    # Adds the corresponding elements of the linear and RBF kernel matrices element-wise
    # Returns the combined kernel matrix
    return linear_kernel + rbf_kernel

# Prepare input format for training an SVM model
def prepare_svm_input(kernel_matrix, labels):
    #  Initializes an empty list to store data points in the SVM input format
    svm_input = []
    # loop over each row in 'kernel_matrix'
    for i, row in enumerate(kernel_matrix.tolist()):
        # {0: i + 1, **{k + 1: v for k, v in enumerate(row)}}: Creates a dictionary where 0 represents the index,
        # and the remaining keys represent the kernel values
        # Appends the dictionary to svm_input
        svm_input.append({0: i + 1, **{k + 1: v for k, v in enumerate(row)}})

    # returns the list of dictionaries representing SVM input
    return svm_input


# Train an SVM model with a custom kernel
def train_svm_model_custom(train_labels, svm_input):
    # Calls the LIBSVM library's svm_train function to train an SVM model
    # The custom kernel type is specified as '-t 4',
    # indicating the use of a custom kernel
    # Returns the trained SVM model
    return svm_train(train_labels, svm_input, '-t 4')

# Prepare test data for SVM prediction with a custom kernel
def prepare_test_data(test_features, train_features, gamma=0.1):
    # Calculates the squared Euclidean distances between test and training features
    test_distances_squared = cdist(test_features, train_features, 'sqeuclidean')

    # Computes the RBF kernel matrix for test data using the specified gamma
    test_rbf_kernel_matrix = np.exp(-gamma * test_distances_squared)

    # Returns the composite kernel matrix by adding the linear and RBF components
    return test_features @ train_features.T + test_rbf_kernel_matrix


# Prepare input format for SVM prediction with a custom kernel for test data
def prepare_svm_test_input(test_kernel_matrix):
    # Similar to prepare_svm_input,
    # it converts the test kernel matrix into a list of dictionaries representing SVM input for prediction
    svm_test_input = []
    for i, row in enumerate(test_kernel_matrix.tolist()):
        svm_test_input.append({0: i + 1, **{k + 1: v for k, v in enumerate(row)}})
    return svm_test_input
