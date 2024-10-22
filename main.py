## Imports
import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import svm_predict, svm_load_model

## custom imports
###----------------------1.GAUSSIAN PROCESS---------------------------###
from packages_gp_svm import gaussian_process

###--------------------2. SVM on MNIST dataset------------------------###
from packages_gp_svm import svm



#########################################################################
###------------------------------MAIN---------------------------------###
#########################################################################
def main():
    ###---------------Part 1 (gaussian process) : task 1---------------##
    # set parameters
    input_features, observations = gaussian_process.load_data('./data/input.data')
    # Set kernel parameters for Gaussian Process Regression
    modified_kernel_params = [1.0, 1.0, 5]  # [alpha, length_scale, beta]
    # Generates a set of points (X_prediction) for visualization
    X_prediction = np.linspace(-60, 60, 100).reshape(-1, 1)
    # Apply Gaussian Process Regression  with modified kernel parameters
    modified_mean_pred, modified_cov_pred = gaussian_process.modified_gaussian_process_regression(input_features, observations,
                                                                                 X_prediction, modified_kernel_params)
    # Visualizes the regression result using visualize_gp_result()
    gaussian_process.visualize_gp_result(input_features, observations, X_prediction, modified_mean_pred, modified_cov_pred)

    ##------------------------------------------------------------------
    ###--------------Part 1 (gaussian process) : task 2---------------##
    # Optimize kernel parameters using optimize_kernel_parameters()
    optimized_kernel_params = gaussian_process.optimize_kernel_parameters(input_features, observations)
    # Output optimized parameters
    print("Optimized Kernel Parameters:", optimized_kernel_params)
    # Apply Gaussian Process Regression with optimized kernel parameters
    optimized_mean_pred, optimized_cov_pred = gaussian_process.modified_gaussian_process_regression(input_features, observations,
                                                                                    X_prediction, optimized_kernel_params)
    # Visualize the regression result with optimized parameters
    gaussian_process.visualize_gp_result(input_features, observations, X_prediction, optimized_mean_pred, optimized_cov_pred)


    ##-----------------------------------------------------------------------
    ##-----------------------------------------------------------------------
    ###----------------------Part 2 (SVM): task 1--------------------------##
    print('-----------------------------------------------------------------')
    print('Part2 : task 1')
    # Load training and testing data
    X_train = svm.load_data_part2('./data/X_train.csv')
    Y_train = svm.load_data_part2('./data/Y_train.csv')
    X_test = svm.load_data_part2('./data/X_test.csv')
    Y_test = svm.load_data_part2('./data/Y_test.csv')

    # Subset the data for digits 0 to 4
    X_train_subset, Y_train_subset = svm.preprocess_data(X_train, Y_train)
    X_test_subset, Y_test_subset = svm.preprocess_data(X_test, Y_test)

    # Define kernel functions
    kernel_list = ['linear', 'polynomial', 'rbf', 'sigmoid', 'custom']
    params_list = ['', '-d 3', '-g 0.1', '-g 0.1 -r 0 -d 3', '']
    imaginary_numbers_list = []
    meanPixels = 0

    # Loop for other kernels
    for kernel, params in zip(kernel_list[:-1], params_list[:-1]):
        # optimzed result
        # C_optimal = 0.1
        # gamma_optimal = 0.01
        # Train SVM model with optimal C and gamma, to verify the result we obtain thanks to the task 2
        # model_file = train_svm_model(kernel, X_train, Y_train, C=C_optimal, gamma=gamma_optimal) # line to verify task2

        model_file = svm.train_svm_model(kernel, X_train_subset, Y_train_subset, params + ' -h 0')
        accuracy = svm.evaluate_svm_model(model_file, X_test_subset, Y_test_subset, params + ' -h 0')
        print(f'Accuracy for {kernel} kernel: {accuracy}%')

        # Calculate imaginary numbers
        kernel_predictions, _, _ = svm_predict(Y_test_subset, X_test_subset, svm_load_model(model_file))
        imaginary_numbers = svm.imagination_number_calculation(X_test_subset, kernel_predictions)

        # Append the imaginary numbers to the list
        imaginary_numbers_list.append(imaginary_numbers)

    # Define the kernels corresponding to imaginary_numbers_list
    kernel_list = [ 'Linear', 'Polynomial', 'RBF', 'Sigmoid']

    # Create subplots for each digit and kernel
    fig, axs = plt.subplots(5, 4, figsize=(12, 15))
    fig.suptitle("Imaginary Numbers for Each Digit and Kernel")

    for i, (imaginary_numbers, kernel_name) in enumerate(zip(imaginary_numbers_list, kernel_list)):
        for c in range(5):
            image = imaginary_numbers[c].reshape(28, 28).T
            axs[c, i].imshow(image, cmap='gray')
            axs[c, i].axis('off')
            axs[c, i].set_title(f'Digit {c} - {kernel_name} Kernel')

    # Adjust layout to prevent overlap
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()


    ##-----------------------------------------------------------------------
    ###------------------------Part 2 (SVM): task 2------------------------##
    print('------------------------------------------------------------')
    print('Part2 : task 2')
    # For RBF Kernel
    svm.grid_search_svm(X_train, Y_train, X_test_subset, Y_test_subset, 'rbf')

    # For linear kernel
    svm.grid_search_svm(X_train, Y_train, X_test_subset, Y_test_subset, 'linear')

    # For polynomial kernel with degree 3 (you can choose a different degree)
    svm.grid_search_svm(X_train, Y_train, X_test_subset, Y_test_subset, 'polynomial', degree=3)

    # For sigmoid kernel
    svm.grid_search_svm(X_train, Y_train, X_test_subset, Y_test_subset, 'sigmoid')


    ##-----------------------------------------------------------------------
    ###------------------------Part 2 (SVM): task 3------------------------##
    print('-----------------------------------------------------------------')
    print('Part2 : task 3')
    # Load training and testing data
    X_train = svm.load_data_p2_task3('data/X_train.csv')
    Y_train = svm.load_data_p2_task3('data/Y_train.csv')
    X_test = svm.load_data_p2_task3('data/X_test.csv')
    Y_test = svm.load_data_p2_task3('data/Y_test.csv')

    # Compute linear kernel matrix for training data
    linear_kernel_matrix = svm.compute_linear_kernel(X_train[:, 1:])

    # Compute RBF kernel matrix for training data
    rbf_kernel_matrix = svm.compute_rbf_kernel(X_train[:, 1:])

    # Combine linear and RBF kernels for training data
    custom_kernel_matrix_np = svm.combine_kernels(linear_kernel_matrix, rbf_kernel_matrix)

    # Prepare the input format for libsvm for training data
    svm_input = svm.prepare_svm_input(custom_kernel_matrix_np, Y_train)

    # Train the SVM model using the custom kernel matrix
    model = svm.train_svm_model_custom(Y_train, svm_input)

    # Prepare test data
    custom_test_kernel_matrix_np = svm.prepare_test_data(X_test[:, 1:], X_train[:, 1:])

    # Prepare the input format for libsvm for test data
    svm_test_input = svm.prepare_svm_test_input(custom_test_kernel_matrix_np)

    # Perform prediction
    test_labels, test_accuracy, _ = svm_predict(Y_test, svm_test_input, model)

    print('accuracy: ', test_accuracy)



if __name__ == '__main__':
    main()
