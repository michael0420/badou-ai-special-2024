"""
RANSAC - RANDOM SAMPLE CONSENSUS
* It is considered as a method, not a model
* A linear regression method to deal with dataset with significant amount of outliers.
* OLS is more commonly used than RANSAC when dataset is relatively clean and without many outliers.
* Method:
1. random sampling - randomly pick couple data point and assume as validated data
2. model fitting - fit these data points to the know model and get the coeff. or constant
                - must know the type of model for ransac
                - OLS is used at this step
3. fit the model with the rest of data point, calculate how many of them are valid and not outliers (model fit error).
4. mark down the total number of data points that fit the model.
5. iterate(repeat) the above step
6. compare and find the best model that fit the most data points

"""

import numpy as np
import scipy as sp
import scipy.linalg as sl

# define a ransac function
"""
data - data point
model - known model
n - the least # of selected data points to fit the model
k - the max # of iteration
t - a tolerance threshold used to judge whether a point fits the model
d -  Minimum Number of Points Required to Consider a Model as Good  
"""


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    i = 0  # iteration
    bestfit = None
    best_err = np.inf  # initialize the best error to a very large value, target is minimize it
    best_inlier_idx = None  # default there is no best sets of inliers

    # iteration loop
    while i < k:
        # 1-split dataset into two sets. maybe_idx for model generation (step 1, 2), test_idx for testing model (step 3)
        maybe_idx, test_idx = random_partition(n, data.shape[0])
        print('test_idx = ', test_idx)
        # maybe_idx stores the random selected indices, then need to extract the actual corresponding data point
        # select rows from data based on 'maybe_idxs'. The ':' indicates that all columns should be selected.
        maybe_inliers = data[maybe_idx, :]
        test_points = data[test_idx]

        # 2-model fitting
        maybe_model = model.fit(maybe_inliers)  # apply the class LinearLeastSquareModel

        # 3-testing model and calculate error (squared diff)
        test_err = model.get_error(test_points, maybe_model)
        # compare test_err with t and return boolean value
        print('test_err = ', test_err < t)
        # store the indices from test_idx that meet the condition of test_err < t
        also_idx = test_idx[test_err < t]
        print('also_idx = ', also_idx)
        also_inliers = data[also_idx, :]  # extract data point from indices

        # The if debug: block is executed only if the debug flag is set to True.
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (i, len(also_inliers)))

        # 4,6-determine whether the model is a good model by satisfying the condition of d
        # if > d, discard the model and continue to iterate
        if (len(also_inliers) < d):
            better_data = np.concatenate((maybe_inliers, test_points))  # combine two subsets
            better_model = model.fit(better_data)
            better_err = model.get_error(better_data, better_model)
            avg_err = np.mean(better_err)  # get the avg error as the new error
            # compare if the current error is smaller than the previous error
            # if yes, replace the best fit model with the current model
            if avg_err < best_err:
                bestfit = better_model
                best_err = avg_err
                best_inlier_idx = np.concatenate((maybe_idx, also_idx))  # update # of inliers

        i += 1

    # when the current model does not satidfy the condition of d
    if bestfit is None:
        raise ValueError("It doesn't fit the model criteria")

    # When return_all is True, the function returns the best fitting model and a dictionary with info of the inliers.
    # When return_all is set to False, the function only returns the best fitting model.
    # Optional
    if return_all:
        return bestfit, {'inliers': best_inlier_idx}
    else:
        return bestfit


# define a function to split the dataset to two subsets.
# 1st set with min n number of data points to fit model
# 2nd set with the rest to test the model
# good practice: arrange based on indices first, then shuffle
# attention: only split the indices of data
def random_partition(n, n_data):
    all_idx = np.arange(n_data)  # generate an array of indices from 0 to n_data-1 for the dataset
    np.random.shuffle(all_idx)  # shuffle indices other than data value itself is better practice and more efficient
    idx1 = all_idx[:n]  # Take the first n indices as the first subset
    idx2 = all_idx[n:]  # Take the remaining indices as the second subset
    return idx1, idx2


# from the model parameter, ransac function apply the following class to fit model
class LinearLeastSquareModel:
    # initializes the instance variables of the class.
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # format matrix A and B
        # Stack the input and output columns vertically and transpose them
        # A is the matrix of all Xi, B is the matrix of all Yi
        # data[:, i] extracts all values from column i of data.
        # np.vstack(...) stacks these arrays vertically to form a 2D array.
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # First column Xi -> row Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # Second column Yi -> row Yi

        # Solve the linear least squares problem
        # x: The solution vector (model coeff. parameters) that minimizes the residuals.
        # residues: Sum of residuals: the differences between observed and fitted data.
        x, resids, rank, s = sl.lstsq(A, B)
        return x  # Return the vector

    def get_error(self, data, model):
        # Stack and transpose input and output columns to form matrix A and B
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T

        # Calculate the predicted output values using the model
        # performs matrix multiplication between A (input features) and 'model' (coefficients) to obtain B
        # B, the predicted output values.
        B_fit = np.dot(A, model)  # B_fit = model.k * A + model.b

        # Calculate the sum of squared errors for each row
        # calculate the error btw the observed output and the predicted output for each data point.
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


# The main function used to run and test ransac method
def test():
    # 1 - Generate dataset part 1
    n_samples = 500  # total sample
    n_inputs = 1  # input number
    n_outputs = 1  # output number

    # np.random.random generates random numbers in the interval of [0.0, 1.0)
    # create an array of 500*1 (500 rows, 1 colum), then scale up the number by 20*, serve as Xi
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # randomly generate an array of (500*1)
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # get random slope from normal distribution
    B_exact = sp.dot(A_exact, perfect_fit)  # Yi = Xi* k

    # add gaussian noise(generate random # from gaussian distribution and has the same shape as A,B)
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1 array, serve as Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1 array, serve as Yi

    # generate dataset part 2
    # if True, enable this block of code to add significant outliers into dataset (not just noise)
    if 1:
        n_outliers = 100
        # generates an array of integers from 0 to 'A_noisy.shape[0] - 1'. serve as indices, total 500.
        all_idx = np.arange(A_noisy.shape[0])
        # get random 100 indices as outliers (1.arrange indices of the 500 data, 2.shuffle, 3.get the first 100 indices)
        np.random.shuffle(all_idx)
        outlier_idx = all_idx[:n_outliers]
        # 4.replaced the indices of dataset(A_noisy) with the selected indices for outlier(outlier_idx)
        # 5.random numbers drawn from np.random.random are filled in these indices
        A_noisy[outlier_idx] = 20 * np.random.random((n_outliers, n_inputs))
        B_noisy[outlier_idx] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # 2- set up model
    all_data = np.hstack((A_noisy, B_noisy))  # format the final dataset into 500*2 array
    # define which columns are input. Explicitly defined by indices, starting from 0 to n_inputs - 1.
    # n_inputs=1, so get the 1st column as input for model
    input_columns = range(n_inputs)
    # define which columns are output of model.
    # get the output column indices starting from 'n_inputs' to 'n_inputs + n_outputs - 1'.
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False
    # define the model with linear least square model with input and output columns
    # The instance is used to fit a linear model to the data and evaluate its performance,
    # facilitating tasks like fitting the model and calculating prediction errors.
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)

    # 3 - uses the lstsq function from the SciPy library to solve the linear least squares problem.
    # This function finds the best-fitting linear model to the input data by minimizing the sum of squared residuals.
    # linear_fit - coefficients of the best-fitting linear model.
    # resids: sum of squared residuals of the solution.(measures the discrepancy btw the observed and predicted values.)
    # rank: The effective rank of matrix A. This gives the number of linearly independent rows or columns in the matrix.
    # s: The singular values of the matrix A. These are used in determining the rank and the stability of the solution.
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # 3- run RANSAC function
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    # 4 - plot data and draw graph, only run the following block of code when the above code ran with no error
    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  #秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  #散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    test()
