import numpy as np
import scipy as sp
import scipy.linalg as sl
import matplotlib.pyplot as plt

def random_partition(n, n_data):
    all_indices = np.arange(n_data)
    np.random.shuffle(all_indices)
    idx1 = all_indices[:n]
    idx2 = all_indices[n:]
    return idx1, idx2 

def ransac(data, model, min_samples, max_iter, threshold, min_inliers):
    iterations = 0
    bestfit = None
    best_error = np.inf
    best_inliers_idx = None
    
    while iterations < max_iter:
        train_idx, test_idx = random_partition(min_samples, data.shape[0])
        train_points = data[train_idx]
        test_points = data[test_idx]
        solution_vec = model.fit(train_points)
        test_error = model.error(test_points, solution_vec)
        inliers_idx = test_idx[test_error < threshold]
        inliers = data[inliers_idx]
        if len(inliers) > min_inliers:
            curr_data = np.concatenate((train_points, inliers))
            curr_soul_vec = model.fit(curr_data)
            curr_error = model.error(curr_data, curr_soul_vec)
            avg_curr_error = np.mean(curr_error)
            if avg_curr_error < best_error:
                bestfit = curr_soul_vec
                best_error = avg_curr_error
                best_inliers_idx = np.concatenate((train_idx, inliers_idx))
        iterations += 1
    
    if bestfit is None:
        raise ValueError("did not find a solution")
    return bestfit, best_inliers_idx

class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns
    
    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s = sl.lstsq(A, B)
        return x
    
    def error(self, data, solution_vec):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = sp.dot(A, solution_vec)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point
    
def run():
    # generate data
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    n_outliers = 100

    A_exact = 20 * np.random.random((n_samples, n_inputs))
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = sp.dot(A_exact, perfect_fit)

    # add a little noise
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    # add some outliers
    all_idxs = np.arange(A_noisy.shape[0])
    np.random.shuffle(all_idxs)
    outliers_idxs = all_idxs[:n_outliers]
    A_noisy[outliers_idxs] = 20 * np.random.random((n_outliers, n_inputs))
    B_noisy[outliers_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # setup model
    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs) # [0]
    output_columns = [n_inputs + i for i in range(n_outputs)] # [1]
    model = LinearLeastSquareModel(input_columns, output_columns)

    linear_fit, _, _, _ = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC algorithm
    ransac_fit, ransac_fit_idxs = ransac(all_data, model, 50, 1000, 7000, 300)

    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]

    plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
    plt.plot(A_noisy[ransac_fit_idxs, 0], B_noisy[ransac_fit_idxs, 0], 'bx', label='RANSAC data')

    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:,0], label='RANSAC fit')
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:,0], label='exact system')
    plt.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:,0], label='linear fit')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    run()