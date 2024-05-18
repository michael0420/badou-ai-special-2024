import numpy as np

def fit_transform(Z,k):
    mean_z  = Z - Z.mean(axis = 0)
    print(mean_z)
    cov = np.dot(mean_z.T,mean_z) / (Z.shape[0]-1)
    print(cov)
    a,b = np.linalg.eig(cov)
    print("-----------------")
    print(a)
    print(b)
    ind = np.argsort(-1 * a)
    print(ind)

    UT = [b[:, ind[i]] for i in range(k)]
    U = np.transpose(UT)

    print("-----------------")
    print(U)
    print("*****************")

    pca_z = np.dot(Z, U)
    return pca_z


if __name__ == '__main__':
    Z = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    reduce_Z = fit_transform(Z,Z.shape[1]-1)
    print(reduce_Z)

    

