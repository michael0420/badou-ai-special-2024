import numpy as np

if __name__ == '__main__':
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    array1 = np.array([1, 2, 3])
    array2 = np.dot(array1, array)

    print(np.array(array2).T[0])
