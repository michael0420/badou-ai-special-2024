import numpy as np

if __name__ == '__main__':
    data = np.genfromtxt("datasets/train_data.csv", delimiter=',')[1:]
    n = data.shape[0]
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    for i in range(n):
        s1 += data[i][0] * data[i][1]
        s2 += data[i][0]
        s3 += data[i][1]
        s4 += data[i][0] * data[i][0]

    k = (n * s1 - s2 * s3) / (n * s4 - s2 * s2)
    b = (s3 - k * s2) / n
    print("k: {}, b: {}".format(k, b))
