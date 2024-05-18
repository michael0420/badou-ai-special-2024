import numpy as np


def get_warp_matrix(src, dst):
    """
    求变换矩阵，即a11 a12 a13 a21 a22 a23 a31 a32
    :param src: 4x2
    :param dst: 4x2
    :return: 3x3
    """
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    num = src.shape[0]

    A = np.zeros((num * 2, 8))
    B = np.zeros((num * 2, 1))
    for i in range(num):
        A_i = src[i, :]
        B_i = dst[i, :]

        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2*i+1, :]=[0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1]=[B_i[1]]

    A=np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，得到变换矩阵
    warp_matrix = A.I * B

    warp_matrix = np.array(warp_matrix).T[0]
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1, axis=0)
    warp_matrix=np.reshape(warp_matrix,(3,3))

    return warp_matrix

def test():
    src = np.float32([[200, 141], [529, 280], [2, 602], [344, 742]])
    dst = np.float32([[0, 0], [437, 0], [0, 588], [437, 588]])

    res=get_warp_matrix(src,dst)

    print(res)


test()