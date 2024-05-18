import numpy as np


def WarpPerspectiveMatrix(src, dst):
    # assert函数用来检查后面的条件式，不满足会报错
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4  # 检查大小一致，且给定点的坐标不少于4组，否则没办法解方程
    nums = src.shape[0]
    A = np.zeros((2*nums, 8))  # A*WarpMatrix = B
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]  # 遍历原始图像给定的四个坐标
        B_i = dst[i, :]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
        """
        np.mat()用于生成矩阵，与np.array相比主要有以下两点不同
        1.生成数组的计算方式不同
        array生成数组，用np.dot()表示矩阵乘积，（*）号或np.multiply()表示点乘
        mat生成数组，（*）和np.dot()相同，点乘只能用np.multiply()
        2. mat生成数组，有一种求逆的简便方式（A.I）
        """
    A = np.mat(A)
    # A*A.I=单位矩阵
    warpMatrix = A.I * B  # 没有a33
    # print(warpMatrix)
    warpMatrix = np.array(warpMatrix).T[0]
    # print("------------------------------------")
    # print(warpMatrix)
    """
    np.insert(warpMatrix, ..., values=1.0, axis=0)：这个函数会在指定的位置插入新的值。
    warpMatrix：要插入值的数组。
    warpMatrix.shape[0]：要插入新值的位置的索引。因为我们是将新值插入到数组的末尾，所以使用数组的长度（即行数）作为索引。注意，在 NumPy 中，索引是从 0 开始的，但在这里由于是在末尾插入，所以索引实际上是有效的。
    values=1.0：要插入的值，这里是浮点数 1.0。
    axis=0：指定沿哪个轴插入值。axis=0 表示在行方向（垂直方向）插入，这意味着新值将被添加为新的行。
    """
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    # print("------------------------------------")
    # print(warpMatrix)
    warpMatrix = warpMatrix.reshape(3, 3)
    return warpMatrix


if __name__ == "__main__":
    print("warpMatrix")
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)
    # print(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)


