import numpy as np  # 导入NumPy库，用于矩阵运算

'''
### 代码逻辑和作用
1. ** 导入库 **:
- 导入NumPy库，用于矩阵和数组运算。

2. ** 定义函数
`WarpPerspectiveMatrix` **:
- ** 输入 **: 源点坐标数组
`src`
和目标点坐标数组
`dst`。
- ** 输出 **: 3
x3的透视变换矩阵。

3. ** 检查输入有效性 **:
- 确保
`src`
和
`dst`
点数相同且不少于4个点。

4. ** 初始化矩阵A和B **:
- A矩阵用于存储线性方程组的系数。
- B矩阵用于存储目标点的坐标。

5. ** 构建矩阵A和B **:
- 遍历每个点，填充A和B矩阵。
- 对于每个点，构建两行方程，分别对应x和y坐标的变换关系。

6. ** 求解透视变换矩阵 **:
- 通过矩阵运算求解A的逆矩阵并与B相乘，得到透视变换矩阵的前8个元素。
- 插入第9个元素（固定值1），构成3x3的透视变换矩阵。

7. ** 主程序 **:
- 定义源点和目标点坐标。
- 调用
`WarpPerspectiveMatrix`
函数计算透视变换矩阵。
- 打印结果。

该代码功能是计算从一个平面到另一个平面的透视变换矩阵，常用于图像处理中的透视校正和映射。'''

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4  # 确保输入的点数量相同且不少于4个点

    nums = src.shape[0]  # 获取点的数量
    A = np.zeros((2 * nums, 8))  # 初始化A矩阵，大小为(2*nums, 8)
    B = np.zeros((2 * nums, 1))  # 初始化B矩阵，大小为(2*nums, 1)

    for i in range(0, nums):  # 遍历每个点
        A_i = src[i, :]  # 获取src中的第i个点
        B_i = dst[i, :]  # 获取dst中的第i个点
        # 设置A矩阵的第2*i行
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]  # 填写A矩阵的第2*i行
        B[2 * i] = B_i[0]  # 填写B矩阵的第2*i行

        # 设置A矩阵的第2*i+1行
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]  # 填写A矩阵的第2*i+1行
        B[2 * i + 1] = B_i[1]  # 填写B矩阵的第2*i+1行

    A = np.mat(A)  # 将A矩阵转换为NumPy矩阵类型
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出透视变换矩阵的前8个元素 (a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32)

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]  # 将warpMatrix转换为NumPy数组并转置
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))  # 将warpMatrix重塑为3x3矩阵
    return warpMatrix  # 返回透视变换矩阵


if __name__ == '__main__':  # 主程序入口
    print('warpMatrix')  # 打印提示信息
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]  # 定义源点坐标
    src = np.array(src)  # 将源点坐标转换为NumPy数组

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]  # 定义目标点坐标
    dst = np.array(dst)  # 将目标点坐标转换为NumPy数组

    warpMatrix = WarpPerspectiveMatrix(src, dst)  # 计算透视变换矩阵
    print(warpMatrix)  # 打印透视变换矩阵
