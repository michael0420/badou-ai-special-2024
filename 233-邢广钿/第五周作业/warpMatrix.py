import numpy as np

def calculateWarpMatrix(src,dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    #坐标点的数量
    num = src.shape[0]
    #构建矩阵A、B
    A = np.zeros((2*num,8))
    B = np.zeros((2*num,1))
    for i  in range(0,num):
        #原坐标跟目标坐标 第i个点的坐标点
        src_i = src[i,:]
        dst_i = dst[i,:]
        # 构建A/B矩阵行
        A[2*i,:] =[src_i[0],src_i[1],1,0,0,0,-src_i[0]*dst_i[0],-src_i[1]*dst_i[0]]
        A[2*i+1,:] = [0,0,0,src_i[0],src_i[1],1,-src_i[0]*dst_i[1],-src_i[1]*dst_i[1]]
        B[2 * i] = dst_i[0]
        B[2*i+1] = dst_i[1]

    A=np.mat(A)
    #求出warpMatrix = B/A = A.I*B
    warpMatrix = A.I*B
    warpMatrix = np.array(warpMatrix).T[0]
    #插入a33 = 1
    warpMatrix = np.insert(warpMatrix,warpMatrix.shape[0],values=1.0,axis=0)
    warpMatrix = warpMatrix.reshape((3,3))
    return warpMatrix


if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = calculateWarpMatrix(src, dst)
    print(warpMatrix)