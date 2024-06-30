import cv2
import numpy as np

def warpPerspective(src,dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >=4
    nums = src.shape[0]
    A = np.zeros((2*nums,8))
    B = np.zeros((2*nums,1))
    for i in range(nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2 * i,:] = [A_i[0],A_i[1],1,0,0,0,-A_i[0]*B_i[0],-A_i[1]*B_i[0]]
        B[2 * i,:] = B_i[0]
        A[2 * i+1, :] = [0,0,0,A_i[0],A_i[1],1,-A_i[0]*B_i[1],-A_i[1]*B_i[1]]
        B[2 * i+1, :] = B_i[1]

    A = np.mat(A)
    warpmatrix = A.I *B
    print(warpmatrix)
    warpmatrix = np.array(warpmatrix).T[0]
    warpmatrix = np.insert(warpmatrix,warpmatrix.shape[0],values=1.0,axis=0)
    print("warpmatrix:",warpmatrix)
    warpmatrix = warpmatrix.reshape((3,3))
    return  warpmatrix

if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    warpmatrix = warpPerspective(src,dst)
    print(warpmatrix)
