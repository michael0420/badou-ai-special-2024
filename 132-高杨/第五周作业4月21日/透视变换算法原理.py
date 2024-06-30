import numpy as np

def warpPerspectMtrix(src,dst):

    assert src.shape[0]==dst.shape[0] and src.shape[0]>=4


    nums = src.shape[0]
    A = np.zeros((2*nums,8))
    B = np.zeros((2*nums,1))

    for i in range(nums):
        # 从a矩阵中取出第一行元素
        A_i = src[i,:]
        B_i = dst[i,:]

        #取偶数行 根据公式把A矩阵和B矩阵相应的值填进去
        A[2*i, :] = [A_i[0],A_i[1],1,0,0,0,-A_i[0]*B_i[0],-A_i[1]*B_i[0]]

        B[2*i]=B_i[0]

        #取奇数行
        A[2*i+1,:] = [0,0,0,A_i[0],A_i[1],1,-A_i[0]*B_i[1],-A_i[1]*B_i[1]]
        B[2*i+1]=B_i[1]
    A = np.mat(A)   #求出A的逆矩阵 ， B矩阵左乘A的逆矩阵得出结果
    warpMtrix=A.I*B
    #转置第一列
    warpMtrix=np.array(warpMtrix).T[0]
    warpMtrix=np.insert(warpMtrix,warpMtrix.shape[0],values=1.0,axis=0)
    warpMtrix= warpMtrix.reshape((3,3))

    return warpMtrix



if __name__ == '__main__':


    # 原图的4组点
    src = [[10.0,457.0],[164.0,37.0],[147.0,247.0],[103.0,57.0]]
    src = np.array(src)

    dst = [[58.0,66.0],[12.0,154.0],[103.0,175.0],[144.0,23.0]]
    dst = np.array(dst)
    res=warpPerspectMtrix(src,dst)

    print(res)