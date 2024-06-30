#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
def warpPerspectiveMatrix(src,dst):
    assert src.shape[0] ==dst.shape[0] and src.shape[0]>=4
    
    nums =src.shape[0]
    A =np.zeros([2*nums,8])
    B =np.zeros([2*nums,1])
    
    for i in range(nums):
        A_i =src[i,:]
        B_i =dst[i,:]
        
        A[2*i,:] =[A_i[0],A_i[1],1,0,0,0,-A_i[0]*B_i[0],-A_i[1]*B_i[0]]
        B[2*i,:] =B_i[0]
        A[2*i+1,:] =[0,0,0,A_i[0],A_i[1],1,-A_i[0]*B_i[1],-A_i[1]*B_i[1]]
        B[2*i+1,:]=B_i[1]
    #求warpMatrix    
    A =np.mat(A)
    warpMatrix =A.I *B
    #插入a33=1
    warpMatrix =np.array(warpMatrix).T[0]
    warpMatrix =np.insert(warpMatrix,warpMatrix.shape[0],values=1,axis=0)
    warpMatrix =warpMatrix.reshape((3,3))
    return warpMatrix    
    
if __name__ =='__main__':
    img =cv2.imread('photo1.JPG')
    src =[[207,151],[517,285],[17,601],[343,731]]
    src =np.array(src)
    dst =[[0,0],[337,0],[0,448],[337,448]]
    dst =np.array(dst)
    print(img.shape)

    m = warpPerspectiveMatrix(src,dst)
    print('warpMatrix')
    print(m)

    result =cv2.warpPerspective(img,m,(337,488))
    cv2.imshow('src',img)
    cv2.imshow('dst',result)
    cv2.waitKey(0)
    


# In[ ]:




