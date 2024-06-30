import numpy as  np
import cv2

#已知顶点求常数矩阵
def perspective_transformation(original_vertex,new_vertex):
    #assert断言： 原图顶点数必须大于等于4个  且原图顶点和新顶点数必须一致，如断言失败，会引发异常
    assert original_vertex.shape[0]==new_vertex.shape[0] and original_vertex.shape[0]>=4

    vertex_nums=original_vertex.shape[0]  #顶点数
    A=np.zeros([vertex_nums*2,8])      #B=warpmatrix*A，已知A,B，warpmatrix=A_T*B A_T表示A的逆矩阵
    B=np.zeros([vertex_nums*2,1])

    for i in range(0,vertex_nums):
        Ai=original_vertex[i,:]   #取出原图的一个顶点
        Bi=new_vertex[i,:]    #取出新图的一个顶点
        A[2*i,:]=[Ai[0],Ai[1],1,0,0,0,-1*Ai[0]*Bi[0],-1*Ai[1]*Bi[0]]
        B[2*i]=Bi[0]
        A[2*i+1,:]=[0,0,0,Ai[0],Ai[1],1,-1*Ai[0]*Bi[1],-1*Ai[1]*Bi[1]]
        B[2*i+1]=Bi[1]     #将顶点的值写入矩阵A B
    A=np.mat(A)  #将A转换为矩阵
    A_T=np.linalg.inv(A) #求A的逆矩阵
    warpmatrix=A_T*B   #求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    warpmatrix=warpmatrix.T  #.T表示转置，列转为行
    a=[1]
    warpmatrix=np.column_stack((warpmatrix,a))  #矩阵加列
    warpmatrix=warpmatrix.reshape((3,3)) #将原矩阵转换为3*3矩阵
    return warpmatrix

if __name__=='__main__':
    img=cv2.imread('F:/PNG/paper.jpg')
    cv2.imshow('Original image',img)
    print(img.shape)
    original_vertex = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    new_vertex= np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    warpmatrix=perspective_transformation(original_vertex,new_vertex)   #调用手写函数perspective_transformation求出转换矩阵
#   warpmatrix=cv2.getPerspectiveTransform(original_vertex,new_vertex)  #直接调用cv2函数求出转换矩阵
    tilt_img=cv2.warpPerspective(img,warpmatrix,(540,960))
    print(warpmatrix)
    cv2.imshow('tilt_img',tilt_img)
    cv2.waitKey()
    cv2.destroyAllWindows()







