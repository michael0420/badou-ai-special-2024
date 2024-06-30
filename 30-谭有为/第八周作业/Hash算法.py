import cv2
import numpy as  np

#均值哈希
def avg_hash(img):
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)  #将原图转换为8*8
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY) #转为灰度图
    pixel_sum=0
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            pixel_sum+=img_gray[i][j]
    pixel_avg=pixel_sum/64   #算出像素平均值
    hash_list=[]
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if img_gray[i][j]>pixel_avg:
                hash_list.append(1)
            else:
                hash_list.append(0)
    return hash_list

#差值哈希
def  dif_hash(img):
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    hash_list=[]
    for i in range(8):
        for j in range(8):
            if img_gray[i][j]>img_gray[i][j+1]:
                hash_list.append(1)
            else:
                hash_list.append(0)
    return hash_list

def pic_compare(hash_list1,hash_list2):
    num=0
    if len(hash_list1)!=len(hash_list2):
        raise ValueError('An error occurred')
    for i in range(len(hash_list1)) :
        if hash_list1[i]!=hash_list2[i]:
           num+=1
    return num

if __name__=='__main__':
    img=cv2.imread('F:/PNG/lenna.png')
    img1=cv2.imread('F:/PNG/salt_pepper_noise_lenna.png')
    hash_list=avg_hash(img)
    hash_list1=avg_hash(img1)
   # print(hash_list,hash_list1)
    Similarity1=pic_compare(hash_list,hash_list1)
    hash_list2=dif_hash(img)
    hash_list3=dif_hash(img1)
    Similarity2=pic_compare(hash_list2,hash_list3)
    print(Similarity1,Similarity2)