import cv2
import numpy as np

class HashCompare:
    def __init__(self,src,dst):
        self.src = src
        self.dst = dst
        self.processed1,self.processed2 = self.processPic()

    def processPic(self):
        H_1, H_2 = self.src, self.dst
        H_1, H_2 = cv2.cvtColor(H_1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(H_2, cv2.COLOR_BGR2GRAY)
        H_1, H_2 = cv2.resize(H_1, (8, 8), interpolation=cv2.INTER_CUBIC), cv2.resize(H_2, (8, 8),interpolation=cv2.INTER_CUBIC)
        return H_1,H_2
    def eHash(self):
        #灰度化

        H_1_mean, H_2_mean = np.sum(self.processed1) / 64 ,np.sum(self.processed2) / 64
        s1,s2 = '',''
        # 遍历每个图像，得到每个图像的哈希值
        for i in  range(8):
            for j in range(8):
                if self.processed1[i,j] > H_1_mean:
                    s1 +='1'
                else:
                    s1 +='0'
                if self.processed2[i, j] > H_2_mean:
                    s2 += '1'
                else:
                    s2 += '0'
        print(s1)
        print(s2)
        postive = 0
        negative = 0
        for m,n in zip(s1,s2):
            if m !=n:
                negative+=1
            else:
                postive+=1

        acc =  postive / (postive+negative)
        print(f'均值哈希相似度是：{negative}')

        return acc



    def dHash(self):
        H_1, H_2 = self.src, self.dst
        H_1, H_2 = cv2.cvtColor(H_1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(H_2, cv2.COLOR_BGR2GRAY)
        H_1 = cv2.resize(H_1, (8, 9))
        H_2 = cv2.resize(H_2, (8, 9))
        s1 = ''
        s2 = ''
        for i in  range(8):
            for j in range(7):
                if H_1[i,j] > H_1[i,j+1]:
                    s1 +='1'
                else:
                    s1 +='0'
                if H_2[i, j] > H_1[i,j+1]:
                    s2 += '1'
                else:
                    s2 += '0'
        postive = 0
        negative = 0
        for m, n in zip(s1, s2):
            if m != n:
                negative += 1
            else:
                postive += 1

        acc = postive / (postive + negative)
        print(f'差值哈希相似度是：{negative}')
        return acc

if __name__ == '__main__':

    src = cv2.imread('source/lenna.png')
    dst = cv2.imread('source/lenna_blur.jpg')
    hasco =   HashCompare(src,dst)
    accuracy = hasco.eHash()
    accuracy2 = hasco.dHash()
    print(accuracy,accuracy2)




