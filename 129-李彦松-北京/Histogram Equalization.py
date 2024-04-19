# -*- coding: gbk -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def equalHist(img):
    # ʵ��ֱ��ͼ���⻯�㷨
    # ���裺
    # ����ɨ��ԭʼ�Ҷ�ͼ���ÿһ�����أ������ͼ��ĻҶ�ֱ��ͼH
    # ����Ҷ�ֱ��ͼ���ۼ�ֱ��ͼ
    # �����ۼ�ֱ��ͼ��ֱ��ͼ���⻯ԭ��õ����������֮���ӳ���ϵ��
    # ������ӳ���ϵ�õ������dst(x, y) = H'(src(x,y))����ͼ��任
    # д������ʵ�������㷨
    # 1. ����ֱ��ͼ
    # ��һ������[dst]��Ҫ����ֱ��ͼ��ͼ���б������ֻ������һ��ͼ��dst��ֱ��ͼ��
    # �ڶ�������[0]��Ҫ�����ͨ���������б������ֻ�����˵�һ��ͨ��������Ϊ0����ֱ��ͼ�����ڻҶ�ͼ��ֻ��һ��ͨ��������������[0]�����ڲ�ɫͼ��������ͨ�����졢�̡����������Ҫ��������ͨ����ֱ��ͼ������Ӧ����[0, 1, 2]��
    # ����������None��һ����ģ��mask��������ṩ����ģ����ֻ������ģ���ص��ֱ��ͼ�������û���ṩ��ģ�����Լ����������ͼ���ֱ��ͼ��
    # ���ĸ�����[256]��ֱ��ͼ�Ĵ�С��Ҳ����ֱ��ͼ���ж��ٸ�bin�������ֱ��ͼ�Ĵ�С��256������ÿ��bin����һ������ֵ�ķ�Χ��
    # ���������[0,256]������ֵ�ķ�Χ�����������ֵ�ķ�Χ�Ǵ�0��256���������ֱ��ͼ���������п��ܵ�����ֵ��
    # ���ԣ�hist = cv2.calcHist([dst],[0],None,[256],[0,256])���д������˼�ǣ�����ͼ��dst��ֱ��ͼ��ֱ��ͼ�Ĵ�С��256������ֵ�ķ�Χ�Ǵ�0��256��Ȼ�󽫽���洢��hist�С�

    # ����ɨ��ԭʼ�Ҷ�ͼ���ÿһ������
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # ����Ҷ�ֱ��ͼ���ۼ�ֱ��ͼ
    cdf = hist.cumsum()
    # �����ۼ�ֱ��ͼ��ֱ��ͼ���⻯ԭ��õ����������֮���ӳ���ϵ
    # ����ӳ���
    table = np.zeros(256)
    for i in range(256):
        table[i] = np.uint8(255 * cdf[i] / cdf[255])
    # ����ӳ���õ����
    dst = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dst[i, j] = table[img[i, j]]

    return dst
# ��ȡ�Ҷ�ͼ��
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# �Ҷ�ͼ��ֱ��ͼ���⻯
# dst = cv2.equalizeHist(gray)
dst1 = equalHist(gray)

plt.plot(dst1)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst1])) # ������ʾԭͼ�;��⻯���ͼ��
cv2.waitKey(0)



# ��ɫͼ��ֱ��ͼ���⻯
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# ��ɫͼ����⻯,��Ҫ�ֽ�ͨ�� ��ÿһ��ͨ�����⻯
(b, g, r) = cv2.split(img)
bH = equalHist(b)
gH = equalHist(g)
rH = equalHist(r)
# �ϲ�ÿһ��ͨ��
result = cv2.merge((bH, gH, rH))

# �ֱ����ÿ��ͨ����ֱ��ͼ
# hist_b = cv2.calcHist([result],[0],None,[256],[0,256])
# hist_g = cv2.calcHist([result],[1],None,[256],[0,256])
# hist_r = cv2.calcHist([result],[2],None,[256],[0,256])

# ����ÿ��ͨ����ֱ��ͼ,��������ͼ�ֱ𻭳���
plt.subplot(311)
plt.plot(bH, color='b')
plt.subplot(312)
plt.plot(gH, color='g')
plt.subplot(313)
plt.plot(rH, color='r')
plt.show()


cv2.imshow("dst_rgb", result)
cv2.waitKey(0)
