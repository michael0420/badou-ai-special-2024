#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist��ֱ��ͼ���⻯ - ʹֱ��ͼ����ƽ��
    ֱ��ͼ���������ͼ������ظ�����ĻҶȼ����ĸ��������ظ����ٵĻҶȼ�����ѹ�����Ӷ��ﵽ���ͼ��ĶԱȶȵ�Ŀ�ġ�
    ��ֱ��ͼ��ֱ��Ч��������������y��Ƚϸߵ�λ�ñ䰫��x�᷽�����ͣ�y��Ƚϰ���λ�ñ�߲���x�᷽��ѹ����
����ԭ�ͣ� equalizeHist(src, dst=None)
src��ͼ�����(��ͨ��ͼ��)
dst��Ĭ�ϼ���
'''

# ��ȡ�Ҷ�ͼ��
img = cv2.imread("D:\cv_workspace\picture\lenna.png", 1)

# ԭͼת�Ҷ�ͼ: cv2.cvtColor(p1,p2) ����ɫ�ռ�ת��������p1����Ҫת����ͼƬ��p2��ת���ɺ��ָ�ʽ��
#                     ������code�� cv2.COLORBGR2GRAY,cv2.COLORBGR2HSV,cv2.COLOR_BGR2RGB
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image_gray", gray)

# �Ҷ�ͼ��ֱ��ͼ���⻯
dst = cv2.equalizeHist(gray)

# ����ֱ��ͼ
# cv2.calcHist(images,channels,mask,histSize,ranges)
# images: ԭͼ��ͼ���ʽΪ uint8 �� ?oat32�������뺯��ʱӦ �������� [] ��������[img]
# channels: ͬ������������,�������ͼ���ǻҶ�ͼ����ֵ����[0]������ǲ�ɫͼ��Ĵ���Ĳ��������� [0][1][2]�� ���Ƿֱ��Ӧ�� BGR��
# mask: ��ģͼ��ͳ����ͼ���ֱ��ͼ�Ͱ���Ϊ None�������������ͳͼ��ĳһ�ֵ�ֱ��ͼ���������һ����ģͼ��ʹ������
# histSize:BIN ����Ŀ��ҲӦ������������    ranges: ����ֵ��Χ��Ϊ [0 256]
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

# �����Զ���ͼ�� figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
# num:ͼ���Ż����ƣ�����Ϊ��� ���ַ���Ϊ����
# figsize:ָ��figure�Ŀ�͸ߣ���λΪӢ�磻
# dpi����ָ����ͼ����ķֱ��ʣ���ÿӢ����ٸ����أ�ȱʡֵΪ80 1Ӣ�����2.5cm,A4ֽ�� 21*30cm��ֽ��
# facecolor:������ɫ
# edgecolor:�߿���ɫ
# frameon:�Ƿ���ʾ�߿�
plt.figure()

# ����ֱ��ͼplt.hist(): ��һ���������״ͼ��
# ��ͳ��ֵ�ķ�Χ�ֶΣ���������ֵ�ķ�Χ�ֳ�һϵ�м����Ȼ�����ÿ��������ж���ֵ��
# ֱ��ͼҲ���Ա���һ������ʾ����ԡ�Ƶ�ʡ� Ȼ������ʾ�����ڼ�������е�ÿ������ռ�ȣ���߶��ܺ͵���1��
# img.ravel()�C�Ѷ�ά����ת����һά����  ��Ϊhist����ֻ֧��һά�����飨�����±�Ϊ�����ֵ꣬Ϊ�����꣩ 256 ��ʾ����������ֵΪ256����256����
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(10000)

'''
# ��ɫͼ��ֱ��ͼ���⻯
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# ��ɫͼ����⻯,��Ҫ�ֽ�ͨ�� ��ÿһ��ͨ�����⻯
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# �ϲ�ÿһ��ͨ��
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
'''




''''
plt.hist()
�����ò����⡿
x: ��ֱ��ͼ��Ҫ�õ����ݣ�������һά���飻��ά��������Ƚ��б�ƽ������ͼ����ѡ������
bins: ֱ��ͼ����������Ҫ�ֵ�������Ĭ��Ϊ10��
range��Ԫ��(tuple)��None���޳��ϴ�ͽ�С����Ⱥֵ������ȫ�ַ�Χ�����ΪNone����Ĭ��Ϊ(x.min(), x.max())����x��ķ�Χ��
density������ֵ�����Ϊtrue���򷵻ص�Ԫ��ĵ�һ������n��ΪƵ�ʶ���Ĭ�ϵ�Ƶ����
weights����x��״��ͬ��Ȩ�����飻��x�е�ÿ��Ԫ�س��Զ�ӦȨ��ֵ�ټ��������normed��densityȡֵΪTrue������Ȩ�ؽ��й�һ������������������ڻ����Ѻϲ������ݵ�ֱ��ͼ��
cumulative������ֵ�����ΪTrue��������ۼ�Ƶ�������normed��densityȡֵΪTrue��������ۼ�Ƶ�ʣ�
bottom�����飬����ֵ��None��ÿ�����ӵײ������y=0��λ�á�����Ǳ���ֵ����ÿ�����������y=0����/���µ�ƫ������ͬ����������飬���������Ԫ��ȡֵ�ƶ���Ӧ�����ӣ���ֱ��ͼ���±��˾��룻
histtype��{��bar��, ��barstacked��, ��step��, ��stepfilled��}��'bar���Ǵ�ͳ������ֱ��ͼ��'barstacked���Ƕѵ�������ֱ��ͼ��'step����δ��������ֱ��ͼ��ֻ����߿򣻡�stepfilled����������ֱ��ͼ����histtypeȡֵΪ��step����stepfilled����rwidth����ʧЧ��������ָ������֮��ļ����Ĭ��������һ��
align��{��left��, ��mid��, ��right��}����left�������ӵ�����λ��bins�����Ե����mid��������λ��bins���ұ�Ե֮�䣻��right�������ӵ�����λ��bins���ұ�Ե��
orientation��{��horizontal��, ��vertical��}�����ȡֵΪhorizontal��������ͼ����y��Ϊ���ߣ�ˮƽ���У������Ϊ����bar()ת����barh()����ת90�㣻
rwidth������ֵ��None�����ӵĿ��ռbins��ı�����
log������ֵ�����ȡֵΪTrue����������Ŀ̶�Ϊ�����̶ȣ����logΪTrue��x��һά���飬�����Ϊ0��ȡֵ�����޳��������طǿյ�(frequency, bins, patches����
color��������ɫ�����飨Ԫ��Ϊ��ɫ����None��
label���ַ��������У���None���ж�����ݼ�ʱ����label��������ע���֣�
stacked������ֵ�����ȡֵΪTrue���������ͼΪ������ݼ��ѵ��ۼƵĽ�������ȡֵΪFalse��histtype=��bar����step�����������ݼ������Ӳ������У�
normed: �Ƿ񽫵õ���ֱ��ͼ������һ��������ʾռ�ȣ�Ĭ��Ϊ0������һ�������Ƽ�ʹ�ã��������density������
edgecolor: ֱ��ͼ�߿���ɫ��
alpha: ͸���ȣ�

������ֵ�����ò������շ���ֵ�������������ݱ�ǩ����
n��ֱ��ͼ��������ÿ�������µ�ͳ��ֵ���Ƿ��һ���ɲ���normed�趨����normedȡĬ��ֵʱ��n��Ϊֱ��ͼ������Ԫ�ص�����������Ƶ������
bins: ���ظ���bin�����䷶Χ��
patches������ÿ��bin������������ݣ���һ��list��
����������plt.bar()���ơ�
'''