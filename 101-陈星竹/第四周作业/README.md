

# 高斯噪声

mean:均值
std_dev:标准差
 np.random.normal 函数用于生成符合指定均值和标准差的高斯分布随机数，然后通过加法将噪声叠加到原始图像中。
````
def add_auessian_nosie(img,mean,std_dev):
    h,w = img.shape
    gauss = np.random.normal(mean,std_dev,(h,w))
    #矩阵相加，限制范围在0-255
    noisy_img = np.clip(img+gauss,0,255)
    # 保证图像数据的取值范围在0-255之间
    return noisy_img.astype(np.uint8)
````
# 椒盐噪声：
随机出现的黑白点
density：椒盐噪声的密度,范围在[0,1]
````
def add_peppersalt_noise(img,density):
    noisy_img = np.copy(img)
    h,w = img.shape
    #得到要加噪声的像素数
    num_noise_pixels = int(density * h * w)

    #添加噪声
    for i in range(num_noise_pixels):
        #取范围内随机坐标
        x = random.randint(0,w-1)
        y = random.randint(0,h-1)
        #给对应像素点添加黑/白点噪声
        noisy_img[y,x] = random.choice([0,255])
    return noisy_img

img = cv2.imread('lenna.png',0)
gauss_img = add_auessian_nosie(img,0,20)
salt_img = add_peppersalt_noise(img,0.02)
````
# 中值滤波
消除椒盐噪声，抑制效果好，基本保留图像清晰度，对高斯噪声抑制效果一般
ksize:滤波器大小
cv2.medianBlur(图像,ksize=)

均值滤波：cv2.blur(img,(ksize,ksize))

最大值滤波:cv2.dilate(image, None, iterations=1)
None:表示使用默认的矩形内核
iterations：表示迭代次数 
````
filltered_salt_img = cv2.medianBlur(salt_img,ksize=3)
filltered_gauss = cv2.medianBlur(gauss_img,ksize=3)
````

# 图像增强:
有目的地强调图像的整体 /局部特性。
## 一、点处理
### 1. 线性变换:调整图像对比度和亮度，a>1增加对比度，0<a<1减小对比度，a=0并且b≠0,b>0图像变亮，b<0图像变暗
````
cv2.convertScaleAbs(img,alpha=a,beta=b) y=a*x+b
````
### 2.分段线性变换：对于某个区域的对比度进行增大或者减小
````
#定义分段线性变换的参数
thresholds = [100, 150, 200]  # 分段阈值
slopes = [1.0, 1.5, 2.0]       # 斜率
intercepts = [0, -75, -150]    # 截距
#应用分段线性变换
transformed_image = piecewise_linear_transform(image, thresholds, slopes, intercepts)
````
### 3.对数变换：扩展图像的低灰度值部分，压缩高灰度值部分，强调灰度部分
````
def log_transform(image, c=1):
    # 对数变换公式
    transformed_image = c * np.log(1 + image)

    # 将像素值缩放到合适的范围（0-255）
    transformed_image = np.clip(transformed_image, 0, 255)
    transformed_image = transformed_image.astype(np.uint8)

    return transformed_image
````
### 4.幂律变换/伽马变换：
用于图像的矫正，对漂白挥着过黑的图片进行修正，γ＞1，增强对比度，0 < γ < 1，降低对比度
````
def power_law_transform(image, gamma=1.0, c=1.0):
    # 对输入图像进行幂律变换
    transformed_image = c * np.power(image, gamma)

    # 将像素值缩放到合适的范围（0-255）
    transformed_image = np.clip(transformed_image, 0, 255)
    transformed_image = transformed_image.astype(np.uint8)
````
    return transformed_image

## 二、领域处理
### 1.直方图均衡化
### 2.图像滤波

# 主成分分析（Principal Component Analysis，PCA）

## 简介

主成分分析（PCA）是一种常用的降维技术，用于将高维数据集转换为低维子空间，同时保留数据集的最重要特征。它通过线性变换将数据投影到新的坐标轴上，以便最大化数据的方差。

## PCA的步骤

1. **标准化数据**：将数据集的特征标准化，使得每个特征的均值为 0，标准差为 1。

2. **计算协方差矩阵**：计算标准化后数据的协方差矩阵。

3. **计算特征值和特征向量**：对协方差矩阵进行特征分解，得到特征值和对应的特征向量。

4. **选择主成分**：根据特征值的大小选择前 k 个特征向量作为主成分，其中 k 是降维后的维度。

5. **投影数据**：将原始数据投影到所选的主成分上，得到降维后的数据集。

## PCA的应用

1. **数据可视化**：通过降维将高维数据可视化到二维或三维空间，便于人类理解和观察。

2. **特征提取**：在机器学习任务中，可以使用PCA来提取最重要的特征，以减少模型训练的计算成本和降低过拟合的风险。

3. **去噪**：在某些情况下，PCA可以用于去除数据中的噪音，从而提高模型的性能。

## 示例代码

```python
import numpy as np
from sklearn.decomposition import PCA

# 创建一个随机数据集
X = np.random.rand(100, 10)

# 使用PCA进行降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("降维后的数据集：")
print(X_pca)

