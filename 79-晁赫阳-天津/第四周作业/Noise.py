import cv2
import matplotlib.pyplot as plt
import random
from skimage import util
def Guass_Noise(src, means, sigma, percentage):
    channels = src.shape[-1] if len(src.shape) == 3 else 1
    NoiseImg = src.copy()  # 创建一个图像副本，以免修改原始图像
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        RandX = random.randint(0, src.shape[0] - 1)
        RandY = random.randint(0, src.shape[1] - 1)
        if channels == 1:
            NoiseImg[RandX, RandY] += random.gauss(means, sigma)
            NoiseImg[RandX, RandY] = max(0, min(255, NoiseImg[RandX, RandY]))  # 限制像素值在0到255之间
        elif channels == 3:
            for j in range(channels):
                NoiseImg[RandX, RandY, j] += random.gauss(means, sigma)
                NoiseImg[RandX, RandY, j] = max(0, min(255, NoiseImg[RandX, RandY, j]))  # 限制像素值在0到255之间
    return NoiseImg

def PepperSalt_Noise(src, percentage):
    channels = src.shape[-1] if len(src.shape) == 3 else 1
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    NoiseImg = src
    for i in range(NoiseNum):
        RandX = random.randint(0, src.shape[0] - 1)
        RandY = random.randint(0, src.shape[1] - 1)
        if channels == 1:
            if random.random() <= 0.5:
                NoiseImg[RandX, RandY] = 0
            else:
                NoiseImg[RandX, RandY] = 255
        elif channels == 3:
            for j in range(channels):
                if random.random() <= 0.5:
                    NoiseImg[RandX, RandY, j] = 0
                else:
                    NoiseImg[RandX, RandY, j] = 255
    return NoiseImg
def display_images(img_dict):
    num_images = len(img_dict)
    if num_images == 0:
        print("没有图像可显示")
        return

    # 计算行数，每行最多三张图像
    rows = (num_images + 2) // 3
    cols = min(3, num_images)  # 最多三列

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows * 5))
    if num_images == 1:
        axes = [[axes]]  # 单图像处理为二维列表
    elif rows == 1:
        axes = [axes]

    # 用items()遍历字典，i是索引，(name, img)是名称和图像对
    for i, (name, img) in enumerate(img_dict.items()):
        row = i // 3
        col = i % 3
        ax = axes[row][col]
        channels = (img.shape)
        if channels == 3:
            ax.imshow(img)  # 如果是彩色图像，移除cmap参数
            ax.axis('off')
            ax.set_title(name)  # 设置图像标题为图像名称
        else:
            ax.imshow(img, cmap='gray')  # 如果是彩色图像，移除cmap参数
            ax.axis('off')
            ax.set_title(name)  # 设置图像标题为图像名称
    plt.tight_layout()
    plt.show()

if __name__ =='__main__':
    img1 = cv2.imread('0216.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = Guass_Noise(img1, 2, 4, 0.8)
    img3 = PepperSalt_Noise(img1, 0.2)
    img3 = util.random_noise(img1, mode='s&p', salt_vs_pepper= 0.2)
    img_dict = {
        'Original Image': img1,
        'Guass_Noise': img2,
        'PepperSalt_Noise': img3
    }
    display_images(img_dict)

