
import numpy as np
import cv2
import matplotlib.pyplot as plt

from Util import cv_imread

# 动态调用01_gray的代码逻辑，因为文件名以数字开头没法直接import
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("module.name", "206-田海龙-北京/第02周/01_gray.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)


def twe_value_01(img):
    """
    二值化图像，根据G值大于128为1，否则为0
    """

    h,w=img.shape[:2]

    img_twe_value = np.zeros([h,w],np.uint8)

    # print(img_twe_value.dtype)

    for i in range(h):
        for j in range(w):
            # BGR
            m=img[i,j][0]
            # print(m)
            # print(img_twe_value[i,j])
            if m>128:
                img_twe_value[i,j]=1
            else:
                img_twe_value[i,j]=0

    return img_twe_value

def twe_value_02(img):
    """
    二值化图像，根据灰度值大于0.5为1，否则为0
    """

    # 获取方法并传递参数，调用了01_gray的方法
    method_to_call = getattr(module, 'img_gray_04')  # 获取方法
    # 调用方法并传递参数
    img_gray=  method_to_call(img)  

    img_twe_value = np.where(img_gray>0.5,1,0)

    return img_twe_value

img_path="206-田海龙-北京/第02周/img/lenna.png"
img=cv_imread(img_path)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img)

img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

img_twe_value=twe_value_01(img)
print(img_twe_value)
plt.subplot(222)
plt.imshow(img_twe_value,cmap='gray')

img_twe_value=twe_value_02(img)
print(img_twe_value)
plt.subplot(223)
plt.imshow(img_twe_value,cmap='gray')

plt.show()