from PIL import Image

#使用PIL库实现最邻近插值法

def nearest_interpolation(image, new_width, new_height):
    img=Image.open(image)  #打开图像
    resized_img = img.resize((new_width, new_height), Image.Resampling.NEAREST)  #用PIL库中的resize()方法来调整图像大小，并指定了最邻近插值算法
    resized_img.show()  #显示调整大小后的图像

# 调用函数
nearest_interpolation("lenna.png", 1000, 1000)


