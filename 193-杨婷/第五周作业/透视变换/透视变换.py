import cv2
import numpy as np

img = cv2.imread("photo1.jpg")
image = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])  # 角点检测算法
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])  # 可以根据输出图片大小确定
print(img.shape)
warp_matrix = cv2.getPerspectiveTransform(src, dst)  # 生成透视变换矩阵3*3
print("warpMatrix:")
print(warp_matrix)

result_image = cv2.warpPerspective(image, warp_matrix, (337, 488))  # (337, 488)这是输出图像的大小（宽度和高度）
# print(result_image.shape) # .shape 会返回三个元素，其中第一个元素是高度，第二个元素是宽度，第三个元素是通道数


# 鼠标事件回调函数：下面的代码可以使原图被读取出来时通过拖拽鼠标得到坐标
def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # 显示坐标
        print(f"Mouse position: ({x}, {y})")


# 创建窗口并设置鼠标回调函数
cv2.namedWindow('original image')
cv2.setMouseCallback('original image', mouse_event)
while True:
    cv2.imshow('original image', img)
    cv2.imshow('result', result_image)

    # 等待按键，按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break










