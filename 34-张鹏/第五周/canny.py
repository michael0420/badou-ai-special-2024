import cv2

def canny_edge_detection(image, low_threshold, high_threshold):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯滤波器进行模糊处理
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用Canny边缘检测算法
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    
    return edges

if __name__ == "__main__":
    # 读取图像
    image = cv2.imread('test.png')

    # 设置Canny边缘检测算法的阈值
    low_threshold = 50
    high_threshold = 150

    # 调用Canny边缘检测函数
    edges = canny_edge_detection(image, low_threshold, high_threshold)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Canny Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
