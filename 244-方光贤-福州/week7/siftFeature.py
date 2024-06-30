import cv2
import numpy as np


def detect_and_draw_sift_keypoints(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return

        # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建SIFT对象
    sift = cv2.xfeatures2d.SIFT_create()

    # 检测关键点并计算描述符（在这里我们只需要关键点）
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 绘制关键点
    if keypoints is not None:
        # 使用DRAW_RICH_KEYPOINTS绘制包含大小和方向的关键点
        img_keypoints = cv2.drawKeypoints(image=img, outImage=img.copy(), keypoints=keypoints,
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                          color=(51, 163, 236))

        # 显示图像
        cv2.imshow('SIFT Keypoints', img_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No keypoints detected.")

    # 调用函数并传入图像路径


detect_and_draw_sift_keypoints("lenna.png")