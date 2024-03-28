import cv2
import numpy


class Nearest_interpolation:

    @staticmethod
    def multiple():
        while True:
            try:
                mul = float(input("请输入图片要扩大的倍数："))
                if mul > 0:
                    # 原基础上扩大 所以+1
                    return mul + 1
                else:
                    print("请输入大于零的扩展倍数！")
            except ValueError:
                print("您的输入有误，请重新输入！")

    def nearest_interpolation(self):
        try:
            mul = float(self.multiple())
            # 读取原图
            img = cv2.imread("lenna.png")
            # 取出 像素大小 和通道
            h, w, channels = img.shape
            # 扩大的倍数
            nh = int(h * mul)
            nw = int(w * mul)
            # 创建一个扩大后的空图片
            new_img = numpy.zeros((nh, nw, channels), numpy.uint8)
            # 遍历每个像素，最近临界插值
            for i in range(nh):
                for j in range(nw):
                    x = int(i / mul)
                    y = int(j / mul)
                    new_img[i, j] = img[x, y]
            cv2.imshow("img", new_img)
            cv2.waitKey(0)
        except ValueError:
            print("无效扩展！")


nearest = Nearest_interpolation()
nearest.nearest_interpolation()
