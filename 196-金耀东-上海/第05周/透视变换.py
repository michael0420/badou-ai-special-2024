import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import MultiCursor
from matplotlib.patches import Polygon

class global_param:
    points_num = None
    src_points = None
    dst_points = None
    maxWidth = None
    maxHeight = None
    def init_param(self):
        self.points_num = 4
        self.src_points = []
        self.dst_points = []
        self.maxHeight = 0
        self.maxWidth = 0

def display_img(winname, img):
    plt.title(winname)
    plt.imshow(img)
    plt.show()

# 以几何中心点为原点建立坐标系根据每个顶点与横坐标的夹角的大小按顺时针调整点的位置，第一个点在第四象限
def adjust_pts_order(pts_2ds):
    # 计算几何中心点的坐标
    cen_x, cen_y = np.mean(pts_2ds, axis=0)
    print(f"几何中心:{(cen_x,cen_y)}")
    plt.plot(cen_x, cen_y, marker='o', markerfacecolor='blue')

    d2s = []
    for i in range(len(pts_2ds)):
        o_x = pts_2ds[i][0] - cen_x
        o_y = pts_2ds[i][1] - cen_y
        atan2 = np.arctan2(o_y, o_x)
        if atan2 < 0:
            atan2 += np.pi * 2
        d2s.append([pts_2ds[i], atan2])
    d2s = sorted(d2s, key=lambda x:x[1])
    order_2ds = np.array([x[0] for x in d2s])
    return order_2ds

# 通过点击鼠标获取4个原始点的坐标
def get_src_points(img):
    # 新建画布与坐标轴
    fig, ax = plt.subplots()

    # 绘制图片
    ax.imshow(img)

    # 创建多光标实例
    mc = MultiCursor(fig.canvas, [ax], horizOn=True, color='r', lw=1)

    # 定义获取坐标的回调函数
    def on_click(event):
        if event.inaxes is None or global_param.points_num < 0:
            return
        if event.button == 1:
            global_param.points_num -= 1
            if global_param.points_num >= 0:
                global_param.src_points.append((event.xdata, event.ydata))
                print("获取点坐标: (%.2f, %.2f)" % (event.xdata, event.ydata))

                # 在图像上标记原始点
                ax.plot(event.xdata, event.ydata, marker='o', markerfacecolor='red')

            if global_param.points_num == 0:
                # 调整原始点的顺序
                global_param.src_points = np.float32( adjust_pts_order(global_param.src_points) )

                # 绘制多边形
                polygon = Polygon(global_param.src_points, facecolor='green', edgecolor='black', alpha=0.5)
                ax.add_patch(polygon)

                # 等待3秒后关闭画布，开始下一步计算
                plt.title(u"waite for 3 seconds...")
                plt.pause(3)
                plt.close()

    # 连接点击事件和回调函数
    fig.canvas.mpl_connect('button_press_event', on_click)

    # 显示图像
    plt.show()

# 根据点4个原始坐标计算目标点的坐标
def get_dst_points(points):
    width_01 = np.sqrt(((points[0][0] - points[1][0]) ** 2) + ((points[0][1] - points[1][1]) ** 2))
    width_23 = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
    global_param.maxWidth = int(max(width_01, width_23))

    height_03 = np.sqrt(((points[0][0] - points[3][0]) ** 2) + ((points[0][1] - points[3][1]) ** 2))
    height_12 = np.sqrt(((points[1][0] - points[2][0]) ** 2) + ((points[1][1] - points[2][1]) ** 2))
    global_param.maxHeight = int(max(height_03, height_12))

    global_param.dst_points = np.float32([
        [global_param.maxWidth - 1, global_param.maxHeight - 1],
        [0, global_param.maxHeight - 1],
        [0, 0],
        [global_param.maxWidth - 1, 0]
    ])

if __name__ == "__main__":
    # 获取图像文件路径并读取图像
    img_dir = "img"
    img_filename = "photo3.png"
    img_path = os.path.join(img_dir, img_filename)
    img = plt.imread(img_path)

    # 初始化参数
    global_param.init_param(global_param)

    # 获取原始点坐标
    get_src_points(img)

    # 获取目标点坐标
    get_dst_points(global_param.src_points)

    # 计算透视矩阵
    transform_mat = cv2.getPerspectiveTransform(global_param.src_points, global_param.dst_points)

    # 进行透视变换
    out = cv2.warpPerspective(img, transform_mat, (global_param.maxWidth, global_param.maxHeight))

    # 输出结果
    display_img("result", out)


