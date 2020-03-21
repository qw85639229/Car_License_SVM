#coding=UTF-8

import numpy as np


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Line(object):  # 直线由两个点组成
    def __init__(self, p1=Point(0, 0), p2=Point(2, 2)):
        self.p1 = p1
        self.p2 = p2

    def distance_point_to_line(self, current_line, mainline):
        angle = self.get_cross_angle(current_line, mainline)
        sin_value = np.sin(angle * np.pi / 180)  # 其中current_line视为斜边
        long_edge = math.sqrt(  # 获取斜边长度
            math.pow(current_line.p2.x - current_line.p1.x, 2) + math.pow(current_line.p2.y - current_line.p1.y,
                                                                          2))  # 斜边长度
        distance = long_edge * sin_value
        return distance

    def get_cross_angle(self, l1, l2):
        arr_a = np.array([(l1.p2.x - l1.p1.x), (l1.p2.y - l1.p1.y)])  # 向量a
        arr_b = np.array([(l2.p2.x - l2.p1.x), (l2.p2.y - l2.p1.y)])  # 向量b
        cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))  # 注意转成浮点数运算
        return np.arccos(cos_value) * (180 / np.pi)  # 两个向量的夹角的角度， 余弦值：cos_value, np.cos(para), 其中para是弧度，不是角度

    def get_main_line(self, mask):
        # 获取最上面和最下面的contour的质心
        contour_list = get_contours(mask)
        c7_x, c7_y = get_centroid(contour_list[7])  # 最上面
        c0_x, c0_y = get_centroid(contour_list[0])  # 最下面

        # 获取串联两个质心，得到主线
        point1 = Point(c7_x, c7_y)
        point2 = Point(c0_x, c0_y)
        mainline = Line(point1, point2)
        return mainline

    #  求seg_img图中的直线与垂直方向的夹角
    def mainline_inclination_angle(self, seg_img):
        # 获取串联两个质心，得到主线
        mainline = self.get_main_line(seg_img)
        # 测试该函数，三角形边长：3,4,5
        mainline.p1.x = 0  # 列
        mainline.p1.y = 0
        mainline.p2.x = 3
        mainline.p2.y = 4
        # 获取参考线，这里用的是垂直方向的直线
        # base_line = Line.get_main_line(normal_mask)
        base_line = Line(Point(mainline.p1.x, mainline.p1.y),
                         Point(mainline.p1.x, mainline.p2.y))  # 同一列mainline.p1.x，行数随便
        # 获取两条线的夹角
        angle = mainline.get_cross_angle(mainline, base_line)
        return angle


# 求最大连通域的中心点坐标
def get_centroid(contour):
    moment = cv2.moments(contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy  # col, row
    else:
        return None


# 过滤面积较小的contour, 不同于后面的get_cnts(),
def get_contours(seg_img, area_thresh=0):
    if len(seg_img.shape) == 3:
        image_gray = cv2.cvtColor(seg_img, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = seg_img.copy()

    # 1, 二值化原图的灰度图,然后求解轮廓contours
    _, mask_contour = cv2.threshold(image_gray.copy(), 0.1, 255, cv2.THRESH_BINARY)
    # 2, 找二值化后图像mask中的contour
    contours, hierarchy = cv2.findContours(image=mask_contour.copy(), mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
    # 3, 遍历轮廓,将轮廓面积大于阈值的保存到contour_list中
    contour_list = []
    if len(contours) == 0:
        return contour_list
    for c_num in range(len(contours)):
        area = cv2.contourArea(contours[c_num])
        if area > area_thresh:
            contour_list.append(contours[c_num])
        else:
            continue
    return contour_list


def get_bbox_num(seg_img, area_thresh=0):
    image = seg_img.copy()
    contour_list = get_contours(image, area_thresh)  # contours排序是从上到下
    return len(contour_list)

a1 = Point(0,0)
print("type of a1", type(a1))
print("data of a1", a1)
a2 = Point(2,2)
b1 = Point(0,0)
b2 = Point(2,0)

line1 = Line(a1,a2)
line2 = Line(b1,b2)
angle = line1.get_cross_angle(line1,line2)
print(angle)