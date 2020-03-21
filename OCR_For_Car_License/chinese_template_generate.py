import cv2
import os
import numpy as np

template_path = './data/Chinese_Template'
templates = os.listdir(template_path)
chinese_chars = [x.split('.')[0] for x in templates]

template_imgs = []
for c in templates:
    path = os.path.join(template_path, c)
    template_imgs.append(cv2.imread(path))

process_template_imgs = []
for j, img in enumerate(template_imgs):
    # cv2.imshow('ori_img', img)
    # cv2.waitKey(0)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu_thres, otsu_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black = np.sum(np.where(otsu_img == 0))
    white = np.sum(np.where(otsu_img == 255))
    if black < white:
        otsu_img = 255 - otsu_img

    otsu_img_array = np.array(otsu_img)
    h_histogram = np.sum(otsu_img_array, axis=1)
    his_h_thres = 1

    start = None
    for i in range(0, len(h_histogram)):
        if h_histogram[i] >= his_h_thres and start is None:
            start = i
            break

    end = None
    for i in range(len(h_histogram)-1, -1, -1):
        if h_histogram[i] >= his_h_thres and end is None:
            end = i
            break

    cropped_otsu_img = otsu_img[start:end, :]

    v_histogram = np.sum(cropped_otsu_img, axis=0)
    his_v_thres = 1

    start = None
    for i in range(0, len(v_histogram)):
        if v_histogram[i] >= his_v_thres and start is None:
            start = i
            break

    end = None
    for i in range(len(v_histogram) - 1, -1, -1):
        if v_histogram[i] >= his_v_thres and end is None:
            end = i
            break

    cropped_otsu_img = cropped_otsu_img[:, start:end]

    # cv2.imshow('otsu', cropped_otsu_img)
    # cv2.waitKey(0)

    cropped_otsu_img = cv2.resize(cropped_otsu_img, (100, 100))
    cv2.imwrite('./data/chinese_template_process/%s.jpg' % chinese_chars[j], cropped_otsu_img)
