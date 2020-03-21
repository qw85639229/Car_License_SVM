import cv2
import os
import numpy as np


test_root = './data/test_chinese/'
test_img_path = os.listdir(test_root)
template_size = (100, 100)


def calculate_similarity(template, image):
    shape1 = template.shape
    shape2 = image.shape
    if shape1 != shape2:
        image = cv2.resize(image, template_size)

    template = np.reshape(template, (-1)) / 255.
    image = np.reshape(image, (-1)) / 255.

    sim = np.sum(template * image) / (np.sqrt(np.sum(template * template)) * np.sqrt(np.sum(image * image)))

    return sim


def calculate_similarity2(template, image):
    shape1 = template.shape
    shape2 = image.shape
    if shape1 != shape2:
        image = cv2.resize(image, template_size)

    template = np.reshape(template, (-1)) / 255.
    image = np.reshape(image, (-1)) / 255.

    bool_template = template.astype(np.bool)
    bool_image = image.astype(np.bool)

    imageV = np.bitwise_and(bool_template, bool_image)
    imageX = np.bitwise_xor(imageV, bool_image)
    imageW = np.bitwise_xor(imageV, bool_template)

    T = np.sum(template)
    U = np.sum(image)
    V = np.sum(imageV)
    X = np.sum(imageX)
    W = np.sum(imageW)
    TUV = (T + U + V) / 3

    sim = V / (W * X * np.sqrt(((T-TUV)**2 + (U-TUV)**2 + (V-TUV)**2)/2) / T / U)

    return sim


template_imgs = []
template_root = './data/chinese_template_process'
templates = os.listdir(template_root)
chinese_chars = [x.split('.')[0] for x in templates]

for path in test_img_path:
    raw_img = cv2.imread(os.path.join(test_root, path))

    grey_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    otsu_thres, otsu_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black = np.sum(np.where(otsu_img == 0))
    white = np.sum(np.where(otsu_img == 255))
    if black < white:
        otsu_img = 255 - otsu_img

    otsu_img_array = np.array(otsu_img)
    h_histogram = np.sum(otsu_img_array, axis=1)
    his_h_thres = np.max(h_histogram) / 50

    start, end = None, None
    for i in range(0, len(h_histogram)):
        if h_histogram[i] >= his_h_thres and start is None:
            start = i

        if h_histogram[i] < his_h_thres and start is not None:
            if i - start > len(h_histogram) / 2:
                end = i
                break
            else:
                start = None

        if i == len(h_histogram) - 1 and h_histogram[i] >= his_h_thres and start is not None:
            end = i

    cropped_otsu_img = otsu_img[start:end, :]

    v_histogram = np.sum(cropped_otsu_img, axis=0)
    his_v_thres = np.max(v_histogram) / 50
    # index = [i for i in range(int(v_histogram.shape[0]))]
    # plt.bar(index, v_histogram)
    # plt.show()
    chars = list()
    bin_v_histogram = [1 if val > his_v_thres else 0 for val in v_histogram]
    tmp = 0
    flag = 0
    for i in range(len(bin_v_histogram) - 1):
        if bin_v_histogram[i + 1] > bin_v_histogram[i] and flag == 0:
            tmp = i
            flag = 1
        if bin_v_histogram[i + 1] < bin_v_histogram[i] and i - tmp > (end - start) * 3 / 5:
            chars.append((tmp, i))
            flag = 0

    print('Totoal %d characters' % len(chars))

    char_imgs = []
    for char in chars:
        char_imgs.append(cropped_otsu_img[:, char[0]:char[1]])

    res_chars = []
    for img in char_imgs:
        char_h_histogram = np.sum(img, axis=1)
        char_his_h_thres = 1

        start_ = None
        for i in range(0, len(char_h_histogram)):
            if char_h_histogram[i] >= char_his_h_thres and start_ is None:
                start_ = i
                break

        end_ = None
        for i in range(len(char_h_histogram) - 1, -1, -1):
            if char_h_histogram[i] >= char_his_h_thres and end_ is None:
                end_ = i
                break

        cropped_img = img[start_:end_, :]

        # cv2.imshow('img', cropped_img)
        # cv2.waitKey(0)
        similarities = []
        for temp in templates:
            temp_img = cv2.imread(os.path.join(template_root, temp))
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
            otsu_temp_thres, otsu_temp_img = cv2.threshold(temp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            similarities.append(calculate_similarity2(otsu_temp_img, cropped_img))

        similarities = np.array(similarities)
        index = int(np.argmax(similarities))
        res_chars.append(chinese_chars[index])

    print(''.join(res_chars))

    cv2.imshow('raw_image', raw_img)
    cv2.waitKey(500)