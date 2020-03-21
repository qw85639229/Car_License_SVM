import random
from PIL import Image
import os
import numpy as np
import cv2

root = '/home/hins/Documents/OCR_project/data/train_images/validation-set/'
save_path = '/home/hins/Documents/OCR_project/data/train_images/val/'
digits = os.listdir(root)

for digit in digits:
    files = os.listdir(root + digit)
    if not os.path.exists(save_path + digit):
        os.mkdir(save_path + digit)
    for file in files:
        img = cv2.imread(root + digit + '/' + file)
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        otsu_thres, otsu_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = Image.fromarray(otsu_img)
        img.save(save_path + digit + '/' + file)

