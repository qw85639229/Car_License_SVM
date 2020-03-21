import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------- load the raw image -----------------
img_path = './data/10.jpg'
img = cv2.imread(img_path)
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('raw_image', grey_img)
cv2.waitKey(0)


# -------- binarizing images with Otsu method --------
otsu_thres, otsu_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('otsu', otsu_img)
cv2.waitKey(0)

black = np.sum(np.where(otsu_img == 0))
white = np.sum(np.where(otsu_img == 255))
if black < white:
    otsu_img = 255 - otsu_img

# ------- crop the area on both top and bottom -------
otsu_img_array = np.array(otsu_img)
h_histogram = np.sum(otsu_img_array, axis=1)
his_h_thres = np.max(h_histogram) / 50
# index = [i for i in range(int(h_histogram.shape[0]))]
# plt.bar(index, h_histogram)
# plt.show()
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

    if i == len(h_histogram)-1 and h_histogram[i] >= his_h_thres and start is not None:
        end = i


cropped_otsu_img = otsu_img[start:end, :]

# pad the top and bottom of the cropped image with zeros
cropped_otsu_img_width = cropped_otsu_img.shape[1]
pad_img = np.zeros([int(cropped_otsu_img.shape[0]/20), cropped_otsu_img_width])
cropped_otsu_img = np.concatenate([pad_img, cropped_otsu_img, pad_img], axis=0)
cv2.imshow('cropped_otsu_img', cropped_otsu_img)
cv2.waitKey(0)


# ----------------- separate digits ------------------
v_histogram = np.sum(cropped_otsu_img, axis=0)
his_v_thres = np.max(v_histogram) / 20
# index = [i for i in range(int(v_histogram.shape[0]))]
# plt.bar(index, v_histogram)
# plt.show()
digits = list()
bin_v_histogram = [1 if val>his_v_thres else 0 for val in v_histogram]
tmp = 0
for i in range(len(bin_v_histogram)-1):
    if bin_v_histogram[i+1] > bin_v_histogram[i]:
        tmp = i
    if bin_v_histogram[i+1] < bin_v_histogram[i]:
        digits.append((tmp, i))

print(len(digits))
# padding the digits to 3:4
digit_imgs = []
for digit in digits:
    digit_imgs.append(cropped_otsu_img[:, digit[0]:digit[1]])

height = np.array(cropped_otsu_img).shape[0]
target_width = int(height)
for i, digit in enumerate(digits):
    cur_width = digit[1] - digit[0]
    if cur_width >= target_width:
        continue
    else:
        pad = int((target_width - cur_width + 1) / 2)
        pad_zeros = np.zeros([height, pad])
        digit_imgs[i] = np.concatenate([pad_zeros, digit_imgs[i], pad_zeros], axis=1)


for i in range(len(digit_imgs)):
    plt.subplot(1, len(digit_imgs), i+1)
    plt.imshow(digit_imgs[i])
    plt.title('digit_%d' % (i+1))
    plt.xticks([])
    plt.yticks([])

plt.show()

