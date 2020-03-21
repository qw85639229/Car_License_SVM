import cv2
import numpy as np

# Read and show the raw images
ImagePath = './data/crop_digits/8.jpg'
rawImage = cv2.imread(ImagePath)
rawImage = cv2.resize(rawImage, (480,270))
rawImage = cv2.rotate(rawImage, cv2.ROTATE_90_CLOCKWISE)
print('Now showing the raw images')
cv2.imshow('rawImage', rawImage)
cv2.waitKey(0)

# pad the image for close operation
pad_img = np.zeros([50, rawImage.shape[1], 3]).astype(np.uint8)
padImage = np.concatenate([pad_img, rawImage, pad_img], axis=0)
pad_img = np.zeros([padImage.shape[0], 50, 3]).astype(np.uint8)
padImage = np.concatenate([pad_img, padImage, pad_img], axis=1)
cv2.imshow('rawImage', rawImage)
cv2.waitKey(0)

# Detect the user interface according HSV
hsvImage = cv2.cvtColor(padImage, cv2.COLOR_BGR2HSV)
low_hsv = np.array([100, 50, 150])
high_hsv = np.array([160, 255, 255])
binImage = cv2.inRange(hsvImage, lowerb=low_hsv, upperb=high_hsv)
cv2.imshow("HSV Binary", binImage)
cv2.waitKey(0)

# MedianBlur filtering
MedianImage = cv2.medianBlur(binImage, 3)
cv2.imshow("MedianBlur Filtering", MedianImage)
cv2.waitKey(0)

# Integrate the target area by using close operation
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
CloseImage = cv2.morphologyEx(MedianImage, cv2.MORPH_CLOSE, kernelX)
cv2.imshow("Close Op", CloseImage)
cv2.waitKey(0)

# Eliminate the independent, small areas by using dilation and erosion
kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
DEImage = cv2.dilate(CloseImage, kernelX)
DEImage = cv2.erode(DEImage, kernelX)
DEImage = cv2.erode(DEImage, kernelY)
DEImage = cv2.dilate(DEImage, kernelY)
cv2.imshow("dilate and erode", DEImage)
cv2.waitKey(0)

print('Done!')
