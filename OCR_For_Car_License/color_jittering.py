import cv2
import random
import numpy as np

img = cv2.imread('./data/522.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv_img)
# H = hsv_img[:,:,0]
# S = hsv_img[:,:,1]
# V = hsv_img[:,:,2]

for i in range(10):
    H_ = (H + 255 * random.random()) % 256
    H_ = H_.astype(np.uint8)
    S_ = S * random.random()
    S_ = S_.astype(np.uint8)
    V_ = V * (1-0.5*random.random())
    V_ = V_.astype(np.uint8)
    new_img = np.stack([H_,S_,V_], axis=2)
    new_bgr = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    if random.random() > 0.5:
        new_bgr = 255 - new_bgr
    cv2.imshow('new_img%d' % i, new_bgr)
    cv2.waitKey(0)

