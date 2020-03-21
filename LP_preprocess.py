#coding=UTF-8
import cv2
import numpy as np
import os

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.equalizeHist(dst)
    #dst = cv2.GaussianBlur(dst, (1, 3), 0)
    #dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst
def cv_imwrite(write_path, img):
    cv2.imencode('.jpg', img,)[1].tofile(write_path)

if __name__ == '__main__':
    # blockSize = 16
    # save_dir='C:\\work\\r\\deep\\LPRNet_Pytorch-master\\data\\train_transform'
    # path='C:\\work\\r\\deep\\LPRNet_Pytorch-master\\data\\train'
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         img_name=file.split()
    #         imgpath=os.path.join(root,file)
    #         img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_COLOR)
    #         dst = unevenLightCompensate(img, blockSize)
    #         isExists = os.path.exists(save_dir)
    #         if not isExists:
    #             os.makedirs(save_dir)
    #         save_path=os.path.join(save_dir,file)
    #         cv_imwrite(save_path, dst)
    blockSize = 16
    img_path = 'test_0281\\1.jpg'
    img = cv2.imread(img_path)
    cv2.imshow("origin",img)
    print("shape of origin: ",img.shape)
    dst = unevenLightCompensate(img, blockSize)
    cv2.imshow("dst",dst)
    print("shape of dst: ", dst.shape)
    cv_imwrite('test_0281\\output_1.jpg', dst)
    cv2.waitKey(0)