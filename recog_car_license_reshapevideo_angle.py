#coding=UTF-8
import argparse
import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from OCR_For_Car_License.cnn import ConvNet_Car_License
from detect_car_license_angle import detect_car_license
# import caffe
# from numpy.linalg import norm
# import sys
# import json
from predict_version_1_onlyforBLUE import CardPredictor

# SZ = 20          #训练图片长宽
# MAX_WIDTH = 1000 #原始图片最大宽度
# Min_Area = 2000  #车牌区域允许最大面积
# PROVINCE_START = 1000

logging.basicConfig(format='%(asctime)s %(levelname)s:%(lineno)s] %(message)s',
                    datefmt='%H:%M:%S', level=logging.INFO)


def diff(name, t_tensor, c_tensor):
    if t_tensor.shape != c_tensor.shape:
        logging.warning('t_tensor: {:}, c_tensor: {:}'.format(t_tensor.shape, c_tensor.shape))
        t_tensor = t_tensor.reshape(c_tensor.data.shape)
    # logging.info('{:s} sum diff: {:}'.format(name, np.abs(t_tensor.data.cpu().numpy() - c_tensor.data).sum()))
    logging.info('{:s} var diff: {:.8f}'.format(name, (t_tensor.data.cpu().numpy() - c_tensor.data).var()))



def parse_auguments():
    parser = argparse.ArgumentParser(description='RetinaPL')
    # 23 good
    parser.add_argument('--img_path', type=str, default='Pytorch_Retina_License_Plate/test_images')
    parser.add_argument('--height', default=60, type=int, help='image height after resizing')
    parser.add_argument('--num_blocks', default=7, type=int, help='block number for otsu')
    parser.add_argument('--weights', type=str, default='./OCR_For_Car_License/saved_ckpts/ConvNet_model_best.pth.tar')
    args = parser.parse_args()
    return args


def char_sep(img, args):
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('img')
    resize_img = cv2.resize(img, (int(img.shape[1] * 60 / img.shape[0] / args.num_blocks) * args.num_blocks, 60))
    grey_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img', grey_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('img')
    grey_splited_imgs = np.hsplit(grey_img, args.num_blocks)
    otsu_img = []
    for splited_img in grey_splited_imgs:
        tmp_otsu_thres, tmp_otsu_img = cv2.threshold(splited_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_img.append(tmp_otsu_img)

    otsu_img = np.hstack(otsu_img)

    black = np.sum(np.where(otsu_img == 0))
    white = np.sum(np.where(otsu_img == 255))
    if black < white:
        otsu_img = 255 - otsu_img

    # cv2.imshow('otsu_img', otsu_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('otsu_img')

    otsu_img_array = np.array(otsu_img)

    right_h_histogram = np.sum(otsu_img_array[:, :int(otsu_img_array.shape[1] / 4)], axis=1)
    right_his_h_thres = np.max(right_h_histogram) / 5
    # h_histogram = np.where(h_histogram > his_h_thres, 1, 0)
    # print(h_histogram)
    right_v_start, right_v_end = None, None
    for i in range(0, len(right_h_histogram)):
        if right_h_histogram[i] >= right_his_h_thres and right_v_start is None:
            right_v_start = i
        if right_h_histogram[i] < right_his_h_thres and right_v_start is not None:
            if i - right_v_start > len(right_h_histogram) / 2.5:
                right_v_end = i
                break
            else:
                right_v_start = None

        if i == len(right_h_histogram) - 1 and right_h_histogram[i] >= right_his_h_thres and right_v_start is not None:
            right_v_end = i

    left_h_histogram = np.sum(otsu_img_array[:, -int(otsu_img_array.shape[1] / 4):], axis=1)
    left_his_h_thres = np.max(right_h_histogram) / 5
    # h_histogram = np.where(h_histogram > his_h_thres, 1, 0)
    # print(h_histogram)
    left_v_start, left_v_end = None, None
    for i in range(0, len(left_h_histogram)):
        if left_h_histogram[i] >= left_his_h_thres and left_v_start is None:
            left_v_start = i
        if left_h_histogram[i] < left_his_h_thres and left_v_start is not None:
            if i - left_v_start > len(left_h_histogram) / 2.5:
                left_v_end = i
                break
            else:
                left_v_start = None

        if i == len(left_h_histogram) - 1 and left_h_histogram[i] >= left_his_h_thres and left_v_start is not None:
            left_v_end = i

    if left_v_start is None or right_v_start is None:
        return []

    v_start = int((left_v_start + right_v_start) / 2)
    v_end = int((left_v_end + right_v_end) / 2)
    cropped_otsu_img = otsu_img[v_start:v_end, :]

    v_start = max(v_start - int(cropped_otsu_img.shape[0] / 8), 0)
    v_end = min(v_end + int(cropped_otsu_img.shape[0] / 8), otsu_img_array.shape[1] - 1)
    cropped_ori_img = resize_img[v_start:v_end, :]

    # avoid character concatenation

    if np.sum(cropped_otsu_img) / 255 / (cropped_otsu_img.shape[0] * cropped_otsu_img.shape[1]) > 0.54:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cropped_otsu_img = cv2.erode(cropped_otsu_img, kernel)

    # cv2.imshow('cropped_img', cropped_otsu_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('cropped_img')

    v_histogram = np.sum(cropped_otsu_img, axis=0)
    his_v_thres = np.max(v_histogram) / 15
    chars = list()
    bin_v_histogram = [1 if val > his_v_thres else 0 for val in v_histogram]
    start_exists = False
    tmp = 0
    i = 0
    while i < len(bin_v_histogram) - 1:
        if bin_v_histogram[i + 1] > bin_v_histogram[i]:
            tmp = i
            start_exists = True

        if bin_v_histogram[i + 1] < bin_v_histogram[i] and (i - tmp) > (len(bin_v_histogram) / 50) and \
                np.sum(cropped_otsu_img[:, tmp:i]) > 100 * 255:
            if i - tmp < len(bin_v_histogram) / 10:
                c_start = int(max(tmp - (len(bin_v_histogram) / 9 - i + tmp) / 2, 0))
                c_end = int(min(i + (len(bin_v_histogram) / 9 - i + tmp) / 2, len(bin_v_histogram) - 1))
                chars.append((c_start, c_end))
            else:
                chars.append((tmp, i))

                start_exists = False

        if i == len(bin_v_histogram) - 2 and bin_v_histogram[i] == 1 and start_exists is True and \
                (i - tmp) > (len(bin_v_histogram) / 50) and \
                np.sum(cropped_otsu_img[:, tmp:i]) > 100 * 255:
            chars.append((tmp, i))
            start_exists = False

        i += 1

    char_imgs = []
    for c in chars:
        char_imgs.append(cropped_ori_img[:, c[0]:c[1]])

    # for i, c in enumerate(char_imgs):
    #     cv2.imshow('char%d'% i, c)
    #     cv2.waitKey(0)

    return char_imgs

#图片缩小及填充
def reshape_pic(img,times):
    h, w = img.shape[:2]
    origin_size = (w, h)
    target_size = (int(w // times), int(h // times))
    #缩小固定倍率
    test_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    #
    border__half_h = int((h - h // times) // 2)
    border_half_w = int((w - w // times) // 2)
    # 添加边界，边界值为[128,128,128]
    test_img = cv2.copyMakeBorder(test_img, border__half_h, border__half_h, border_half_w, border_half_w,
                                  cv2.BORDER_CONSTANT, value=[128, 128, 128])
    # 为避免个别像素未对齐，重新再resize回去
    output_img = cv2.resize(test_img, origin_size, interpolation=cv2.INTER_LINEAR)
    return output_img

#统计出现最多的字
def max_count(lt):
    # 定义一个字典，用于存放元素及出现的次数
    d = {}
    # 记录最大的次数的元素
    max_key = None
    # 遍历列表，统计每个元素出现的次数，然后保存到字典中
    for i in lt:
        if i not in d:
            # 计算元素出现的次数
            count = lt.count(i)
            # 保存到字典中
            d[i] = count
            # 记录次数最大的元素
            if count > d.get(max_key, 0):
                max_key = i
    return max_key

if __name__ == '__main__':

    args = parse_auguments()
    c = CardPredictor()
    c.train_svm()


    save_dir = ''

    font = ImageFont.truetype('simhei.ttf', 24)
    #add the videoCapture

    video_path = 'testvideo'

    file = 'IMG_0278.MOV'
    plate_path = file[0:-4]
    plate_number = 0
    file = os.path.join(video_path, file)

    cap = cv2.VideoCapture(file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    outputfile_dir = os.path.join(save_dir, os.path.basename(file))
    outputfile_dir = outputfile_dir[:-3]
    outputfile_dir = outputfile_dir+"avi"
    videoWriter = cv2.VideoWriter(outputfile_dir, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)
    ret, frame = cap.read()

    # 添加字典功能
    record_count = 0
    noresult_count = 0
    record = []
    record_re = []
    final_result = []
    text = ' counting'
    for i in range(7):
        record_re.append([])

    while (ret):
        plate_number += 1
        # 展示一帧
        #cv2.imshow("capture", frame)

        #添加原图缩小及填充操作以定位离摄像头太近的车牌视频（实际使用应该无需此步骤）
        times = 2.4 #测试缩小固定倍数以识

        frame = reshape_pic(frame,times)
        car_licenses, license_boxes, license_angle1, license_angle2 = detect_car_license(frame)

        #判断是否能检测到车牌，当检测不到车牌累计一定次数之后刷新字典
        if len(license_boxes) == 0:
            noresult_count += 1
            if noresult_count >= 40:
                text = ' no result'
                # 刷新字典及计数
                record_count = 0
                record = []
                record_re = []
                final_result = []
                for i in range(7):
                    record_re.append([])

        raw_img = frame
        for img, b in zip(car_licenses, license_boxes):
            r, roi, color = c.predict(img)
            if len(r) >= 7:
                save_plate_path = os.path.join(plate_path, str(plate_number))
                save_plate_path = save_plate_path + '.jpg'
                #cv2.imwrite(save_plate_path,img)
                #计数操作及将检测到的车牌结果计入字典
                record_count += 1
                noresult_count = 0
                record.append(r)
                if record_count >= 25: #字典累计次数
                    # 重排序字典
                    for i in range(7):
                        for j in range(len(record)):
                            record_re[i].append(record[j][i])
                    #找到车牌每个位置出现最多的情况
                    for i in range(7):
                        single_result = max_count(record_re[i])
                        final_result.append(single_result)
                    #print("final_result: ", final_result)
                    text1 = ''.join(final_result)
                    text = text1
                    #刷新字典及计数
                    record_count = 0
                    record = []
                    record_re = []
                    final_result = []
                    for i in range(7):
                        record_re.append([])
                # else:
                    # if 'test1' in dir():
                    #     text = test1
                    # else:
                    #     text = ' counting'
            else:
                #判断累计测不到结果后清空字典
                noresult_count += 1
                if noresult_count >=40:
                    text = ' no result'
                    # 刷新字典及计数
                    record_count = 0
                    record = []
                    record_re = []
                    final_result = []
                    for i in range(7):
                        record_re.append([])

            text_angle = text + "  " + str(license_angle1)[:4] + "  " + str(license_angle2)[:4]
            raw_img = cv2.rectangle(raw_img, (b[0], b[1]), (b[2], b[3]), color=(0, 0, 255), thickness=3,
                                    lineType=cv2.LINE_AA)
            cx = b[0]
            cy = b[1] - 24

            img_pil = Image.fromarray(raw_img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((cx, cy), text_angle, font=font, fill=(0, 0, 255, 0))
            raw_img = np.array(img_pil)

        videoWriter.write(raw_img)
        cv2.waitKey(fps)
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


