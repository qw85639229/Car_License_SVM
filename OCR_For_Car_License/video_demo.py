import cv2
import argparse
from cnn import ConvNet
from PIL import Image
import torch
import numpy as np


# ------------------ setting ---------------------
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-']

parser = argparse.ArgumentParser('This is a demo for image')
parser.add_argument('--weights', type=str)
parser.add_argument('--video_path', type=str)
parser.add_argument('--location', nargs='+', type=int)
args = parser.parse_args()

def digit_sep(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu_thres, otsu_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    otsu_img_array = np.array(otsu_img)
    h_histogram = np.sum(otsu_img_array, axis=1)
    his_h_thres = np.max(h_histogram) / 10
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
    cropped_otsu_img_width = cropped_otsu_img.shape[1]
    pad_img = np.zeros([int(cropped_otsu_img.shape[0] / 5), cropped_otsu_img_width])
    cropped_otsu_img = np.concatenate([pad_img, cropped_otsu_img, pad_img], axis=0)

    v_histogram = np.sum(cropped_otsu_img, axis=0)
    his_v_thres = np.max(v_histogram) / 15
    digits = list()
    bin_v_histogram = [1 if val > his_v_thres else 0 for val in v_histogram]
    tmp = 0
    for i in range(len(bin_v_histogram) - 1):
        if bin_v_histogram[i + 1] > bin_v_histogram[i]:
            tmp = i
        if bin_v_histogram[i + 1] < bin_v_histogram[i]:
            digits.append((tmp, i))

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

    return digit_imgs


# digit recognition
def digit_reg(model, region):
    digit_imgs = digit_sep(region)
    digit_imgs = [Image.fromarray(img).resize((28, 28)) for img in digit_imgs]
    # for img in digit_imgs:
    #     cv2.imshow('img', np.array(img))
    #     cv2.waitKey(0)
    digit_imgs = np.array([np.array(img)[np.newaxis,:,:] / 255. - 0.5 for img in digit_imgs])
    with torch.no_grad():
        input_var = torch.autograd.Variable(torch.from_numpy(digit_imgs))
    output = model(input_var).data.numpy().copy()
    output_indics = list(np.argmax(output, axis=1))
    results = []
    for index in output_indics:
        results.append(classes[index])

    return ''.join(results)

def main():
    assert len(args.location) % 4 == 0, 'Each area is determined by 2 points,' \
                                        'so the number of numbers must be divisible by 4, ' \
                                        'but got {} numbers!'.format(len(args.location))
    num_regions = int(len(args.location) / 4)

    video = cv2.VideoCapture(args.video_path)

    # load ConvNet model
    model = ConvNet(num_class=len(classes))
    checkpoint = torch.load(args.weights, map_location=torch.device('cpu'))
    print("model epoch {} best prec@1: {}\n".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model.eval()
    # model.cuda()

    count = 0
    ret, frame = video.read()
    while ret:
        count += 1
        cropped_regions = []
        for i in range(num_regions):
            points = args.location[i * 4:(i + 1) * 4]
            cropped_regions.append(frame[points[0]:points[2], points[1]:points[3]])
            cv2.rectangle(frame, (points[1], points[0]), (points[3], points[2]), (0, 0, 255), 3)

        results = []
        for i in range(num_regions):
            points = args.location[i * 4:(i + 1) * 4]
            rst = digit_reg(model, cropped_regions[i])
            results.append(rst)
            cv2.putText(frame, rst, (10, points[0]+30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255))

        print('frame {}, results: {}\n'.format(count, results))
        cv2.imshow('cur_frame', frame)
        cv2.waitKey(100)
        ret, frame = video.read()



if __name__ == '__main__':
    main()