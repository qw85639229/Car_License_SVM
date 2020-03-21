#coding=UTF-8
import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json
from util import thresholdIntegral1
import matplotlib.pyplot as plt
import time

SZ = 20          #训练图片长宽
MAX_WIDTH = 1000 #原始图片最大宽度
Min_Area = 2000  #车牌区域允许最大面积
PROVINCE_START = 1000
#读取图片文件
def imreadex(filename):
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
	
def point_limit(point):
	if point[0] < 0:
		point[0] = 0
	if point[1] < 0:
		point[1] = 0

#k阈值分割
def k_threshold(gray_img,k):
	r, c = gray_img.shape
	c_part = int(c // k)
	for i in range(0, k):
		if i == 0:
			gray_img_1 = gray_img[:, 0:c_part]
			# cv2.imshow("color", gray_img_1)
			# cv2.waitKey(0)
			ret, gray_img_1 = cv2.threshold(gray_img_1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			output_gray_img = gray_img_1
		else:
			gray_img_1 = gray_img[:, c_part * i:c_part * (i+1)]
			# cv2.imshow("color", gray_img_1)
			# cv2.waitKey(0)
			ret, gray_img_1 = cv2.threshold(gray_img_1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			output_gray_img = np.hstack([output_gray_img, gray_img_1])
	return output_gray_img

def k_integral_threshold(gray_img,k):
	r, c = gray_img.shape
	c_part = int(c // k)
	for i in range(0, k):
		if i == 0:
			gray_img_1 = gray_img[:, 0:c_part]
			#gray_img_1 = cv2.bitwise_not(gray_img_1)
			roii = cv2.integral(gray_img_1)
			gray_img_1 = thresholdIntegral1(gray_img_1, roii).astype('uint8')
			# ret, gray_img_1 = cv2.threshold(gray_img_1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			output_gray_img = gray_img_1
		else:
			gray_img_1 = gray_img[:, c_part*i:c_part*(i+1)]
			roii = cv2.integral(gray_img_1)
			gray_img_1 = thresholdIntegral1(gray_img_1, roii).astype('uint8')
			#ret, gray_img_1 = cv2.threshold(gray_img_1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			output_gray_img = np.hstack([output_gray_img, gray_img_1])
	return output_gray_img

#k阈值分割
def kl_threshold(gray_img,k,l):
	r, c = gray_img.shape
	r_part = int(r//l)
	for j in range(0,l):
		if j ==0:
			gray_r_part = gray_img[0:r_part,:]
			gray_r_part = k_threshold(gray_r_part,k)
			output_gray_img = gray_r_part
		else:
			gray_r_part = gray_img[r_part*j:r_part*(j+1), :]
			gray_r_part = k_threshold(gray_r_part, k)
			output_gray_img = np.vstack([output_gray_img, gray_r_part])
	return output_gray_img

#根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
	up_point = -1#上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []
	for i,x in enumerate(histogram):
		if is_peak and x < threshold:
			if i - up_point > 2:
				is_peak = False
				wave_peaks.append((up_point, i))
		elif not is_peak and x >= threshold:
			is_peak = True
			up_point = i
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))
	return wave_peaks

#根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
	part_cards = []
	for wave in waves:
		part_cards.append(img[:, wave[0]:wave[1]])
	return part_cards

#来自opencv的sample，用于svm训练
def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img
#来自opencv的sample，用于svm训练
def preprocess_hog(digits):
	samples = []
	for img in digits:
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bin_n = 16
		bin = np.int32(bin_n*ang/(2*np.pi))
		bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)
		
		# transform to Hellinger kernel
		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps
		
		samples.append(hist)
	return np.float32(samples)
#不能保证包括所有省份
provinces = [
"zh_cuan", "川",
"zh_e", "鄂",
"zh_gan", "赣",
"zh_gan1", "甘",
"zh_gui", "贵",
"zh_gui1", "桂",
"zh_hei", "黑",
"zh_hu", "沪",
"zh_ji", "冀",
"zh_jin", "津",
"zh_jing", "京",
"zh_jl", "吉",
"zh_liao", "辽",
"zh_lu", "鲁",
"zh_meng", "蒙",
"zh_min", "闽",
"zh_ning", "宁",
"zh_qing", "靑",
"zh_qiong", "琼",
"zh_shan", "陕",
"zh_su", "苏",
"zh_sx", "晋",
"zh_wan", "皖",
"zh_xiang", "湘",
"zh_xin", "新",
"zh_yu", "豫",
"zh_yu1", "渝",
"zh_yue", "粤",
"zh_yun", "云",
"zh_zang", "藏",
"zh_zhe", "浙"
]
class StatModel(object):
	def load(self, fn):
		self.model = self.model.load(fn)  
	def save(self, fn):
		self.model.save(fn)
class SVM(StatModel):
	def __init__(self, C = 1, gamma = 0.5):
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)
#训练svm
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
#字符识别
	def predict(self, samples):
		r = self.model.predict(samples)
		return r[1].ravel()

class CardPredictor:
	def __init__(self):
		#车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
		f = open('config.js')
		j = json.load(f)
		for c in j["config"]:
			if c["open"]:
				self.cfg = c.copy()
				break
		else:
			raise RuntimeError('没有设置有效配置参数')

	def __del__(self):
		self.save_traindata()
	def train_svm(self):
		#识别英文字母和数字
		self.model = SVM(C=1, gamma=0.5)
		#识别中文
		self.modelchinese = SVM(C=1, gamma=0.5)
		if os.path.exists("svm.dat"):
			self.model.load("svm.dat")
		else:
			chars_train = []
			chars_label = []
			
			for root, dirs, files in os.walk("train\\chars2"):
				if len(os.path.basename(root)) > 1:
					continue
				root_int = ord(os.path.basename(root))
				for filename in files:
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_train.append(digit_img)
					#chars_label.append(1)
					chars_label.append(root_int)
			
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)
			#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
			chars_label = np.array(chars_label)
			print(chars_train.shape)
			self.model.train(chars_train, chars_label)
		if os.path.exists("svmchinese.dat"):
			self.modelchinese.load("svmchinese.dat")
		else:
			chars_train = []
			chars_label = []
			for root, dirs, files in os.walk("train\\charsChinese"):
				if not os.path.basename(root).startswith("zh_"):
					continue
				pinyin = os.path.basename(root)
				index = provinces.index(pinyin) + PROVINCE_START + 1 #1是拼音对应的汉字
				for filename in files:
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_train.append(digit_img)
					#chars_label.append(1)
					chars_label.append(index)
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)
			#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
			chars_label = np.array(chars_label)
			print(chars_train.shape)
			self.modelchinese.train(chars_train, chars_label)

	def save_traindata(self):
		if not os.path.exists("svm.dat"):
			self.model.save("svm.dat")
		if not os.path.exists("svmchinese.dat"):
			self.modelchinese.save("svmchinese.dat")

	def predict(self, car_pic):
		#目前取消了车牌颜色识别及车牌边缘微调
		card_imgs = []
		card_imgs.append(car_pic)
		colors = []
		colors.append('blue')
		#以下为识别车牌中的字符
		predict_result = []
		roi = None
		card_color = None
		for i, color in enumerate(colors):
			if color in ("blue", "yello", "green"):
				card_img = card_imgs[i]
				#20200319改用去光照转灰度图
				#裁切一部分
				#card_img = card_img[int(card_img.shape[0] * cut_rate):int(card_img.shape[0] - card_img.shape[0] * cut_rate)]
				gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
				r, c = gray_img.shape
				r = r * 10
				c = c * 10
				gray_img = cv2.resize(gray_img, (c, r))
				cv2.imshow("yes", gray_img)
				gray_img = cv2.bitwise_not(gray_img)
				#黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
				#多阈值生成二值图
				gray_img_threshold = k_integral_threshold(gray_img,1)
				#20200320测试采用多阈值方法是否合理
				concentration = np.sum(gray_img_threshold)/(gray_img_threshold.shape[0]*gray_img_threshold.shape[1])
				#print("concentration: ",concentration)
				if concentration <= 50: #判断积分阈值失败
					gray_img = cv2.bitwise_not(gray_img)
					#cv2.imshow("test1", gray_img)
					k = 6
					gray_img_threshold = k_threshold(gray_img,k)
					#cv2.imshow("test2", gray_img_threshold)
					#cv2.waitKey(0)
				gray_img = gray_img_threshold
				plt_gray_img = gray_img
				#查找水平直方图波峰
				x_histogram  = np.sum(gray_img, axis=1)
				x_min = np.min(x_histogram)
				x_average = np.sum(x_histogram)/x_histogram.shape[0]
				x_threshold = (x_min + x_average)/2
				wave_peaks = find_waves(x_threshold, x_histogram)
				if len(wave_peaks) == 0:
					print("peak less 0:")
					continue
				#认为水平方向，最大的波峰为车牌区域
				wave = max(wave_peaks, key=lambda x:x[1]-x[0])
				gray_img = gray_img[wave[0]:wave[1]]
				#查找垂直直方图波峰
				row_num, col_num= gray_img.shape[:2]
				#去掉车牌上下边缘1个像素，避免白边影响阈值判断
				gray_img = gray_img[1:row_num-1]
				y_histogram = np.sum(gray_img, axis=0)
				y_min = np.min(y_histogram)
				y_average = np.sum(y_histogram)/y_histogram.shape[0]
				y_threshold = (y_min + y_average)/5#U和0要求阈值偏小，否则U和0会被分成两半

				wave_peaks = find_waves(y_threshold, y_histogram)

				#for wave in wave_peaks:
				#	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2) 
				#车牌字符数应大于6
				if len(wave_peaks) <= 6:
					print("peak less 1:", len(wave_peaks))
					continue
				
				wave = max(wave_peaks, key=lambda x:x[1]-x[0])
				max_wave_dis = wave[1] - wave[0]
				#判断是否是左侧车牌边缘
				if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/3 and wave_peaks[0][0] == 0:
					wave_peaks.pop(0)
				
				#组合分离汉字
				cur_dis = 0
				for i,wave in enumerate(wave_peaks):
					if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
						break
					else:
						cur_dis += wave[1] - wave[0]
				if i > 0:
					wave = (wave_peaks[0][0], wave_peaks[i][1])
					wave_peaks = wave_peaks[i+1:]
					wave_peaks.insert(0, wave)
				
				#去除车牌上的分隔点
				if len(wave_peaks) <= 3:
					break
				point = wave_peaks[2]
				if point[1] - point[0] < max_wave_dis/3:
					point_img = gray_img[:,point[0]:point[1]]
					if np.mean(point_img) < 255/5:
						wave_peaks.pop(2)
				
				if len(wave_peaks) <= 6:
					print("peak less 2:", len(wave_peaks))
					continue
				part_cards = seperate_card(gray_img, wave_peaks)

				# 20200314修改策略从直接输出前七位改为累计输出7位
				predict_result_number = 0
				for i, part_card in enumerate(part_cards):
					#可能是固定车牌的铆钉
					if np.mean(part_card) < 255/5:
						#print("a point")
						continue
					#w = abs(part_card.shape[1] - SZ)//2
					#修改填边框方式
					#删除最后一个字母的右边缘
					if predict_result_number >= 6:
						if concentration <= 50:
							part_card = part_card[:,0:int(part_card.shape[1]*0.8)]
						# cv2.imshow("part_card",part_card)
						# cv2.waitKey(0)
					part_card_0 = part_card
					w = abs(part_card.shape[1] - part_card.shape[0]) // 2

					part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])
					part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
					#part_card = deskew(part_card)
					part_card = preprocess_hog([part_card])
					if i == 0:
						#判断是否为两个字粘连
						if part_card_0.shape[1] > 1.2*part_card_0.shape[0]:
							part_card_1 = part_card_0[:,0:int(part_card_0.shape[1]//2)]
							part_card_2 = part_card_0[:,int(part_card_0.shape[1]//2):part_card_0.shape[1]]
							# 第一个字测试
							w = abs(part_card_1.shape[1] - part_card_1.shape[0]) // 2
							part_card_1 = cv2.copyMakeBorder(part_card_1, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
							part_card_1 = cv2.resize(part_card_1, (SZ, SZ), interpolation=cv2.INTER_AREA)
							part_card_1 = preprocess_hog([part_card_1])
							resp = self.modelchinese.predict(part_card_1)
							charactor = provinces[int(resp[0]) - PROVINCE_START]
							predict_result.append(charactor)
							predict_result_number += 1
							# 第二个字测试
							w = abs(part_card_2.shape[1] - part_card_2.shape[0]) // 2
							part_card_2 = cv2.copyMakeBorder(part_card_2, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
							part_card_2 = cv2.resize(part_card_2, (SZ, SZ), interpolation=cv2.INTER_AREA)
							part_card_2 = preprocess_hog([part_card_2])
							resp = self.model.predict(part_card_2)
							charactor = chr(int(resp[0]))
						else:
							resp = self.modelchinese.predict(part_card)
							charactor = provinces[int(resp[0]) - PROVINCE_START]
					else:
						resp = self.model.predict(part_card)
						charactor = chr(int(resp[0]))
						# part_card_1 = cv2.resize(part_card_1, (100, 100), interpolation=cv2.INTER_AREA)
						# cv2.imshow("part_card",part_card_1)
						# cv2.waitKey(0)
					#判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
					if color == "green":
						if predict_result_number >= 8:
							continue
					elif predict_result_number >= 7:
						continue
					predict_result_number +=1
					predict_result.append(charactor)
				roi = card_img
				card_color = color
				break
		print(predict_result)
		plt.figure()
		plt.subplot(3, 1, 1)
		rgb_part_card = card_img[:, :, [2, 1, 0]]
		plt.imshow(rgb_part_card)
		plt.subplot(3, 1, 2)
		plt.imshow(plt_gray_img,cmap='gray')
		if 'part_cards' in dir():
			for i, part_card in enumerate(part_cards):
				plt.subplot(3, len(part_cards), (i+len(part_cards)*2+1))
				plt.imshow(part_card, cmap='gray')
		plt.show()

		return predict_result, roi, card_color#识别到的字符、定位的车牌图像、车牌颜色

if __name__ == '__main__':
	c = CardPredictor()
	c.train_svm()
	img_path = 'IMG_0047\\229.jpg'
	img = cv2.imread(img_path)
	# cv2.imshow("origin",img)
	cut_rate = 0.08
	r, roi, color = c.predict(img)
	# print(r)
	# for k in range(1,16):
	# 	for l in range(1,16):
	# 		r, roi, color = c.predict(img,k,l)
	# 		print(r,k,l)


	#遍历所有样本
	# path = 'IMG_0277_clear'
	# original_images = []
	# right_number = 0
	# cut_rate = 0
	# max_right_number = 0
	# target = ['粤', 'A', '5', '1', '5', 'X', 'E']
	# for root, dirs, filenames in os.walk(path):
	# 	for filename in filenames:
	# 		original_images.append(os.path.join(path, filename))
	# print("len of original_images: ", len(original_images))
	# for j in range(30):
	# 	cut_rate += 0.01
	# 	right_number = 0
	# 	for i, dirs in enumerate(original_images):
	# 		img = cv2.imread(dirs)
	# 		#print(dirs)
	# 		r, roi, color = c.predict(img, 0.08)
	# 		if r == target:
	# 			right_number += 1
	# 	print("Accuracy rate: ", (right_number / len(original_images)))
	# 	print("cut_rate: ", cut_rate)
	# 	if right_number >= max_right_number:
	# 		max_right_number = right_number
	# 		right_cut_rate = cut_rate
	# print("Max_accuracy rate: ", (max_right_number/len(original_images)))
	# print("Right_cut_rate: ", right_cut_rate)




	# #path = 'IMG_0277'
	# path = 'IMG_0047'
	# #path = '.\\..\\outputvideo'
	# # path = 'IMG_0281'
	# original_images = []
	# for root, dirs, filenames in os.walk(path):
	# 	for filename in filenames:
	# 		original_images.append(os.path.join(path, filename))
	# print("len of original_images: ", len(original_images))
	#
	# max_right_number = 0
	# target = ['粤', 'Y', '9', '6', '6', 'U', '1']
	# #target = ['粤', 'A', '5', '1', '5', 'X', 'E']
	# # for x in range(30):
	# #cut_rate = 0.08
	# right_number = 0
	# for i, dirs in enumerate(original_images):
	# 	img = cv2.imread(dirs)
	# 	r, roi, color = c.predict(img)
	# 	if r == target:
	# 		# print(dirs)
	# 		# print(r)
	# 		right_number +=1
	# 	# else:
	# 		# print(dirs)
	# 		# print(r)
	# print("Accuracy rate: ", (right_number / len(original_images)))
	# #print("cut_rate: ", cut_rate)
	# # 	if right_number >= max_right_number:
	# # 		max_right_number = right_number
	# # 		right_cut_rate = cut_rate
	# # print("Max_accuracy rate: ", (max_right_number/len(original_images)))
	# # print("Right_cut_rate: ", right_cut_rate)