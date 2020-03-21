import torch.utils.data as data
import cv2
from PIL import Image
import os
import os.path
import numpy as np
import torch
import random
from torchvision import transforms


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class Dataset_Number(data.Dataset):
    def __init__(self, root_path, train, num_per_class=8):

        self.root_path = root_path
        self.train = train
        self.num_per_class = num_per_class
        self.input_size = (28, 28)

        self._parse_classes()

    def _load_image(self, path):
        img = Image.open(path).convert('L')
        img = img.resize(self.input_size)
        return img

    def _parse_classes(self):
        if self.train:
            self.train_path = self.root_path + 'data/MNIST_plus/train/'
            self.classes = os.listdir(self.train_path)
            self.classes = sorted(self.classes)

        else:
            self.val_path = self.root_path + 'data/MNIST_plus/test/'
            self.classes = os.listdir(self.val_path)
            self.classes = sorted(self.classes)
            self.val_file_list = []
            for cls in self.classes:
                files = os.listdir(self.val_path + cls)
                for file in files:
                    self.val_file_list.append(self.val_path + cls + '/' + file)

    def _get_train_images(self, class_index):
        img_files = os.listdir(self.train_path + self.classes[class_index])
        random.shuffle(img_files)
        sample_files = img_files[:self.num_per_class]
        labels = [class_index for _ in range(self.num_per_class)]

        images = list()
        for file in sample_files:
            img_path = self.train_path + self.classes[class_index] + '/' + file
            img = self._load_image(img_path)
            images.append(np.array(img)[np.newaxis,:,:] / 255. - 0.5)

        process_data = torch.from_numpy(np.array(images))
        labels = np.array(labels)

        return process_data, labels

    def _get_val_images(self, index):
        path = self.val_file_list[index]
        label = self.classes.index(path.split('/')[-2])
        image = self._load_image(path)
        process_data = torch.from_numpy(np.array(image)[np.newaxis,:,:] / 255. - 0.5)
        return process_data, label

    def __getitem__(self, index):

        if not self.train:
            sampled_images = self._get_val_images(index)
        else:
            sampled_images = self._get_train_images(index)

        return sampled_images

    def __len__(self):
        if not self.train:
            return len(self.val_file_list)
        else:
            return len(self.classes)

    def _get_num_class(self):
        return len(self.classes)


class Dataset_English(data.Dataset):
    def __init__(self, root_path, train, num_per_class=8, random_crop=True, color_jittering=True, color_flip=True):

        self.root_path = root_path
        self.train = train
        self.num_per_class = num_per_class
        self.input_size = (28, 28)
        self.expand_size = (32, 32)

        self.random_crop = random_crop
        self.color_jittering = color_jittering
        self.color_flip = color_flip
        self.toTensor = transforms.ToTensor()

        self._parse_classes()

    def _load_image(self, path):
        img = cv2.imread(path)
        pad_ud = int(img.shape[0] / 7.5)
        b = img[0, int(img.shape[1] / 2), 0]
        g = img[0, int(img.shape[1] / 2), 1]
        r = img[0, int(img.shape[1] / 2), 2]
        pad_b = b * np.ones([pad_ud, img.shape[1]])
        pad_g = g * np.ones([pad_ud, img.shape[1]])
        pad_r = r * np.ones([pad_ud, img.shape[1]])
        pad_img = np.stack([pad_b, pad_g, pad_r], axis=2)
        img = np.concatenate([pad_img, img, pad_img], axis=0).astype(np.uint8)
        if img.shape[1] < img.shape[0]/2:
            pad = int((img.shape[0]*3/4 - img.shape[1] + 1) / 2)
            b = img[int(img.shape[0] / 2), 0, 0]
            g = img[int(img.shape[0] / 2), 0, 1]
            r = img[int(img.shape[0] / 2), 0, 2]
            pad_b = b * np.ones([img.shape[0], pad])
            pad_g = g * np.ones([img.shape[0], pad])
            pad_r = r * np.ones([img.shape[0], pad])
            pad_img = np.stack([pad_b, pad_g, pad_r], axis=2)
            img = np.concatenate([pad_img, img, pad_img], axis=1).astype(np.uint8)
        if self.train:
            if self.random_crop:
                if random.random() > 0.5:
                    img = cv2.resize(img, self.expand_size)
                    img = self._random_crop(img)
                else:
                    img = cv2.resize(img, self.input_size)
            if self.color_flip:
                if random.random() > 0.5:
                    img = 255 - img
            if self.color_jittering:
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                H, S, V = cv2.split(hsv_img)
                H_ = (H + 255 * random.random()) % 256
                H_ = H_.astype(np.uint8)
                S_ = S * random.random()
                S_ = S_.astype(np.uint8)
                V_ = V * (1 - 0.5 * random.random())
                V_ = V_.astype(np.uint8)
                img = np.stack([H_, S_, V_], axis=2)
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            img = cv2.resize(img, self.input_size)
        else:
            img = cv2.resize(img, self.input_size)

        return img

    def _random_crop(self, img):
        shape = img.shape
        x_start = int((shape[1] - self.input_size[1]) * random.random())
        y_start = int((shape[0] - self.input_size[0]) * random.random())
        return img[y_start:y_start+self.input_size[0], x_start:x_start+self.input_size[1]]

    def _parse_classes(self):
        if self.train:
            self.train_path = os.path.join(self.root_path, 'lower', 'train')
            self.classes = os.listdir(self.train_path)
            self.classes = sorted(self.classes)

        else:
            self.val_path = os.path.join(self.root_path, 'lower', 'test')
            self.classes = os.listdir(self.val_path)
            self.classes = sorted(self.classes)
            self.val_file_list = []
            for cls in self.classes:
                files = os.listdir(os.path.join(self.val_path, cls))
                for file in files:
                    self.val_file_list.append(os.path.join(self.val_path, cls, file))

    def _get_train_images(self, class_index):
        img_files = os.listdir(os.path.join(self.train_path, self.classes[class_index]))
        random.shuffle(img_files)
        sample_files = img_files[:self.num_per_class]
        labels = [class_index for _ in range(self.num_per_class)]

        images = list()
        for file in sample_files:
            img_path = os.path.join(self.train_path, self.classes[class_index], file)
            img = self._load_image(img_path)
            images.append(self.toTensor(np.array(img))-0.5)

        process_data = torch.stack(images, dim=0)
        labels = torch.from_numpy(np.array(labels))

        return process_data, labels

    def _get_val_images(self, index):
        path = self.val_file_list[index]
        label = self.classes.index(path.split('/')[-2])
        image = self._load_image(path)
        process_data = self.toTensor(np.array(image)) - 0.5
        label = torch.from_numpy(np.array(label))
        return process_data, label

    def __getitem__(self, index):

        if not self.train:
            sampled_images = self._get_val_images(index)
        else:
            sampled_images = self._get_train_images(index)

        return sampled_images

    def __len__(self):
        if not self.train:
            return len(self.val_file_list)
        else:
            return len(self.classes)

    def _get_num_class(self):
        return len(self.classes)


class Dataset_Car_License(data.Dataset):
    def __init__(self, root_path, train, num_per_class=8, random_crop=True, color_jittering=True, color_flip=True):

        self.root_path = root_path
        self.train = train
        self.num_per_class = num_per_class
        self.input_size = (28, 28)
        self.expand_size = (32, 32)

        self.random_crop = random_crop
        self.color_jittering = color_jittering
        self.color_flip = color_flip
        self.toTensor = transforms.ToTensor()

        self._parse_classes()

    def _load_image(self, path):
        img = cv2.imread(path)
        pad_ud = int(img.shape[0] / 7.5)
        b = img[0, int(img.shape[1] / 2), 0]
        g = img[0, int(img.shape[1] / 2), 1]
        r = img[0, int(img.shape[1] / 2), 2]
        pad_b = b * np.ones([pad_ud, img.shape[1]])
        pad_g = g * np.ones([pad_ud, img.shape[1]])
        pad_r = r * np.ones([pad_ud, img.shape[1]])
        pad_img = np.stack([pad_b, pad_g, pad_r], axis=2)
        img = np.concatenate([pad_img, img, pad_img], axis=0).astype(np.uint8)
        if img.shape[1] < img.shape[0]/2:
            pad = int((img.shape[0]*3/4 - img.shape[1] + 1) / 2)
            b = img[int(img.shape[0] / 2), 0, 0]
            g = img[int(img.shape[0] / 2), 0, 1]
            r = img[int(img.shape[0] / 2), 0, 2]
            pad_b = b * np.ones([img.shape[0], pad])
            pad_g = g * np.ones([img.shape[0], pad])
            pad_r = r * np.ones([img.shape[0], pad])
            pad_img = np.stack([pad_b, pad_g, pad_r], axis=2)
            img = np.concatenate([pad_img, img, pad_img], axis=1).astype(np.uint8)
        if self.train:
            if self.random_crop:
                if random.random() > 0.5:
                    img = cv2.resize(img, self.expand_size)
                    img = self._random_crop(img)
                else:
                    img = cv2.resize(img, self.input_size)
            if self.color_flip:
                if random.random() > 0.5:
                    img = 255 - img
            if self.color_jittering:
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                H, S, V = cv2.split(hsv_img)
                H_ = (H + 255 * random.random()) % 256
                H_ = H_.astype(np.uint8)
                S_ = S * random.random()
                S_ = S_.astype(np.uint8)
                V_ = V * (1 - 0.5 * random.random())
                V_ = V_.astype(np.uint8)
                img = np.stack([H_, S_, V_], axis=2)
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            img = cv2.resize(img, self.input_size)
        else:
            img = cv2.resize(img, self.input_size)

        return img

    def _random_crop(self, img):
        shape = img.shape
        x_start = int((shape[1] - self.input_size[1]) * random.random())
        y_start = int((shape[0] - self.input_size[0]) * random.random())
        return img[y_start:y_start+self.input_size[0], x_start:x_start+self.input_size[1]]

    def _parse_classes(self):
        if self.train:
            self.train_path = os.path.join(self.root_path, 'train')
            self.classes = os.listdir(self.train_path)
            self.classes = sorted(self.classes)

        else:
            self.val_path = os.path.join(self.root_path, 'test')
            self.classes = os.listdir(self.val_path)
            self.classes = sorted(self.classes)
            self.val_file_list = []
            for cls in self.classes:
                files = os.listdir(os.path.join(self.val_path, cls))
                for file in files:
                    self.val_file_list.append(os.path.join(self.val_path, cls, file))

    def _get_train_images(self, class_index):
        img_files = os.listdir(os.path.join(self.train_path, self.classes[class_index]))
        random.shuffle(img_files)
        sample_files = img_files[:self.num_per_class]
        labels = [class_index for _ in range(self.num_per_class)]

        images = list()
        for file in sample_files:
            img_path = os.path.join(self.train_path, self.classes[class_index], file)
            img = self._load_image(img_path)
            images.append(self.toTensor(np.array(img))-0.5)

        process_data = torch.stack(images, dim=0)
        labels = torch.from_numpy(np.array(labels))

        return process_data, labels

    def _get_val_images(self, index):
        path = self.val_file_list[index]
        label = self.classes.index(path.split('/')[-2])
        image = self._load_image(path)
        process_data = self.toTensor(np.array(image)) - 0.5
        label = torch.from_numpy(np.array(label))
        return process_data, label

    def __getitem__(self, index):

        if not self.train:
            sampled_images = self._get_val_images(index)
        else:
            sampled_images = self._get_train_images(index)

        return sampled_images

    def __len__(self):
        if not self.train:
            return len(self.val_file_list)
        else:
            return len(self.classes)

    def _get_num_class(self):
        return len(self.classes)