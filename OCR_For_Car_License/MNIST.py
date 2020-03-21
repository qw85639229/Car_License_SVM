import torch
import os
from PIL import Image
import cv2

print torch.__version__
print aa

train_data, train_labels = torch.load('./data/processed/training.pt')
test_data, test_labels = torch.load('./data/processed/training.pt')

num_train = train_data.size(0)
num_test = test_data.size(0)

train_data = train_data.numpy()
test_data = test_data.numpy()
train_labels = train_labels.numpy()
test_labels = test_labels.numpy()

count = [0 for _ in range(10)]
for i in range(num_train):
    path = './data/MNIST_plus/train/{}/'.format(train_labels[i])
    if not os.path.exists(path):
        os.mkdir(path)
    count[train_labels[i]] += 1
    img = Image.fromarray(train_data[i])
    img.save(path + '{}.jpg'.format(count[train_labels[i]]))

count = [0 for _ in range(10)]
for i in range(num_test):
    path = './data/MNIST_plus/test/{}/'.format(test_labels[i])
    if not os.path.exists(path):
        os.mkdir(path)
    count[test_labels[i]] += 1
    img = Image.fromarray(test_data[i])
    img.save(path + '{}.jpg'.format(count[test_labels[i]]))