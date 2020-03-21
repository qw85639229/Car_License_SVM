import os
import numpy
import shutil

root = './data/chinese/'
chars = sorted(os.listdir(root))

for c in chars:
    if not os.path.exists(os.path.join('./data/train/', c)):
        os.mkdir(os.path.join('./data/train/', c))
    if not os.path.exists(os.path.join('./data/test/', c)):
        os.mkdir(os.path.join('./data/test/', c))

    image_names = sorted(os.listdir(os.path.join(root, c)))
    num_imgs = len(image_names)
    for i, name in enumerate(image_names):
        src_path = os.path.join(root, c, name)
        if i < num_imgs * 0.9:
            dst_path = os.path.join('./data/train/', c, name)
            shutil.move(src_path, dst_path)
        else:
            dst_path = os.path.join('./data/test/', c, name)
            shutil.move(src_path, dst_path)

