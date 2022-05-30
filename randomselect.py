import os
import random
import shutil
from PIL import Image
import numpy as np

path = '/home/knuvi/Desktop/kim/dataset/imagenet/train'
dir_list = os.listdir(path)
cnt = 0
fin = 350000


f = []
while True:
    rand_dir = random.choice(dir_list)
    pth = os.path.join(path, rand_dir)
    #dir 내 1개 선택해야함
    indirlist = os.listdir(pth)
    rand_img = random.choice(indirlist)
    std = os.path.join(pth, rand_img)
    dst = os.path.join('/home/knuvi/Desktop/hyunobae/SRGAN/data/train', rand_img)
    ig = Image.open(std)
    numimg = np.array(ig)
    try:
        h, w, c = numimg.shape
        if w>96 and h>96 and c ==3:
            shutil.copy(std, dst)
            print(len(os.listdir('/home/knuvi/Desktop/hyunobae/SRGAN/data/train')))
            if len(os.listdir('/home/knuvi/Desktop/hyunobae/SRGAN/data/train')) == 350000:
                break
        else:
            while True:
                rand_img = random.choice(indirlist)
                std = os.path.join(pth, rand_img)
                ig = Image.open(std)
                numimg = np.array(ig)
                try: 
                    h, w, c = numimg.shape
                    if h>96 and w>96 and c==3: break
                except ValueError:
                    pass

    except ValueError:
        while True:
            rand_img = random.choice(indirlist)
            std = os.path.join(pth, rand_img)
            ig = Image.open(std)
            numimg = np.array(ig)
            try: 
                h, w, c = numimg.shape
                if h>96 and w>96 and c==3: break
            except ValueError:
                pass

    shutil.copy(std, dst)
    print(len(os.listdir('/home/knuvi/Desktop/hyunobae/SRGAN/data/train')))
    if len(os.listdir('/home/knuvi/Desktop/hyunobae/SRGAN/data/train')) == 350000:
        break
