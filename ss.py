import os
import random
import shutil

path = '/home/knuvi/Desktop/kim/dataset/imagenet/train'
dir_list = os.listdir(path)
cnt = 0
p = '/home/knuvi/Desktop/hyunobae/SRGAN/data/train'
f = os.listdir(p)
print(len(f))

while True:
    rand_dir = random.choice(dir_list)
    pth = os.path.join(path, rand_dir)
    #dir 내 1개 선택해야함
    indirlist = os.listdir(pth)
    rand_img = random.choice(indirlist)
    if rand_img in f:
        indirlist.remove(rand_img)
        rand_img = random.choice(indirlist)

    std = os.path.join(pth, rand_img)
    dst = os.path.join('/home/knuvi/Desktop/hyunobae/SRGAN/data/train', rand_img)
    shutil.copy(std, dst)
    print(len(os.listdir(p)))
    if len(os.listdir(p)) == 350000:
        break


