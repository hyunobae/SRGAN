import argparse
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--dir_name', default='D:/compressed_dataset/vvc', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='PGDSRGAN/G.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False


MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

vid_dir = opt.dir_name
dir_list = os.listdir(vid_dir)
for folder in dir_list:
    dirname = folder
    print(dirname)
    folder = vid_dir + '/' + folder + '/' + 'lr_x4_BI'
    img_list = os.listdir(folder)

    for img in img_list:
        IMAGE_NAME = folder + '/' + img
        image = Image.open(IMAGE_NAME)
        image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
        if TEST_MODE:
            image = image.cuda()
            out = model(image)
            out_img = ToPILImage()(out[0].data.cpu())
            if not os.path.exists('D:/SRGAN/results/' + dirname):
                os.makedirs('D:/SRGAN/results/'+ dirname)
            out_img.save('D:/SRGAN/results/' + dirname+'/' + str(UPSCALE_FACTOR) + '_' + img)
