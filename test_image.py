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
parser.add_argument('--dir_name', default='/home/knuvi/Desktop/davis10/lr', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='/home/knuvi/Desktop/netG_epoch_4_126.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False


MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME))
else:
    model.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))

vid_dir = opt.dir_name
dir_list = os.listdir(vid_dir)
for folder in dir_list:
    dirname = folder
    print(dirname)
    folder = vid_dir + '/' + folder + '/' + 'lr'
    img_list = os.listdir(folder)

    for img in img_list:
        IMAGE_NAME = folder + '/' + img
        print(IMAGE_NAME)
        image = Image.open(IMAGE_NAME)
        image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
        if TEST_MODE:
            image = image.cuda()
            out = model(image)
            out_img = ToPILImage()(out[0].data.cpu())
            if not os.path.exists('/home/knuvi/Desktop/daviso/' + dirname):
                os.makedirs('/home/knuvi/Desktop/daviso/'+ dirname)
            out_img.save('/home/knuvi/Desktop/daviso/' + dirname+'/' + str(UPSCALE_FACTOR) + '_' + img)
