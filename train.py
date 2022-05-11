import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator
# from fortest import *
# from model import Discriminator
from distillmodel import Discriminator
import time
from torch.utils.tensorboard import SummaryWriter

# def scheduler(cfg, netD, fadein):
#     batch_size = cfg.batch_size
#     ttick = cfg.trans_tick
#     stick = cfg.stabile_tick
#
#     delta = 1.0/(2*ttick + 2*stick)
#     d_alpha = 1.0*batch_size/ttick/cfg.TICK
#
#     if cfg.is_fade == True: # fadein layer exist -> update alpha
#         fadein.update_alpha(d_alpha)


parser = argparse.ArgumentParser('PGDSRGAN')  # progressive growing discriminator SRGAN

parser.add_argument('--fsize', default=128, type=int)
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--TICK', type=int, default=1000)
parser.add_argument('--trans_tick', type=int, default=200)
parser.add_argument('--stabile_tick', type=int, default=100)
parser.add_argument('--is_fade', type=bool, default=False)
parser.add_argument('--grow', type=int, default=0)
parser.add_argument('--max_grow', type=int, default=3)
parser.add_argument('--when_to_grow', type=int, default=256)  # discriminator 증가 언제
parser.add_argument('--version', type=int, default=0)

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    batch_size = opt.batch_size
    count_image_number = 0
    trns_tick = opt.trans_tick
    stab_tick = opt.stabile_tick
    is_fade = opt.is_fade
    change_iter = opt.when_to_grow
    cur_grow = 0
    version = opt.version

    delta = 1.0 / (2 * trns_tick + 2 * stab_tick)
    d_alpha = 1.0 * batch_size / trns_tick / opt.TICK

    fadein = {'dis': is_fade}

    writer = SummaryWriter('runs/distill')

    train_set = TrainDatasetFromFolder('/home/knuvi/Desktop/hyunobae/BasicSR/datasets/train/gt', crop_size=CROP_SIZE,
                                       upscale_factor=UPSCALE_FACTOR,
                                       batch_size=batch_size)
    val_set = ValDatasetFromFolder('/home/knuvi/Desktop/hyunobae/BasicSR/datasets/val/gt',
                                   upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    # netD = Discriminator()
    netD = Discriminator(opt)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        # print(netD)
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_flag = 0
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()

        for data, target in train_bar:  # train epoch
            count_image_number += batch_size

            g_update_first = True
            running_results['batch_sizes'] += batch_size

            if (epoch == 20 or epoch == 55) and cur_grow < opt.max_grow and epoch_flag == 0:
                print(f"PGD {opt.version}")
                epoch_flag = 1
                opt.version = opt.version + 1
                netD = Discriminator(opt)
                optimizerD = optim.Adam(netD.parameters())
                print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
                netD.cuda()
                cur_grow += 1

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            ##
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(
                    (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        # save model parameters
        torch.save(netG.state_dict(), 'epochs/pgd/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/pgd/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        writer.add_scalar("VAL/psnr", valing_results['psnr'], epoch)
        writer.add_scalar("VAL/ssim", valing_results['ssim'], epoch)
        writer.add_scalar("loss/g_loss", running_results['g_loss'] / running_results['batch_sizes'], epoch)
        writer.add_scalar("loss/d_loss", running_results['d_loss'] / running_results['batch_sizes'], epoch)

        # if epoch % 10 == 0 and epoch != 0:
        #     out_path = 'statistics/'
        #     data_frame = pd.DataFrame(
        #         data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
        #               'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
        #         index=range(1, epoch+1))
        #     data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')

    end = time.time()
    print('time elapsed: ', end - start)
