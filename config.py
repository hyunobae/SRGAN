import argparse

parser = argparse.ArgumentParser('PGDSRGAN')#progressive growing discriminator SRGAN

parser.add_argument('--fsize', default=128, type=int)
parser.add_argument()
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=64, type=int)

