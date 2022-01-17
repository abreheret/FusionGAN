
import torch, time, cv2, os, math, random, argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image
from model import *
from dataset import YouTubePose


input_channel = 3
channel = 64
def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        if classname.find('Conv') != -1:
                nn.utils.spectral_norm(m)

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=int, default=None, help="checkpoint to initiaze the training")
opt = parser.parse_args()

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
generator = Generator(ResidualBlock,input_channel, channel)
#generator.apply(weights_init)
print("load checkpoint-{}.pt".format(opt.load))
checkpoint = torch.load('./CheckPoint/checkpoint-{}.pt'.format(opt.load), map_location=device)
# print(checkpoint['netG_state_dict'].keys())
# print("="*80)
# print(generator.state_dict().keys())
#exit()

generator.load_state_dict(torch.load('CheckPoint/checkpoint-{}.pt'.format(opt.load))['netG_state_dict'])
generator.cuda()

#generator.load_state_dict(checkpoint['netG_state_dict'],strict=False)
#generator.load_checkpoint('./CheckPoint/checkpoint-{}.pt'.format(opt.load))
generator.eval()



resolution = (256,256)
transform = transforms.Compose([
        transforms.Resize(resolution),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = YouTubePose('./Dataset/', 3, transform)
train_dataloader = DataLoader(train_dataset, batch_size = 1,shuffle = True)

for i, sample in enumerate(train_dataloader, 0): #start = 0
        x = sample['x'].to(device)
        y = sample['y'].to(device)
        gxy = generator(x,y)
        x = x.detach().cpu()
        y = y.detach().cpu()
        gxy = gxy.detach().cpu()
        sample = []
        sample.extend([x[0], y[0], gxy[0] ])
        result_img = utils.make_grid(sample, padding = 0,normalize = True, nrow = 3)
        result_img = result_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR )
        cv2.imshow("result",result_img) 
        if cv2.waitKey(0) == 27:
                exit(0)



