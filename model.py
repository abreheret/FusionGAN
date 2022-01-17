import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride = 1, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = 3,stride = stride, padding = padding)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, padding = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        if torch.cuda.is_available():
             x = torch.cuda.FloatTensor(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out) 

        return out

class Generator(nn.Module):
    def __init__(self, resblock, input_channel, channel):
        super(Generator, self).__init__()
        
        self.x_conv1 = nn.Sequential(nn.Conv2d(input_channel, channel, 7, 1, 3),nn.BatchNorm2d(channel),nn.ReLU(),)
        self.y_conv1 = nn.Sequential(nn.Conv2d(input_channel, channel, 7, 1, 3),nn.BatchNorm2d(channel),nn.ReLU(),)
        
        self.x_conv2 = nn.Sequential(nn.Conv2d(channel, channel * 2, 3, 2, 1),nn.BatchNorm2d(channel * 2),nn.ReLU(),)
        self.y_conv2 = nn.Sequential(nn.Conv2d(channel, channel * 2, 3, 2, 1),nn.BatchNorm2d(channel * 2),nn.ReLU(),)
        
        self.x_conv3 = nn.Sequential(nn.Conv2d(channel * 2, channel * 4, 3, 2, 1),nn.BatchNorm2d(channel * 4),nn.ReLU(),)
        self.y_conv3 = nn.Sequential(nn.Conv2d(channel * 2, channel * 4, 3, 2, 1),nn.BatchNorm2d(channel * 4),nn.ReLU(),)
        
        self.x_resblock = resblock(channel * 4, channel * 4)
        self.y_resblock = resblock(channel * 4, channel * 4)
        self.resblock = resblock(channel * 4, channel * 4)
        
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(channel * 4, channel * 2, 4, 2, 1),nn.BatchNorm2d(channel * 2),nn.ReLU(),)
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(channel * 2, channel, 4, 2, 1),nn.BatchNorm2d(channel),nn.ReLU(),)
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(channel, input_channel, 7, 1, 3),nn.Tanh(),)

    def forward(self, x, y):
        x_out = self.x_conv1(x) 
        x_out = self.x_conv2(x_out) 
        x_out1 = x_out
        x_out = self.x_conv3(x_out)
        x_out2 = x_out
        x_out = self.x_resblock(x_out)
        x_out = self.x_resblock(x_out)
        
        y_out = self.y_conv1(y)
        y_out = self.y_conv2(y_out)
        y_out1 = y_out
        y_out = self.y_conv3(y_out)
        y_out2 = y_out
        y_out = self.y_resblock(y_out)
        y_out = self.y_resblock(y_out)
        
        out = torch.add(x_out, y_out)
        out = self.resblock(out)
        out = self.resblock(out)
        out = torch.add(out, x_out2)
        out = torch.add(out, y_out2)
       
        out = self.deconv1(out)
        out = torch.add(out, x_out1)
        out = torch.add(out, y_out1)
        out = self.deconv2(out)
        out = self.deconv3(out)
        return out
        
class Discriminator(nn.Module): #batch Norm 
    def __init__(self, input_channel, channel):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel * 2, channel, 4, 2, 1),nn.LeakyReLU(0.2),)
        self.conv2 = nn.Sequential(nn.Conv2d(channel, channel * 2, 4, 2, 1),nn.BatchNorm2d(channel * 2),nn.LeakyReLU(0.2),nn.Conv2d(channel * 2, channel * 4, 4, 2, 1),nn.BatchNorm2d(channel * 4),nn.LeakyReLU(0.2),)
        self.conv3 = nn.Sequential(nn.Conv2d(channel * 4, channel * 8, 3, 1, 1),nn.BatchNorm2d(channel * 8),nn.LeakyReLU(0.2),nn.Conv2d(channel*8,channel * 8, 3, 1, 1),nn.BatchNorm2d(channel * 8),nn.LeakyReLU(0.2),)
        self.conv4 = nn.Sequential(nn.Conv2d(channel * 8, 1, 3, 1, 1),nn.Sigmoid())
    
    def forward(self,x, y):
        xy = torch.cat([x,y], dim = 1)
        out = self.conv1(xy)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out