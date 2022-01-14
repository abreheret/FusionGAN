import torch, time, cv2, os, math, random, argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=int, default=None, help="checkpoint to initiaze the training")

opt = parser.parse_args()

    
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
resolution = (256,256)
dataset_dir = './Dataset/'
transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
class_num = 3
input_channel = 3
channel = 64
batch_size = 8
num_epochs = 1000
lr_D = 0.0003
lr_G = 0.0001
alpha = 2
beta = 1

save_dir = './CheckPoint/'

class YouTubePose(Dataset):
    def __init__(self, dataset_dir, class_num, transform = None):
        self.dataset_dir = dataset_dir
        self.class_num = class_num
        self.transform = transform
          
    def __len__(self):
        length = []
        for i in range(self.class_num):
            dir_path = self.dataset_dir +'train/class{}_cropped'.format(i + 1)
            length.append(len(os.walk(dir_path).__next__()[2]))
        max_len = max(length)
        return max_len    
    
    def __getitem__(self, idx):
        randx, randy = random.sample(range(1, self.class_num + 1), 2)
        randi = random.choice(range(1, self.class_num + 1))
        x, x_hat, = random.sample(os.listdir(self.dataset_dir + 'train/class{}_cropped'.format(randx)), 2)
        id1, id2 = random.sample(os.listdir(self.dataset_dir + 'train/class{}_cropped'.format(randi)), 2)
        y = random.choice(os.listdir(self.dataset_dir + 'train/class{}_cropped'.format(randy)))
        x = Image.open(self.dataset_dir + 'train/class{}_cropped/'.format(randx)+ x)
        y = Image.open(self.dataset_dir + 'train/class{}_cropped/'.format(randy)+ y)
        x_hat = Image.open(self.dataset_dir + 'train/class{}_cropped/'.format(randx)+ x_hat)
        id1 = Image.open(self.dataset_dir + 'train/class{}_cropped/'.format(randi)+ id1)
        id2 = Image.open(self.dataset_dir + 'train/class{}_cropped/'.format(randi)+ id2)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
            x_hat = self.transform(x_hat)
            id1 = self.transform(id1)
            id2 = self.transform(id2)
       
        sample = {'x' : x, 'y' : y, 'x_hat' : x_hat, 'id1' : id1, 'id2' : id2}
        return sample
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    
    if classname.find('Conv') != -1:
        nn.utils.spectral_norm(m)
        

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
    def __init__(self, resblock):
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
    def __init__(self):
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


train_dataset = YouTubePose(dataset_dir, class_num, transform)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size,shuffle = True)
                             
def save_checkpoint(state, dirpath, epoch):
    filename = 'checkpoint-{}.pt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)
    print('--- checkpoint saved to ' + str(checkpoint_path) + ' ---')
    
generator = Generator(ResidualBlock).to(device)
generator.apply(weights_init)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)

optimizer_G = optim.Adam(generator.parameters(), lr = lr_G, betas = (0.5,0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr = lr_D, betas = (0.5, 0.999))


real_label = 1.
fake_label = 0.
size_pool = 8
pooling = nn.MaxPool2d(size_pool, stride=size_pool, return_indices=True)
def MinPatchPooling(out_model):
    out_model = 1 - out_model
    pool, indice = pooling(out_model)
    return 1 - pool , indice
    
def QuarterPatchPooling(out_model):
    return out_model[out_model<out_model.mean()/4]

def MedianPatchPooling(out_model):
    return out_model[out_model<out_model.median()]
    
def create_image_sample(x,y,fake,Gyy,Gxy2,id1,id2,out_ls1,indice=None) :
    sample = []
    for i in range(min(batch_size,4)):
        sample.extend([x[i], y[i], fake[i], Gyy[i], Gxy2[i], id1[i], id2[i], out_ls1[i] ])
    result_img = utils.make_grid(sample, padding = 0,normalize = True, nrow = 8)
    result_img = result_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR )
    y = 230
    cv2.putText(result_img, '  X   ', (0*256+80,y) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(result_img, '  Y   ', (1*256+80,y) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(result_img, 'G(X,Y)', (2*256+80,y) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(result_img, 'G_ls2a', (3*256+80,y) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(result_img, 'G_ls2b', (4*256+80,y) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(result_img, '  id1 ', (5*256+80,y) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(result_img, '  id2 ', (6*256+80,y) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(result_img, 'G_ls1 ', (7*256+80,y) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
    if indice is not None : 
        for i_batch in range(min(batch_size,4)):
            i_patch = indice[i_batch].detach().cpu()
            y_center = 8*i_patch.div(32,rounding_mode='trunc').flatten().numpy()
            x_center = 8*i_patch.remainder(32).flatten().numpy()
            for i in range(len(x_center)):
                xx, yy = x_center[i]+256*2 , y_center[i]+256*i_batch
                cv2.rectangle(result_img,(xx-2*size_pool, yy-2*size_pool),(xx+2*size_pool, yy+2*size_pool),(0,0,255))
    return result_img

def Train(load = None):
    epoch_start = 0
    if load is not None:
        print("load checkpoint-{}.pt".format(load))
        dataloaded = torch.load('./CheckPoint/checkpoint-{}.pt'.format(load))
        generator.load_state_dict(dataloaded['netG_state_dict'])
        discriminator.load_state_dict(dataloaded['netD_state_dict'])
        optimizer_G.load_state_dict(dataloaded['gen_opt'])
        optimizer_D.load_state_dict(dataloaded['disc_opt'])
        epoch_start = dataloaded['epoch']
        print("Checkpoint-{}.pt is loaded".format(load))
        
    generator.train()
    discriminator.train()
    Start = time.time()
    loss_i = nn.MSELoss()
    loss_s = nn.L1Loss()
    
    for epoch in range(epoch_start, num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 10)

        start = time.time()
        for i, sample in enumerate(train_dataloader, 0): #start = 0
            x, y, x_hat = sample['x'].to(device), sample['y'].to(device), sample['x_hat'].to(device)
            id1 = sample['id1'].to(device)
            id2 = sample['id2'].to(device)

            # -----------------------------------------------------------------------------
            # Ix != Iy:  Update the G ← maximizes Identity loss LI (Min-Patch Training is applied)
            optimizer_G.zero_grad()
            fake = generator(x, y)
            #disc_real = discriminator(x, x_hat)
            out = discriminator(x, fake)
            # disc_fake, indice = MinPatchPooling(out)
            # disc_fake = out
            # disc_fake = MeanPatchPooling(out)
            disc_fake = QuarterPatchPooling(out)
            if len(disc_fake.flatten()) < 16 :
                disc_fake, indice = MinPatchPooling(out)
            else :
                indice = None
            # print(disc_fake.shape)
            #flabel = torch.full((disc_fake.size()), fake_label, dtype=disc_fake.dtype, device=device)
            rlabel = torch.full((disc_fake.size()), real_label, dtype=disc_fake.dtype, device=device)
            LI_G = loss_i(rlabel, disc_fake) # + loss_i(flabel, disc_real)
            LI_G.backward()
            optimizer_G.step()

            # -----------------------------------------------------------------------------
            # Update the D ← minimizes Identity loss LI
            optimizer_D.zero_grad()
            disc_fake = discriminator(x, fake.detach())
            disc_real = discriminator(x, x_hat)
            flabel = torch.full((disc_fake.size()), fake_label, dtype=disc_fake.dtype, device=device)
            rlabel = torch.full((disc_real.size()), real_label, dtype=disc_real.dtype, device=device)
            LI_Dr = loss_i(rlabel, disc_real)
            LI_Df = loss_i(flabel, disc_fake)
            LI_Dr.backward()
            LI_Df.backward()
            LI_D = LI_Dr+LI_Df
            optimizer_D.step()

            # -----------------------------------------------------------------------------
            # Update the G ← minimizes Shape loss Ls2a, Ls2b
            optimizer_G.zero_grad()
            fake = fake.detach()
            Gyy = generator(y, fake)
            Gxy2 = generator(fake, y)
            Ls2a = loss_s(y, Gyy)
            Ls2b = loss_s(fake, Gxy2)
            Ls2a.backward()
            Ls2b.backward()
            Ls2 = Ls2a + Ls2b
            optimizer_G.step()
            
            # -----------------------------------------------------------------------------
            # Update the D ← minimizes Identity loss LI
            # optimizer_D.zero_grad()
            # disc_fake1 = discriminator(Gyy.detach(), y)
            # disc_fake2 = discriminator(Gxy2.detach(), x)
            # f1abel = torch.full((disc_fake1.size()), fake_label, dtype=disc_fake1.dtype, device=device)
            # LI_Df1 = loss_i(f1abel, disc_fake1)
            # LI_Df2 = loss_i(f1abel, disc_fake2)
            # LI_Df1.backward()
            # LI_Df2.backward()
            # LI_Df = LI_Df1+LI_Df2
            # optimizer_D.step()


            # -----------------------------------------------------------------------------
            # Ix == Iy : Update the G ← minimizes Shape loss Ls1            
            optimizer_G.zero_grad()
            output_ls1 = generator(id1, id2)
            Ls1 = loss_s(id2, output_ls1) 
            Ls1.backward()
            optimizer_G.step()

            if (i % 10 == 0):
                print("[{:d}/{:d}] LI_G:{:.7f} LI_D:{:.4f} Ls1:{:.4f} Ls2:{:0.4f} Ls2a:{:.4f} Ls2b:{:.4f}".format(i, len(train_dataloader),  LI_G, LI_D, Ls1, Ls2, Ls2a, Ls2b))
                result_img = create_image_sample(x.detach().cpu(),
                                           y.detach().cpu(),
                                           fake.detach().cpu(),
                                           Gyy.detach().cpu(),
                                           Gxy2.detach().cpu(),
                                           id1.detach().cpu(),
                                           id2.detach().cpu(),
                                           output_ls1.detach().cpu(),
                                           indice)
                    
                cv2.imshow("result",result_img)                
                
                if cv2.waitKey(5) == 27:
                    exit(0)
                
        save_checkpoint({'epoch': epoch + 1,'netG_state_dict': generator.state_dict(),'netD_state_dict': discriminator.state_dict(),'gen_opt': optimizer_G.state_dict(),'disc_opt': optimizer_D.state_dict()}, save_dir, epoch + 1)    
        print("="*80)
        print('Time taken by epoch: {:.0f}h {:.0f}m {:.0f}s'.format(((time.time() - start) // 60) // 60, (time.time() - start) // 60, (time.time() - start) % 60))
        result_img = create_image_sample(x.detach().cpu(),
                                   y.detach().cpu(),
                                   fake.detach().cpu(),
                                   Gyy.detach().cpu(),
                                   Gxy2.detach().cpu(),
                                   id1.detach().cpu(),
                                   id2.detach().cpu(),
                                   output_ls1.detach().cpu(),
                                   indice)
        cv2.imwrite("./result/reult-{}epoch.png".format(epoch + 1),result_img)

Train(opt.load)