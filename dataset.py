import os, random
from torch.utils.data import Dataset
from PIL import Image

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