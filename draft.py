import math
import torch
import torch.nn as nn
import torch.optim as optim
from MYresnet import resnet50
from torchvision import transforms
from data_loader import AVADataset
from torch.nn import functional as F

#rnn = nn.LSTM(10, 10, 2)
#input = torch.randn(2, 3, 10)
#h0 = torch.randn(2, 3, 10)
#c0 = torch.randn(2, 3, 10)
#loss_function = nn.MSELoss()
#optimizer = optim.SGD([
#        {'params': rnn.parameters(), 'lr': 0.005}], momentum=0.9
#        )
#output, (hn, cn) = rnn(input, (h0, c0))
#optimizer.zero_grad()
#loss = loss_function(output, input)
#loss.backward()
#optimizer.step()

#GOOD_CSV_FILE = './good.csv'
#GOOD_IMG_PATH = './images'
#BAD_CSV_FILE = './bad.csv'
#BAD_IMG_PATH = './images'
#
#Image_transform = transforms.Compose([
##    transforms.RandomResizedCrop(299),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor()])
#imageset = AVADataset(csv_file=GOOD_CSV_FILE, root_dir=GOOD_IMG_PATH, transform=Image_transform)
#image_loader = torch.utils.data.DataLoader(imageset, batch_size=1, shuffle=False)
#base_model = resnet50(pretrained=True)
#
#for i,image_DATA in enumerate(image_loader):
#    output = base_model(image_DATA['image'])
#    conv1x1 = nn.Conv2d(512, 1, kernel_size=1)
#    output = conv1x1(output)
#    StrideX = math.floor(output.squeeze().shape[0]/4) + 1
#    StrideY = math.floor(output.squeeze().shape[1]/4) + 1
#    maxpooling = nn.MaxPool2d(kernel_size=3, stride=[StrideX, StrideY])
#    output = maxpooling(output).squeeze().view(-1)

import matplotlib.pyplot as plt
import numpy as np

sig = 1
sig2 = 0.5
sig3 = 1.5
sig4 = 0.8
mu = 0
np.random.seed([39])
x = np.linspace(-5,5,200)
y = (1/(np.sqrt(2*np.pi)*sig))*np.exp(-pow(x-mu,2)/(2*pow(sig,2)))
y2 = (1/(np.sqrt(2*np.pi)*sig2))*np.exp(-pow(x-mu,2)/(2*pow(sig2,2)))
y3 = (1/(np.sqrt(2*np.pi)*sig3))*np.exp(-pow(x-mu,2)/(2*pow(sig3,2)))
y4 = (1/(np.sqrt(2*np.pi)*sig4))*np.exp(-pow(x-mu,2)/(2*pow(sig4,2)))
for i in range(len(y3)):
    y4[i] = pow(-1, np.random.randint(1))*np.random.rand()/30 + y4[i]

plt.plot(x,y)
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(x,y4)
