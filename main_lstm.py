import torch
import torch.optim as optim
import torch.nn as nn
#import torch.nn.functional as F
from torchvision import transforms
from data_loader import AVADataset
from loss import CONTRASTLoss, CORLoss, EDWKLLoss, EMDLoss
from utils import compute_acc
#import torchvision.models as models
from model_featuremap import ourNet
from model_featuremap import FEATURE_ON_X_AXIS, FEATURE_ON_Y_AXIS
#from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ourNet()
model = model.to(device)
contrastLoss = CONTRASTLoss()
optimizer = optim.SGD([
        {'params': model.parameters(), 'lr': 0.005}], momentum=0.9
        )
optimizer_oflstm = optim.SGD([
        {'params': model.parameters(), 'lr': 0.005}], momentum=0.9
        )

#GOOD_CSV_FILE = '/home/paranoia/desktop/AVA_dataset/good.csv'
#GOOD_IMG_PATH = '/home/paranoia/desktop/AVA_dataset/images/images'
#BAD_CSV_FILE = '/home/paranoia/desktop/AVA_dataset/bad.csv'
#BAD_IMG_PATH = '/home/paranoia/desktop/AVA_dataset/images/images'

GOOD_CSV_FILE = './good.csv'
GOOD_IMG_PATH = './images'
BAD_CSV_FILE = './bad.csv'
BAD_IMG_PATH = './images'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH1 = 5
EPOCH2 = 10
VECTOR_TRAIN_MARK = 1

#def makePairs(image_loader1, image_loader2):
#    training_pairs = []
#    for i, image_DATA1 in enumerate(image_loader1):
#        outputs = model(image_DATA1['image'].to(device))
#        target = image_DATA1['annotations'].view(10,1)
#        training_pairs.append([outputs, target])
#    for i, image_DATA2 in enumerate(image_loader2):
#        outputs = model(image_DATA1['image'].to(device))
#        target = image_DATA1['annotations'].view(10,1)
#        training_pairs.append([outputs, target])
#    return training_pairs

INPUT_CSV_FILE_1 = GOOD_CSV_FILE
INPUT_IMG_PATH_1 = GOOD_IMG_PATH
INPUT_CSV_FILE_2 = BAD_CSV_FILE
INPUT_IMG_PATH_2 = BAD_IMG_PATH

csv_file1 = GOOD_CSV_FILE
csv_file2 = BAD_CSV_FILE
root_dir1 = GOOD_IMG_PATH
root_dir2 = BAD_IMG_PATH

Image_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
imageset1 = AVADataset(csv_file=csv_file1, root_dir=root_dir1, transform=Image_transform)
imageset2 = AVADataset(csv_file=csv_file2, root_dir=root_dir2, transform=Image_transform)
image_loader1 = torch.utils.data.DataLoader(imageset1, batch_size=1, shuffle=False)
image_loader2 = torch.utils.data.DataLoader(imageset2, batch_size=1, shuffle=False)

if VECTOR_TRAIN_MARK == 1:
    step = 0
    for i in range(EPOCH1):
        for image_DATA1, image_DATA2 in zip(image_loader1, image_loader2):
            outputs1 = model(image_DATA1['image'].to(device))
            outputs2 = model(image_DATA2['image'].to(device))
            if outputs1.shape != outputs2.shape:
                continue
            if outputs1.shape[0] != 16:
                continue
            outputs1 = outputs1.view(-1, 16, 1)
            outputs2 = outputs2.view(-1, 16, 1)
            step += 1
            optimizer.zero_grad()
            loss = contrastLoss(outputs1, outputs2)
#            print(loss)
#            print('*'*20)
#            print(outputs1)
#            print(outputs2)
            loss.backward()
            optimizer.step()

#training_pairs = makePairs(image_loader1, image_loader2)

rnn = nn.LSTM(1, 10, 5)
#Loss_oflstm = nn.MSELoss()
#Loss_oflstm = CORLoss
#Loss_oflstm = CONTRASTLoss()
Loss_oflstm = EDWKLLoss()
#Loss_oflstm = EMDLoss()

for i in range(EPOCH2):
    TRAIN_ACC = []
    VAL_ACC = []
    for step, image_DATA1 in enumerate(image_loader1):
        input_oflstm = model(image_DATA1['image'].to(device)).view(-1,1,1).float()
        labels_oflstm = image_DATA1['annotations']
        optimizer_oflstm.zero_grad()
        output_oflstm, (hn, cn) = rnn(input_oflstm)
        if output_oflstm.shape[0] != 16:
            continue
        TRAIN_ACC.append(compute_acc(
                labels_oflstm, 
                output_oflstm[FEATURE_ON_X_AXIS*FEATURE_ON_Y_AXIS-1]).item())
        loss_oflstm = Loss_oflstm(
                output_oflstm[FEATURE_ON_X_AXIS*FEATURE_ON_Y_AXIS-1].squeeze(),
                labels_oflstm.float().squeeze())
#        loss_oflstm = Loss_oflstm(
#                output_oflstm[FEATURE_ON_X_AXIS*FEATURE_ON_Y_AXIS-1].squeeze(),
#                labels_oflstm.float().squeeze(),
#                Mark = 'IN')
#        print(loss_oflstm)
        print('Latest ACC: %.4f' % (sum(TRAIN_ACC)/len(TRAIN_ACC)))
        loss_oflstm.backward()
        optimizer_oflstm.step()
    for step, image_DATA2 in enumerate(image_loader2):
        input_oflstm = model(image_DATA2['image'].to(device)).view(-1,1,1).float()
        labels_oflstm = image_DATA2['annotations']
        optimizer_oflstm.zero_grad()
        output_oflstm, (hn, cn) = rnn(input_oflstm)
        if output_oflstm.shape[0] != 16:
            continue
        TRAIN_ACC.append(compute_acc(
                labels_oflstm, 
                output_oflstm[FEATURE_ON_X_AXIS*FEATURE_ON_Y_AXIS-1]).item())
        loss_oflstm = Loss_oflstm(
                output_oflstm[FEATURE_ON_X_AXIS*FEATURE_ON_Y_AXIS-1].squeeze(),
                labels_oflstm.float().squeeze())
#        loss_oflstm = Loss_oflstm(
#                output_oflstm[FEATURE_ON_X_AXIS*FEATURE_ON_Y_AXIS-1].squeeze(),
#                labels_oflstm.float().squeeze(),
#                Mark = 'IN')
#        print(loss_oflstm)
        print('Latest ACC: %.4f' % (sum(TRAIN_ACC)/len(TRAIN_ACC)))
        loss_oflstm.backward()
        optimizer_oflstm.step()
        