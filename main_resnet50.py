import torch
import torch.optim as optim
from torchvision import transforms
from data_loader import AVADataset

from loss import CORLoss, MSELoss, EMDLoss, CEPLoss, EDWKLLoss

from model_resnet50 import ResNet50

from utils import compute_acc
import torchvision.models as models

##########################################config
TRAIN_CSV_FILE = '/home/paranoia/AVA_test/train.csv'
VAL_CSV_FILE = '/home/paranoia/AVA_test/val.csv'
TRAIN_IMG_PATH = '/home/paranoia/AVA_test/train'
VAL_IMG_PATH = '/home/paranoia/AVA_test/val'

torch.manual_seed(1)
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
EPOCH = 4

LR = 0.001
conv_base_lr = 0.01
dense_lr = 0.01
lr_decay_rate = 0.95
lr_decay_freq = 10

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
val_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
corLoss = CORLoss()
mseLoss = MSELoss()
emdLoss = EMDLoss()
cepLoss = CEPLoss()
edwklLoss = EDWKLLoss()

base_model = models.resnet50(pretrained=True)
model = ResNet50(base_model)
model = model.to(device)

optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
        )

trainset = AVADataset(csv_file=TRAIN_CSV_FILE, root_dir=TRAIN_IMG_PATH, transform=train_transform)
valset = AVADataset(csv_file=VAL_CSV_FILE, root_dir=VAL_IMG_PATH, transform=val_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE,
                                           shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(valset, batch_size=VAL_BATCH_SIZE,
                                         shuffle=False, num_workers=4)

train_loss_public = torch.tensor([])
val_loss_public = torch.tensor([])
for epoch in range(EPOCH):
    print('-' * 5 + '>' + 'Epoch {}/{}'.format(epoch, EPOCH - 1))
    print('=' * 40)

    train_batch_losses = []
    val_batch_losses = []
    step = 0
    TRAIN_ACC = []
    VAL_ACC = []

    for i, DATA in enumerate(train_loader):
        images = DATA['image'].to(device)
        labels = DATA['annotations'].to(device).float()
        outputs = model(images)
        step += 1
        optimizer.zero_grad()
        outputs = outputs.view(-1, 10, 1)
        TRAIN_ACC.append(compute_acc(labels,outputs).item())
        train_losses = corLoss(labels, outputs)
#        train_losses = mseLoss(labels, outputs)
#        train_losses = emdLoss(labels, outputs)
#        train_losses = cepLoss(labels, outputs)
#        train_losses = edwklLoss(labels, outputs)
        train_batch_losses.append(train_losses.item())
        train_losses.backward()
        optimizer.step()
        if (i + 1) % 1000 == 0:
            print('--->')
            print('Latest loss: %.4f' % (train_batch_losses))
            print('Latest ACC: %.4f' % (sum(TRAIN_ACC)/len(TRAIN_ACC)))

    avg_loss = sum(train_batch_losses) / (len(trainset) // TRAIN_BATCH_SIZE + 1)
    acc = sum(TRAIN_ACC)/len(TRAIN_ACC)

    print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))
    print('Epoch %d training ACC: %.4f' % (epoch + 1, acc))

    if (epoch + 1) % 10 == 0:
        conv_base_lr = conv_base_lr * lr_decay_rate ** ((epoch + 1) / lr_decay_freq)
        dense_lr = dense_lr * lr_decay_rate ** ((epoch + 1) / lr_decay_freq)
        optimizer = optim.SGD([
                {'params': model.features.parameters(), 'lr': conv_base_lr},
                {'params': model.classifier.parameters(), 'lr': dense_lr}],
                momentum=0.9
                )

    for j, DATA in enumerate(val_loader):
        model.eval()
        images = DATA['image'].to(device)
        labels = DATA['annotations'].to(device).float()
        with torch.no_grad():
            outputs = model(images)
        step += 1
        outputs = outputs.view(-1, 10, 1)
        VAL_ACC.append(compute_acc(labels,outputs).item())
        val_losses = corLoss(labels, outputs)
#        val_losses = mseLoss(labels, outputs)
#        val_losses = emdLoss(labels, outputs)
#        val_losses = cepLoss(labels, outputs)
#        val_losses = edwklLoss(labels, outputs)
        val_batch_losses.append(val_losses.item())
        
    avg_loss = sum(val_batch_losses) / (len(valset) // VAL_BATCH_SIZE + 1)
    acc = sum(VAL_ACC)/len(VAL_ACC)

    print('Epoch %d averaged valuing loss: %.4f' % (epoch + 1, avg_loss))
    print('Epoch %d valuing ACC: %.4f' % (epoch + 1, acc))

print('Training completed.')
