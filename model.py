import torch
from torch import nn as nn
from torch.nn import functional as F
from model_utils import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux, BasicConv2d


class ourNet(nn.Module):

    def __init__(self, num_classes=10, aux_logits=False, transform_input=False):
        super(ourNet, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
#        self.Cascade_5b = nn.Linear(256 * 35 * 35, 128)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Cascade_5c = nn.Linear(288 * 35 * 35, 128)
        self.Mixed_5d = InceptionA(288, pool_features=64)
#        self.Cascade_5d = nn.Linear(288 * 35 * 35, 128)
        self.Mixed_6a = InceptionB(288)
#        self.Cascade_6a = nn.Linear(768 * 17 * 17, 128)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Cascade_6b = nn.Linear(768 * 17 * 17, 128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Cascade_6c = nn.Linear(768 * 17 * 17, 128)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
#        self.Cascade_6d = nn.Linear(768 * 17 * 17, 128)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
#        self.Cascade_6e = nn.Linear(768 * 17 * 17, 128)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
#        self.Cascade_7a = nn.Linear(1280 * 8 * 8, 128)
        self.Mixed_7b = InceptionE(1280)
        self.Cascade_7b = nn.Linear(2048 * 8 * 8, 128)
        self.Mixed_7c = InceptionE(2048)
#        self.Cascade_7c = nn.Linear(2048 * 8 * 8, 128)
        self.Cascade_final = nn.Linear(2048, 1024)
        self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        #        temp = x.view(x.size(0), -1)
        #        Inception_Cascade_A_1 = self.Cascade_5b(temp)
        x = self.Mixed_5c(x)
        temp = x.view(x.size(0), -1)
        Inception_Cascade_A_2 = self.Cascade_5c(temp)
        x = self.Mixed_5d(x)
        #        temp = x.view(x.size(0), -1)
        #        Inception_Cascade_A_3 = self.Cascade_5d(temp)
        x = self.Mixed_6a(x)
        #        temp = x.view(x.size(0), -1)
        #        Inception_Cascade_B_1 = self.Cascade_6a(temp)
        x = self.Mixed_6b(x)
        temp = x.view(x.size(0), -1)
        Inception_Cascade_B_2 = self.Cascade_6b(temp)
        x = self.Mixed_6c(x)
        temp = x.view(x.size(0), -1)
        Inception_Cascade_B_3 = self.Cascade_6c(temp)
        x = self.Mixed_6d(x)
        #        temp = x.view(x.size(0), -1)
        #        Inception_Cascade_B_4 = self.Cascade_6d(temp)
        x = self.Mixed_6e(x)
        #        temp = x.view(x.size(0), -1)
        #        Inception_Cascade_B_5 = self.Cascade_6e(temp)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        #        temp = x.view(x.size(0), -1)
        #        Inception_Cascade_C_1 = self.Cascade_7a(temp)
        x = self.Mixed_7b(x)
        temp = x.view(x.size(0), -1)
        Inception_Cascade_C_2 = self.Cascade_7b(temp)
        x = self.Mixed_7c(x)
        #        temp = x.view(x.size(0), -1)
        #        Inception_Cascade_C_3 = self.Cascade_7c(temp)
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        Inception_Cascade_final = self.Cascade_final(x)
        #        temp = torch.cat([
        #                Inception_Cascade_A_1,
        #                Inception_Cascade_A_2,
        #                Inception_Cascade_A_3,
        #                Inception_Cascade_B_1,
        #                Inception_Cascade_B_2,
        #                Inception_Cascade_B_3,
        #                Inception_Cascade_B_4,
        #                Inception_Cascade_B_5,
        #                Inception_Cascade_C_1,
        #                Inception_Cascade_C_2,
        #                Inception_Cascade_C_3,
        #                Inception_Cascade_final],1)
        temp = torch.cat([
            Inception_Cascade_A_2,
            Inception_Cascade_B_2,
            Inception_Cascade_B_3,
            Inception_Cascade_C_2,
            Inception_Cascade_final], 1)
        x = self.fc(temp)
        if self.training and self.aux_logits:
            return x, aux
        return x
