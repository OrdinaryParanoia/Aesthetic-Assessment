import torch
from torch import nn as nn
import torch.nn.functional as F

__all__ = ['CORLoss', 'MSELoss', 'EMDLoss', 'CEPLoss', 'EDWKLLoss', 'CONTRASTLoss']

def corLoss(distribution1, distribution2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distribution1 = distribution1.float()
    distribution2 = distribution2.float()
    distribution1 = distribution1.squeeze()
    distribution2 = distribution2.squeeze()
    assert distribution1.shape == distribution2.shape
    loss = torch.zeros([1]).to(device)
    for i in range(len(distribution1)):
        mean1 = torch.mean(distribution1[i])+0.001
        mean2 = torch.mean(distribution2[i])+0.001
        std1 = torch.std(distribution1[i])
        std2 = torch.std(distribution2[i])
        COV = sum((distribution1[i][j]-mean1)*(distribution2[i][j]-mean2) for j in range(len(distribution1[i]))/(len(distribution1[i])-1))
        COR = COV / (std1*std2+0.001)
        COR = pow(COR,3)
        temp_loss_1 = -torch.log((COR + 1 + 0.001) / 2)
        temp_loss_2 = emd_loss(F.softmax(distribution1[i],dim=0).view(-1,1),F.softmax(distribution2[i],dim=0).view(-1,1))
        loss += abs(COR)*temp_loss_1 + (1-abs(COR))*temp_loss_2
    return loss

class CORLoss(nn.Module):
    
    def __init__(self):
        super(CORLoss, self).__init__()
        return

    def forward(self, distribution1, distribution2):
        loss = corLoss(distribution1, distribution2)
        return loss

def mseLoss(distribution1, distribution2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distribution1 = distribution1.float()
    distribution2 = distribution2.float()
    distribution1 = distribution1.squeeze()
    distribution2 = distribution2.squeeze()
    assert distribution1.shape == distribution2.shape
    loss = torch.zeros([1]).to(device)
    for i in range(len(distribution1)):
       MSE = 0.5*sum(pow(distribution1[i][j]-distribution2[i][j],2) for j in range(10))
       loss += MSE
    return loss

class MSELoss(nn.Module):
    
    def __init__(self):
        super(MSELoss, self).__init__()
        return

    def forward(self, distribution1, distribution2):
        loss = mseLoss(distribution1, distribution2)
        return loss
    
def single_emd_loss(p, q, r=2):
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += sum(torch.abs(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)

def emd_loss(p, q, r=2):
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size

class EMDLoss(nn.Module):
    
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        cdf_target = torch.cumsum(p_target, dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()
   
def cepLoss(distribution1,distribution2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distribution1 = distribution1.float()
    distribution2 = distribution2.float()
    distribution1 = distribution1.squeeze()
    distribution2 = distribution2.squeeze()
    assert distribution1.shape == distribution2.shape
    loss = torch.zeros([1]).to(device)
    for i in range(len(distribution1)):
        CEP = -sum(distribution1[i][j]*torch.log(distribution2[i][j]+1e-5) for j in range(len(distribution1)))
        loss += CEP
    
    return loss

class CEPLoss(nn.Module):
    
    def __init__(self):
        super(CEPLoss, self).__init__()
    
    def forward(self, distribution1, distribution2):
        loss = cepLoss(distribution1, distribution2)
        return loss

class EDWKLLoss(nn.Module):
    def __init__(self):
        super(EDWKLLoss, self).__init__()
    def caculateK(self, p):
        p_mean = torch.mean(p,dim=1).view(-1,1)
        p_pmean = p - p_mean
        u1 = torch.mean(pow(p_pmean,1), dim=1).view(-1,1)
        u2 = torch.mean(pow(p_pmean,2), dim=1).view(-1,1)
        u3 = torch.mean(pow(p_pmean,3), dim=1).view(-1,1)
        u4 = torch.mean(pow(p_pmean,4), dim=1).view(-1,1)

        K1 = u1
        K2 = u2 - u1*u1
        K3 = u3 - 3*u2*u1 + 2*u1*u1*u1
        K4 = u4 - 4*u3*u1 + 12*u2*u1 - 6*u1*u1*u1*u1
        return K1,K2,K3,K4
        
    def forward(self, p_target, p_estimate):
        p_target = p_target.view(-1,10,1)
        p_estimate = p_estimate.view(-1,10,1)
        assert p_target.shape == p_estimate.shape
        p_target = p_target.squeeze().view(-1,10)
        p_estimate = p_estimate.squeeze().view(-1,10)
        KX1, KX2, KX3, KX4 = self.caculateK(p_estimate)
        X1 = (p_estimate - KX1)*torch.sqrt(1/KX2)
        KX11, KX12, KX13, KX14 = self.caculateK(X1)
        
        KY1, KY2, KY3, KY4 = self.caculateK(p_target)
        Y1 = (p_target-KY1)*torch.sqrt(1/KY2)
        KY11, KY12, KY13, KY14 = self.caculateK(Y1)
        
        alpha = (KX1 - KY1)/KY2
        beta = torch.sqrt(KX2)/KY2
        
        c2 = torch.pow(alpha,2) + torch.pow(beta,2)
        c3 = torch.pow(alpha,3) + 3*alpha*torch.pow(beta,2)
        c4 = torch.pow(alpha,4) + 6*torch.pow(alpha,2)*torch.pow(beta,2) + 3*torch.pow(beta,4)
        c6 = torch.pow(alpha,6) + 15*torch.pow(alpha,4)*torch.pow(beta,2) + 45*torch.pow(alpha,2)*torch.pow(beta,4) + 15*torch.pow(beta,6)
        
        a1 = c3 - 3*alpha/KY2
        a2 = c4 - 9*c2/KY2 + 3/torch.pow(KY2,2)
        a3 = c6 - 15*c4/KY2 + 45*c2/torch.pow(KY2,2) - 15/torch.pow(KY2,3)
        
        KL1 = 1/12*KX13*KX13/KX2/KX2
        + 1/2*(torch.log(KY2/KX2)
        - 1 + 1/KY2*(KX1-KY1+torch.sqrt(KX2)*(KX1-KY1+torch.sqrt(KX2))))
        - (KY13*a1/6 + KY14*a2/24 + KY13*KY13*a3/72)
        - 1/2*KY13*KY13/36*(c6 - 6*c4/KX2 + 9*c2/KY2/KY2)
        - 10*KX13*KY13*(KX1 - KY1)*(KX2 - KY2)/torch.pow(KY2,6)
        
        KL2 = 1/12*KY13*KY13/KY2/KY2
        + 1/2*(torch.log(KX2/KY2)
        -1 + 1/KX2*(KY1-KX1 + torch.sqrt(KY2)*(KY1 - KX1 + torch.sqrt(KY2))))
        - (KX13*a1/6 + KX14*a2/24 + KX13*KX13*a3/72)
        - 1/2*KX13*KX13/36*(c6 - 6*c4/KY2 + 9*c2/KX2/KX2)
        - 10*KY13*KX13*(KY1 - KX1)*(KY2 - KX2)/torch.pow(KX2,6)
        return 1/((-torch.log(KL1)/2 - torch.log(KL2)/2).mean())
    
def IN_contrastLoss(distribution1,distribution2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distribution1 = distribution1.float()
    distribution2 = distribution2.float()
    distribution1 = distribution1.squeeze()
    distribution2 = distribution2.squeeze()
    assert distribution1.shape == distribution2.shape
    loss = torch.zeros([1]).to(device)
    CEP = -sum(distribution1[i]*torch.log(distribution2[i]+1e-5) for i in range(len(distribution1)))
    loss += CEP
    
    return loss

def OUT_contrastLoss(distribution1,distribution2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distribution1 = distribution1.float()
    distribution2 = distribution2.float()
    distribution1 = distribution1.squeeze()
    distribution2 = distribution2.squeeze()
    assert distribution1.shape == distribution2.shape
    loss = torch.zeros([1]).to(device)
    mean1 = torch.mean(distribution1)+0.001
    mean2 = torch.mean(distribution2)+0.001
    std1 = torch.std(distribution1)
    std2 = torch.std(distribution2)
    COV = sum((distribution1[i]-mean1)*(distribution2[i]-mean2) for i in range(len(distribution1)))/(len(distribution1)-1)
    COR = COV / (std1*std2+0.001)
    loss += 0.5*pow(COR+0.001,2)
    
    return loss


class CONTRASTLoss(nn.Module):
    
    def __init__(self):
        super(CONTRASTLoss, self).__init__()
        
    def forward(self, distribution1, distribution2, Mark='OUT'):
        if Mark == 'IN':
            loss = IN_contrastLoss(distribution1, distribution2)
        if Mark == 'OUT':
            loss = OUT_contrastLoss(distribution1, distribution2)
        return loss