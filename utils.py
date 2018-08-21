import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#def compute_acc(label: torch.Tensor, pred: torch.Tensor):
#    dist = (torch.arange(10).float()-5.5).to(device)
#    p_mean = (pred.view(-1, 10) * dist).sum(dim=1)
#    l_mean = (label.view(-1, 10) * dist).sum(dim=1)
#    p_good = p_mean > 0
#    l_good = l_mean > 0
#    acc = (p_good == l_good).float().mean()
#    return acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def compute_acc(label: torch.Tensor, pred: torch.Tensor):
    label = label.float()
    pred = pred.float()
    dist = (torch.arange(10).float()).to(device)
    p_mean = (pred.view(-1, 10) * dist).sum(dim=1)
    l_mean = (label.view(-1, 10) * dist).sum(dim=1)
    p_good = p_mean >= 4.5
    l_good = l_mean >= 4.5
    acc = (p_good == l_good).float().mean()
    return acc