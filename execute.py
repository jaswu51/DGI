import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import DGI, LogReg
from utils import process
from utils.sampleData import KarateDataset,KarateDatasetCorrupt

import torch.nn.functional as F



data=KarateDataset()[0]
data_corrupt=KarateDatasetCorrupt()[0]



# training params
batch_size = 1
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 36
sparse = True
nonlinearity = 'prelu' # special name to separate parameters
nb_classes=2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data =  data.to(device)
data_corrupt=data
model = DGI(in_channels=data.num_node_features, hidden_channels=hid_units,out_channels=16,heads=3,edge_dim=data.edge_dim).to(device) 


optimizer_name = "Adam"
lr = 1e-1
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
epochs = 200
loss=nn.BCELoss()

mask_train = torch.cat((data.train_mask,data.train_mask),0).type(torch.LongTensor)
mask_test = torch.cat((data.test_mask,data.test_mask),0).type(torch.LongTensor)
label=torch.cat((torch.ones(40), torch.zeros(40)), 0)
label_train=label[mask_train]
label_test=label[mask_test]
def train():
  model.train()
  optimizer.zero_grad()
  loss(model(data,data_corrupt)[mask_train], label_train).backward()
  optimizer.step()

@torch.no_grad()
def test():
    model.eval()
    logits = model(data,data_corrupt)
    pred_test = logits[mask_train]
    train_loss=loss(pred_test,label_train)
#   acc1 = pred1.eq( torch.cat((torch.ones(40), torch.zeros(40)), 0).type(torch.LongTensor)[torch.cat((data.train_mask,data.train_mask),0).type(torch.LongTensor)]).sum().item() / mask1.sum().item()
    pred_test = logits[mask_test]
    test_loss=loss(pred_test,label_test)
  
#   acc = pred.eq( torch.cat((torch.ones(40), torch.zeros(40)), 0).type(torch.LongTensor)[torch.cat((data.test_mask,data.test_mask),0).type(torch.LongTensor)]).sum().item() / mask.sum().item()

    return train_loss,test_loss

for epoch in range(1, epochs):
    train()
    train_loss,test_loss = test()
    print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')