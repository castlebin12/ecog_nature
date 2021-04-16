import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from train_test import *
from model import *
from dataset import *
from torch.utils.data import DataLoader

epochs = 5
lr = 0.01
momentum = 0.5
log_interval = 200
seed = 1
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_fnusa_train,dataset_fnusa_valid = Dataset('/db/ictadmin/DATASET_FNUSA/').split_reviewer(1)
train_loader = DataLoader(dataset=dataset_fnusa_train,
                   batch_size=32,
                   shuffle=True,
                   drop_last=False,
                   num_workers=24)

test_loader = DataLoader(dataset=dataset_fnusa_valid,
                   batch_size=32,
                   shuffle=True,
                   drop_last=False,
                   num_workers=24)


if __name__ == "__main__":
    # model = CNN().to(device)
    # model = LSTM().to(device)
    model = CNN_GRU().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sgd_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        py_train(log_interval, model, device, train_loader, optimizer, epochs)
        py_test(model, device, test_loader)

    torch.save(model, './saved_model/model.pt') 
