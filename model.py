import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3 ,1)
        self.conv2 = nn.Conv2d(32, 64, 3 ,1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, data):
        data = self.conv1(data)
        data = F.relu(data)
        data = self.conv2(data)
        data = F.relu(x)
        data = F.max_pool2d(data, 2)
        data = self.dropout1(data)
        data = torch.flatten(data, 1)
        data = self.fc1(data)
        data = F.relu(data)
        data = F.dropout2(data)
        data = self.fc2(data)
        output = F.log_softmax(x, dim=1)
        return output


class LSTM(nn.Module):
    def __init__(self,NFILT=256,NOUT=4):
        super(LSTM,self).__init__()
        self.conv0 = nn.Conv2d(1,NFILT,kernel_size=(200,3),padding=(0,1),bias=False)
        self.bn0 = nn.BatchNorm2d(NFILT)
        self.gru = nn.GRU(input_size=NFILT,hidden_size=128,num_layers=1,batch_first=True,bidirectional=False)
        self.fc1 = nn.Linear(128,NOUT)


    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = x.squeeze().permute(0,2,1)
        x,_ = self.gru(x)
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.fc1(x)
        return x


class CNN_GRU(nn.Module):
    def __init__(self,NFILT=256,NOUT=4):
        super(CNN_GRU, self).__init__()
        self.conv0 = nn.Conv2d(1,NFILT,kernel_size=(200,3),padding=(0,1),bias=False)
        self.bn0 = nn.BatchNorm2d(NFILT)
        self.gru = nn.GRU(input_size=NFILT,hidden_size=128,num_layers=1,batch_first=True,bidirectional=False)
        self.fc1 = nn.Linear(128,NOUT)



    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = x.squeeze().permute(0,2,1)
        x,_ = self.gru(x)
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.fc1(x)
        return x
