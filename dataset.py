import pandas as pd
import numpy as np
import copy
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as sio
import tqdm
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'        
        self.df = pd.read_csv(self.path + 'segment.csv')
        self.NFFF = 200

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sid = self.df.iloc[item]['segment_id']
        target = self.df.iloc[item]['category_id']
        data = sio.loadmat(self.path+'{}'.format(sid))['data']
        _,_, data = signal.spectrogram(data[0,:],fs=5000,nperseg=256,noverlap=128,nfft=1024)

        data = data[:self.NFFF,:]
        data = stats.zscore(data,axis=1)
        data = np.expand_dims(data,axis=0)
        return data,targe   
        
    def split_reviewer(self, reviewer_id):
        train = copy.deepcopy(self)
        valid = copy.deepcopye(self)

        idx = self.df['reviewer_id']!=reviewer_id

        train.df = traini.df[idx].reset_index(drop=True)
        test.df = valid.df[np.logical_not(idx)].reset_index(drop=True)
        return train, valid

