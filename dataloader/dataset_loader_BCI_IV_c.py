import numpy as np
import time
import scipy
import scipy.signal
import scipy.io
# import self defined functions 
from torch.utils.data import Dataset
import random
import scipy.io as sio
from scipy import interp


class DatasetLoader_BCI_IV_subjects(Dataset):

    def __init__(self, setname, args, train_aug=False):

        subject_id = 1
        data_folder='./data'
        data = sio.loadmat(data_folder+"/cross_sub/cross_subject_data_"+str(subject_id)+".mat")
        test_X	= data["test_x"][:,:,750:1500] # [trials, channels, time length]
        train_X	= data["train_x"][:,:,750:1500]

        test_y	= data["test_y"].ravel()
        train_y = data["train_y"].ravel()

        train_y-=1
        test_y-=1
        window_size = 400
        step = 50
        n_channel = 22  
        
        def windows(data, size, step):
            start = 0
            while ((start+size) < data.shape[0]):
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if(len(data[start:end]) == window_size):
                    segments = segments + [data[start:end]]
            return np.array(segments)


        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        train_win_y=train_y
        test_win_y=test_y

        expand_factor=train_win_x.shape[1]

        train_x=np.reshape(train_win_x,(-1,train_win_x.shape[2], train_win_x.shape[3]))  
        test_x=np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y=np.repeat(train_y, expand_factor)
        test_y=np.repeat(test_y, expand_factor)


        train_x=np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y=np.reshape(train_y, [train_y.shape[0]]).astype('float32')
        
        test_x=np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y=np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        test_x=test_x[2000:,:,:,:]
        test_y=test_y[2000:]
        
        val_x=test_x[:2000,:,:,:]
        val_y=test_y[:2000]

        train_win_x=train_win_x.astype('float32')       
        test_win_x=test_win_x[500:,:,:,:].astype('float32')
        test_win_y=test_win_y[500:]
        val_win_x=test_win_x[:500,:,:,:].astype('float32')
        val_win_y=test_win_y[:500]

        self.X_val=val_win_x
        self.y_val=val_win_y

        if setname == 'train':
            self.data=train_win_x
            self.label=train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label=val_win_y
        elif setname == 'test':
            self.data=test_win_x
            self.label=test_win_y

        self.num_class=4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label=self.data[i], self.label[i]
        return data, label


