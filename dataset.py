import torch.utils.data as data
import dill
import os
import random
import numpy as np
from hrv.filters import moving_average
from hrv.rri import RRi
import copy

class MitbinDataset(data.Dataset):
    def __init__(self, opt, is_for_train):
        super(MitbinDataset, self).__init__()
        self._name = 'MitbinDataset'
        self._root = './data'
        self._opt = opt
        self._is_for_train = is_for_train

        self._data = []
        self._label = []
        self._subject = []

        self._mean_train = []
        self._sdnn_train = []
        self._pnn50_train = []
        self._rmssd_train = []
        self._lnrmssd_train = []
        self._vlf_train = []
        self._lf_train = []
        self._hf_train = []
        self._rlh_train = []


        # read dataset
        self._read_dataset_paths()

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        #if self._is_for_train:  # random
        #    index = random.randint(0, self._dataset_size-1)

        # get sample data
        data = self._data[index] / 1000
        data = list(data)

        data_reverse = copy.deepcopy(data)
        data_reverse.reverse()
        
        filt_rri1 = list(moving_average(RRi(data), order=1))      
        filt_rri2 = list(moving_average(RRi(data), order=2))
        filt_rri3 = list(moving_average(RRi(data), order=3))

        filt_rri1_reverse = copy.deepcopy(filt_rri1)
        filt_rri2_reverse = copy.deepcopy(filt_rri2)
        filt_rri3_reverse = copy.deepcopy(filt_rri3)
        filt_rri1_reverse.reverse()
        filt_rri2_reverse.reverse()
        filt_rri3_reverse.reverse()

        order_data = [filt_rri1, filt_rri2, filt_rri3]
        order_data_reverse = [filt_rri1_reverse, filt_rri2_reverse, filt_rri3_reverse]
        
        label = int(self._label[index])
        subject = self._subject[index]

        mean = self._mean_train[index]
        sdnn = self._sdnn_train[index]
        pnn50 = self._pnn50_train[index]
        rmssd = self._rmssd_train[index]
        lnrmssd = self._lnrmssd_train[index]
        vlf = self._vlf_train[index]
        lf = self._lf_train[index]
        hf = self._hf_train[index]
        rlh = self._rlh_train[index]

        features = list(np.stack((mean, sdnn, pnn50, rmssd, lnrmssd, \
                             vlf, lf, hf, rlh )))

        makeup_length = 512-len(data)
        if len(data) > 512:
            data = data[:512]
        else:
            data.extend(0 for _ in range(makeup_length))
       
        return data, data_reverse, order_data, order_data_reverse, label, subject, features

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        fold = self._opt.fold
        data_name = 'mitbin_train_%dfold.pkl' % fold if self._is_for_train else 'mitbin_test_%dfold.pkl' % fold
        read_path = os.path.join(self._root, data_name)
        with open(read_path, 'rb') as fin:
            res = dill.load(fin)

        self._data = res['data']
        self._label = res['label']
        self._subject = res['subject']

        self._mean_train = res['mean']
        self._sdnn_train = res['sdnn']
        self._pnn50_train = res['pnn50']
        self._rmssd_train = res['rmssd']
        self._lnrmssd_train = res['lnrmssd']
        self._vlf_train = res['vlf']
        self._lf_train = res['lf']
        self._hf_train = res['hf'] 
        self._rlh_train = res['rlh']

        self._dataset_size = len(self._data)
        
        
        

    
        
