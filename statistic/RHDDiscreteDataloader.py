from __future__ import print_function, unicode_literals

import pickle
import torch
import os
from torch.utils.data import Dataset
import numpy as np
path_to_db = './RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/shicheng/RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/home/csc/dataset/RHD_published_v2/'
if (not os.path.exists(path_to_db)):
    path_to_db = '/mnt/ssd/csc/RHD_published_v2/'
if (not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/csc/RHD_published_v2/'
    #os.environ["DISPLAY"] = "localhost:11.0"

set = 'training'
set = 'evaluation'
def get32fTensor(a)->torch.Tensor:
    if(torch.is_tensor(a)):
        return a.float()
    return torch.tensor(a,dtype=torch.float32)
class RHDDuscreteDataloader(Dataset):
    def __init__(self, train=True,path_name=path_to_db):
        print("loading rhd")
        if(train):self.mode='training'
        else:self.mode='evaluation'
        self.path_name=path_name
        with open(os.path.join(self.path_name, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            self.anno_all = pickle.load(fi)
            self.num_samples = len(self.anno_all.items())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if (idx == 20500 or idx == 28140): idx = 0
        anno=self.anno_all[idx]
        kp_coord_xyz = anno['xyz'].astype(np.float32).copy()  # x, y, z coordinates of the keypoints, in meters
        kp_coord_xyz = kp_coord_xyz[-21:, :]
        RHD2mano_skeidx=[0,8,7,6, 12,11,10, 20,19,18, 16,15,14, 4,3,2,1, 5,9,13,17]
        kp_coord_xyz=kp_coord_xyz[RHD2mano_skeidx].copy()
        return get32fTensor(kp_coord_xyz)



if __name__ == '__main__':

    train_dataset=RHDDuscreteDataloader(train=True)
    for i in range(10):
        train_dataset.__getitem__(i)
