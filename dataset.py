import os, sys, glob
import time
from tqdm import tqdm
import numpy as np
import h5py
import torch
import torch.utils.data
import open3d as o3d
from torch.utils.data.sampler import Sampler
import MinkowskiEngine as ME

def read_h5(filedir):
    coords = h5py.File(filedir, 'r')['coords'][:].astype('int16')
    feats = h5py.File(filedir, 'r')['feats'][:].astype('int8')


    return coords,feats

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    coords, feats = list(zip(*list_data))
    coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)

    return coords_batch, feats_batch



def yuv_rgb(YUV):
    R =np.expand_dims(YUV[:,0] + 1.13983 * YUV[:,2],1)
    # R= R.reshape(R.shape[0],1,R.shape[1],R.shape[2])
    G = np.expand_dims(YUV[:,0] - 0.39465 * (YUV[:,1]) - 0.58060 * (YUV[:,2]),1)
    # G= G.reshape(G.shape[0],1,G.shape[1],G.shape[2])
    B = np.expand_dims(YUV[:,0] + 2.03211 * (YUV[:,1]),1)
    # B= B.reshape(B.shape[0],1,B.shape[1],B.shape[2])
    RGB= np.concatenate([R,G,B],1)
    return RGB
def rgb_yuv(RGB):
    Y=0.2990*RGB[:,0] + 0.5870*RGB[:,1] + 0.1140*RGB[:,2]
    Y1=np.expand_dims(Y,1)
    # Y=Y.reshape(Y.shape[0],1,Y.shape[1],Y.shape[2])
    U= np.expand_dims(-0.14713*RGB[:,0] -0.28886*RGB[:,1] +0.436*RGB[:,2],1)
    # U=U.reshape(U.shape[0],1,U.shape[1],U.shape[2])
    V= np.expand_dims(0.615*RGB[:,0] -0.51498*RGB[:,1]-0.10001*RGB[:,2],1)
    # V=V.reshape(V.shape[0],1,V.shape[1],V.shape[2])
    YUV= np.concatenate([Y1,U,V],1)
    return YUV
class PCDataset(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files
        self.name = os.path.split(str(files))[-1].split('.')[0]+'.ply'

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):
        filedir = self.files[idx]

        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            if filedir.endswith('.h5'):
                coords,feats = read_h5(filedir)
                feats=rgb_yuv(feats/255)
            if filedir.endswith('.ply'):
                pcd = o3d.io.read_point_cloud(filedir)
                feats =np.asarray(pcd.colors)
                feats=rgb_yuv(feats)
                coords=np.asarray(pcd.points).astype('int')
            # cache
            self.cache[idx] = (coords, feats)
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent
        feats = feats.astype("float32")

        return (coords, feats)

class PCDataset_lossless(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files
        self.name = os.path.split(str(files))[-1].split('.')[0]+'.ply'

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):
        filedir = self.files[idx]

        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            if filedir.endswith('.h5'):
                coords,feats = read_h5(filedir)
                feats=rgb_yuv(feats/255)
            if filedir.endswith('.ply'):
                pcd = o3d.io.read_point_cloud(filedir)
                coords=np.asarray(pcd.points).astype('int')

                ori='./8iVFB_test'+(filedir.split('_r')[0]+'.ply').split('pcgcv2')[-1]
                ori = o3d.io.read_point_cloud(ori)
                feats =np.asarray(ori.colors)
                feats=rgb_yuv(feats)
                coords_ori=np.asarray(ori.points).astype('int')
                feats=rgb_get(coords,coords_ori,feats)
            # cache
            self.cache[idx] = (coords, feats)
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent
        feats = np.array(feats.cpu()).astype("float32")
        # indices=np.lexsort((coords[:,0],coords[:,1],coords[:,2]))
        # coords=coords[indices]
        # feats=feats[indices]
        return (coords, feats)
def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False,
                     collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader

def make_data_loader_mulgpu(dataset, train_sampler,batch_size=1, shuffle=True, num_workers=1, repeat=False,
                     collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    # if repeat:
    #     args['sampler'] = InfSampler(dataset, shuffle)
    # else:
    #     args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset,sampler=train_sampler,**args)

    return loader


if __name__ == "__main__":
    # filedirs = sorted(glob.glob('/home/ubuntu/HardDisk2/color_training_datasets/training_dataset/'+'*.h5'))
    filedirs = sorted(glob.glob('/./8iVFB_test/' + '*.ply'))
    test_dataset = PCDataset(filedirs)
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=2, shuffle=True, num_workers=1, repeat=False,
                                       collate_fn=collate_pointcloud_fn)
    for idx, (coords, feats) in enumerate(tqdm(test_dataloader)):
        print("=" * 20, "check dataset", "=" * 20,
              "\ncoords:\n", coords, "\nfeat:\n", feats)

    test_iter = iter(test_dataloader)
    print(test_iter)
    for i in tqdm(range(10)):
        coords, feats = test_iter.next()
        print("=" * 20, "check dataset", "=" * 20,
              "\ncoords:\n", coords, "\nfeat:\n", feats)



