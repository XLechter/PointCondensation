'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import h5py

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNetDataLoader(Dataset):
    def __init__(self, root, num_point, num_category, split='train', process_data=False):
        self.root = root
        self.npoints = num_point
        self.process_data = False
        self.uniform = False
        self.use_normals = False
        self.num_category = num_category

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # shape_ids = {}
        # if self.num_category == 10:
        #     shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
        #     shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        # else:
        #     shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        #     shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        # assert (split == 'train' or split == 'test')
        # shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
        #                  in range(len(shape_ids[split]))]
        # print('The size of %s data is %d' % (split, len(self.datapath)))

        # if self.uniform:
        #     self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        # else:
        #     self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))


        # self.all_data = []
        # self.all_label = []
        # for index in range(len(self.datapath)):
        #     fn = self.datapath[index]
        #     cls = self.classes[self.datapath[index][0]]
        #     label = np.array([cls]).astype(np.int32)
        #     point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

        #     if self.uniform:
        #         point_set = farthest_point_sample(point_set, self.npoints)
        #     else:
        #         point_set = point_set[0:self.npoints, :]
                
        #     point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        #     if not self.use_normals:
        #         point_set = point_set[:, 0:3]
        #     self.all_data.append(np.expand_dims(np.array(point_set),0).astype('float32'))
        #     self.all_label.append(np.expand_dims(np.array(label),0).astype('int32'))
        # self.all_data = np.concatenate(self.all_data, axis=0)
        # self.all_label = np.concatenate(self.all_label, axis=0)
        # print('all_data.shape[0]', self.all_data.shape, self.npoints, split)

        # f = h5py.File('/home/zhangwenxiao/PointCondensation/modelnet10_'+split+'.h5', 'w')
        # f.create_dataset("data", data=self.all_data)
        # f.create_dataset("label", data=self.all_label)
        # f.close()

        # print('finish'+'split'+' h5')
        f = h5py.File('modelnet10_'+split+'.h5', 'r')
        self.all_data = np.array(f['data']).astype('float32')
        self.all_label = np.array(f['label']).astype('int32')
        f.close()
        print(self.all_data.shape)

    def __len__(self):
        return self.all_data.shape[0]

    def _get_item(self, index):
        pointcloud = self.all_data[index]
        label = self.all_label[index]
        pointcloud = torch.from_numpy(np.array(pointcloud))
        label = torch.from_numpy(np.array(label).astype(np.int64))
        return pointcloud, label
    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
