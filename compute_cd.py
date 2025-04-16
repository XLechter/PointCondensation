import os
import numpy as np
import torch
import argparse
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, ParamDiffAug, TensorDataset
import copy
import gc
import sys

proj_dir = '/data/zhangwenxiao/SDT'
sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D
from fscore import fscore
chamLoss = dist_chamfer_3D.chamfer_3DDist()

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='random', help='random/herding/DSA/DM')
    parser.add_argument('--dataset', type=str, default='ModelNet40', help='dataset')
    parser.add_argument('--model', type=str, default='PointNet', help='model')
    parser.add_argument('--ipc', type=int, default=5, help='image(s) per class')
    parser.add_argument('--steps', type=int, default=5, help='5/10-step learning')
    parser.add_argument('--num_eval', type=int, default=3, help='evaluation number')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='modelnet40_normal_resampled/', help='dataset path')
    parser.add_argument('--saved_data_path', type=str, default='condensed_data/image_syn_ScanObjectNN_0.05.npy', help='data path for saved condensed data')
    parser.add_argument('--saved_label_path', type=str, default='condensed_data/label_syn_ScanObjectNN_0.05.npy', help='data path for saved condensed data label')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True # augment images for all methods
    args.dsa_strategy = 'color_crop_cutout_flip_scale_rotate' # for CIFAR10/100

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)


    ''' all training data '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)


    condensed_images_all = np.load(args.saved_data_path)
    condensed_labels_all = np.load(args.saved_label_path)
    condensed_indices_class = [[] for c in range(num_classes)]

    for i, lab in enumerate(condensed_labels_all):
        print(i, lab)
        condensed_indices_class[lab].append(i)
    print('condensed_images_all.shape', condensed_images_all.shape)
    print('condensed_labels_all', condensed_labels_all.shape)
    print(condensed_indices_class)

    condensed_images_all = torch.tensor(condensed_images_all, device=args.device)
    condensed_labels_all = torch.tensor(condensed_labels_all, dtype=torch.long, device=args.device)
    
    cd_sum = 0
    for c in range(num_classes):
        images_trian = images_all[indices_class[c]]
        images_condensed = condensed_images_all[condensed_indices_class[c]]
        print('len(images_condensed)', len(images_condensed), len(images_trian))
        for i in range(len(images_condensed)):
            ref = torch.unsqueeze(images_condensed[i], 0).repeat(images_trian.shape[0], 1, 1).cuda()
            #print(ref.shape)
            dist1, dist2, _, _ = chamLoss(ref.float(), images_trian)
            # print(dist1, dist2)
            cd = torch.min(dist1.mean(1)+ dist2.mean(1))
            cd_sum = cd_sum + cd

    cd_sum = cd_sum / len(condensed_images_all) * 1000
    print(cd_sum)

    # for c in range(num_classes):
    #     print('class c = %d: %d real images' % (c, len(indices_class[c])))

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    def get_condensed_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(condensed_indices_class[c])[:n]
        return condensed_images_all[idx_shuffle]

if __name__ == '__main__':
    main()