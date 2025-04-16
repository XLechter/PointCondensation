import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from vis_pcd import create_plots, write_pcd

def truncated_normal(tensor, mean=0, std=1, trunc_std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return 

def sample_gaussian(size, truncate_std=None, gpu=None):
    y = torch.randn(*size).float()
    y = y if gpu is None else y.cuda(gpu)
    if truncate_std is not None:
        truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
    return y
    
def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DM', help='DC/DSA/DM')
    parser.add_argument('--dataset', type=str, default='ShapeNet', help='dataset')
    parser.add_argument('--model', type=str, default='PointNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='test', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=10000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.0001, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.001, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='pcaugment', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--num_point', type=int, default=2048, help='input point number of each point cloud')
    parser.add_argument('--num_category', type=int, default=10, help='category number')
    parser.add_argument('--ratio', type=float, default=0.0001, help='category number')
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()
    #args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist()# if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, args.num_category, args.num_point)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    record_file_name = 'eval_record_'+get_time()+'_dataset_'+args.dataset+'_model_'+args.model+'_lr'+str(args.lr_img)+'_ratio'+str(args.ratio)+'.txt'

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        print(torch.unsqueeze(dst_train[0][0], dim=0).shape)

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        print('images_all labels_all', images_all.shape, labels_all.shape)
        

        syn_per_classes = []
        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))
            syn_per_classes.append(len(indices_class[c])*args.ratio)
        #print('num_classes', syn_per_classes)
        syn_per_classes = np.ceil(np.array(syn_per_classes)).astype(int)
        print('syn_per_classes',syn_per_classes)


        index = [0]
        temp = 0
        for i in range(num_classes):
            temp = temp + syn_per_classes[i]
            index.append(temp)
        index = np.array(index)
        print('index', index)

        
        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]


        # for ch in range(channel):
        #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        #image_syn = torch.randn(size=(num_classes*args.ipc, args.num_point, channel), dtype=torch.float, requires_grad=True, device=args.device)
        #label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        
        # image_syn = sample_gaussian((np.sum(syn_per_classes), args.num_point, channel), 0.5).to(args.device)*0.5
        # image_syn.requires_grad=True

        # label_syn = []
        # label_syn = np.array(label_syn)
        # for i in range(num_classes):
        #     label_syn=np.append(label_syn, np.ones(syn_per_classes[i])*i)
        # print('label_syn', label_syn)
        label_syn = np.load('syn_data/label_syn_it6000.npy')
        label_syn = torch.tensor(label_syn, dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        image_syn = np.load('syn_data/image_syn_it6000.npy')
        image_syn = torch.tensor(image_syn, dtype=torch.float32, requires_grad=False, device=args.device)

        log_file = get_time()+'eval_record_lr'+str(args.lr_img)+'_ratio_'+str(args.ratio)+'_num_point_'+str(args.num_point)+'_dataset_'+str(args.dataset)+'_model_'+str(args.model)+'.txt'

        seed = 0
        eval_acc = 1.0

        for seedi in range(500):
            torch.random.manual_seed(0)
            np.random.seed(0)

            ''' training '''
            optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.9) # optimizer_img for synthetic data
            optimizer_img.zero_grad()
            print('%s training begins'%get_time())

            ''' Evaluate synthetic data '''
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))
                with open(log_file, 'a') as f:
                    f.write('begin \n')
                accs = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, record_file_name)
                    accs.append(acc_test)
                    del net_eval
                with open(log_file, 'a') as f:
                    if np.mean(accs) > eval_acc:
                        eval_acc = np.mean(accs)
                        seed = seedi
                    f.write('Evaluate %d random %s, mean = %.4f std = %.4f seed = %.4f best_seed = %.4f, best_seed_acc = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs), seedi, seed, eval_acc))
        

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


