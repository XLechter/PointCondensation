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
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=4000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.0001, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.0001, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='pcaugment', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--num_point', type=int, default=2048, help='input point number of each point cloud')
    parser.add_argument('--num_category', type=int, default=10, help='category number')
    parser.add_argument('--ratio', type=float, default=0.0001, help='category number')
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    #args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    #eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist()# if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, args.num_category, args.num_point)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    #eval_it_pool = [0, 100, 500, 1000, 2000, 4000, 6000]
    eval_it_pool = [0, 2000, args.Iteration]
    print('eval_it_pool: ', eval_it_pool)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    record_file_name = 'eval_record_'+get_time()+'_dataset_'+args.dataset+'_model_'+args.model+'_lr'+str(args.lr_img)+'_ratio'+str(args.ratio)+'.txt'
    torch.random.manual_seed(args.random_seed)
    np.random.seed(175)

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
        
        def pc_normalize(pc):
            centroid = np.mean(pc, axis=0)
            pc = pc - centroid
            m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
            pc = pc / m
            return pc


        # for ch in range(channel):
        #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        #image_syn = torch.randn(size=(num_classes*args.ipc, args.num_point, channel), dtype=torch.float, requires_grad=True, device=args.device)
        #label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        #image_syn = torch.randn(size=(np.sum(syn_per_classes), args.num_point, channel), dtype=torch.float, requires_grad=True, device=args.device)
        image_syn = sample_gaussian((np.sum(syn_per_classes), args.num_point, channel), 0.5).to(args.device)*0.5
        image_syn.requires_grad=True

        #print(image_syn)

        label_syn = []
        label_syn = np.array(label_syn)
        for i in range(num_classes):
            label_syn=np.append(label_syn, np.ones(syn_per_classes[i])*i)
        print('label_syn', label_syn)
        label_syn = torch.tensor(label_syn, dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        
        
        
        print('image_syn.shape, label_syn.shape: ', image_syn.shape, label_syn.shape)

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                #image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
                image_syn.data[index[c]:index[c+1]] = get_images(c, syn_per_classes[c]).detach().data
        else:
            print('initialize synthetic data from random noise')


        log_file = 'seed_test/'+'seed_('+str(args.random_seed)+').txt'

        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.9) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
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
                        f.write('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs
                
                ''' visualize and save '''
                # save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                
                if(args.vis):
                    path = get_time()+'vis_record_lr'+str(args.lr_img)+'_ratio_'+str(args.ratio)+'_num_point_'+str(args.num_point)+'_dataset_'+str(args.dataset)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                    create_plots(np.array(image_syn_vis), it, path)
                    write_pcd(np.array(image_syn_vis), it, path)
            # if it%100==0:
            #     path = get_time()+'vis_record_lr'+str(args.lr_img)+'_ratio_'+str(args.ratio)+'_num_point_'+str(args.num_point)+'_dataset_'+str(args.dataset)+'_init_'+str(args.init)
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            #     create_plots(np.array(image_syn_vis), it, path)
            #     write_pcd(np.array(image_syn_vis), it, path)
            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

            loss_avg = 0

            ''' update synthetic data '''
            if 'BN' not in args.model: # for ConvNet
                loss = torch.tensor(0.0).to(args.device)
                #print(num_classes)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    #img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, args.num_point, channel))
                    img_syn = image_syn[index[c]:index[c+1]].reshape((syn_per_classes[c], args.num_point, channel))
                    
                    #print(it>500, it)

                    if args.dsa:# and it>500:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param, real = True)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param, real = False)

                    #print(img_real.shape, img_syn.shape)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)
                    #print('output_real syn', output_real.shape, output_syn.shape)

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            else: # for ConvNetBN
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)

                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)

                loss += torch.sum((torch.mean(torch.FloatTensor(num_classes).to(output_real.device).uniform_(0.9, 1.1)*output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(torch.FloatTensor(num_classes).to(output_syn.device).uniform_(0.9, 1.1)*output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)



            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()


            loss_avg /= (num_classes)

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


