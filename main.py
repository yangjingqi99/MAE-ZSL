import argparse
import torch
from sklearn.model_selection import train_test_split


from util.pos_embed import interpolate_pos_embed
import numpy as np
import models_vit
import torch.nn as nn
import os,sys
import scipy.io as sio

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.special import softmax
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch import optim

use_cuda = torch.cuda.is_available()


def get_args_parser():
    parser = argparse.ArgumentParser('test for image classification', add_help=False)

    parser.add_argument('--batch_size', default=523, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')


    parser.add_argument('--drop_path', default=0.1, type=float, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
 
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--num_workers', default=10, type=int)


    # change it for different classification tasks
    parser.add_argument('--nb_classes', default=312, type=int,
                        help='number of the classfication types')
    parser.add_argument('--predict', default='/home/kmyh/D/yjq/MAE/mae-main/model_save/mae_finetuned_vit_large.pth',
                        help='predict from checkpoint')
    parser.add_argument('--data_path', default='/home/kmyh/D/yjq/datasets/APY/APY_all/', type=str,
                        help='dataset path')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    return parser

class DataLoader(Dataset):
    def __init__(self, root, image_files, labels, transform=None):
        self.root  = root
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        # read the iterable image
        img_pil = Image.open(os.path.join(self.root, self.image_files[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)#3x224x224
        # label
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()# 将内部指针重置为指向数组的第一个元素

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def initMae(args):
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    checkpoint = torch.load(args.predict, map_location='cpu')
    checkpoint_model = checkpoint['model']
    checkpoint_model.pop('head.weight')
    checkpoint_model.pop('head.bias')
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)

    print(msg)
    return model



def train(model, data_loader, train_attrbs, optimizer, use_cuda, lamb_1=1.0):
    """returns trained model"""
    # initialize variables to monitor training and validation loss
    loss_meter = AverageMeter()
    """ train the model  """
    model.train()
    tk = tqdm(data_loader, total=int(len(data_loader)))  # 可扩展的Python进度条
    for batch_idx, (data, label) in enumerate(tk):
        # move to GPU
        if use_cuda:
            data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        # feat_g = model(data)
        x_g = model.vit(data)[0]  # 32x1024
        global feature
        feat_g = model.mlp_g(x_g)  # 32x312
        logit_g = feat_g @ train_attrbs.T  # 32x100
        loss = lamb_1 * F.cross_entropy(logit_g, label)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), label.shape[0])
        tk.set_postfix({"loss": loss_meter.avg})


    # print training/validation statistics

    print('Train: Average loss: {:.4f}\n'.format(loss_meter.avg))


def get_reprs(model, data_loader, use_cuda):
    model.eval()
    reprs = []
    for _, (data, _) in enumerate(data_loader):
        if use_cuda:
            data = data.cuda()  # 256X3X224X224
        with torch.no_grad():
            # only take the global feature
            feat = model.vit(data)[0]  # 256x1024
            feat = model.mlp_g(feat)  # 256x312
            # feat = model(data)
        reprs.append(feat.cpu().data.numpy())
    reprs = np.concatenate(reprs, 0)  # 1175x312
    return reprs


def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
    return np.mean(acc_per_class)


def validation(model, seen_loader, seen_labels, unseen_loader, unseen_labels, attrs_mat, use_cuda, gamma=None):
    # Representation
    with torch.no_grad():
        seen_reprs = get_reprs(model, seen_loader, use_cuda)  # 1175x312
        unseen_reprs = get_reprs(model, unseen_loader, use_cuda)

    # Labels
    uniq_labels = np.unique(np.concatenate([seen_labels, unseen_labels]))
    updated_seen_labels = np.searchsorted(uniq_labels, seen_labels)  # map操作
    uniq_updated_seen_labels = np.unique(updated_seen_labels)  # 100
    updated_unseen_labels = np.searchsorted(uniq_labels, unseen_labels)
    uniq_updated_unseen_labels = np.unique(updated_unseen_labels)  # 50
    uniq_updated_labels = np.unique(np.concatenate([updated_seen_labels, updated_unseen_labels]))  # 0-150

    # truncate the attribute matrix
    trunc_attrs_mat = attrs_mat[uniq_labels]  # 150x312

    #### ZSL ####
    zsl_unseen_sim = unseen_reprs @ trunc_attrs_mat[uniq_updated_unseen_labels].T  # 590x50
    pred_labels = np.argmax(zsl_unseen_sim, axis=1)  # 590
    zsl_unseen_predict_labels = uniq_updated_unseen_labels[pred_labels]
    zsl_unseen_acc = compute_accuracy(zsl_unseen_predict_labels, updated_unseen_labels, uniq_updated_unseen_labels)

    #### GZSL ####
    # seen classes
    gzsl_seen_sim = softmax(seen_reprs @ trunc_attrs_mat.T, axis=1)
    # unseen classes
    gzsl_unseen_sim = softmax(unseen_reprs @ trunc_attrs_mat.T, axis=1)

    gammas = np.arange(0.0, 1.1, 0.1)
    gamma_opt = 0
    H_max = 0
    gzsl_seen_acc_max = 0
    gzsl_unseen_acc_max = 0
    # Calibrated stacking
    for igamma in range(gammas.shape[0]):
        # Calibrated stacking
        gamma = gammas[igamma]
        gamma_mat = np.zeros(trunc_attrs_mat.shape[0])
        gamma_mat[uniq_updated_seen_labels] = gamma

        gzsl_seen_pred_labels = np.argmax(gzsl_seen_sim - gamma_mat, axis=1)
        # gzsl_seen_predict_labels = uniq_updated_labels[pred_seen_labels]
        gzsl_seen_acc = compute_accuracy(gzsl_seen_pred_labels, updated_seen_labels, uniq_updated_seen_labels)

        gzsl_unseen_pred_labels = np.argmax(gzsl_unseen_sim - gamma_mat, axis=1)
        # gzsl_unseen_predict_labels = uniq_updated_labels[pred_unseen_labels]
        gzsl_unseen_acc = compute_accuracy(gzsl_unseen_pred_labels, updated_unseen_labels, uniq_updated_unseen_labels)

        H = 2 * gzsl_seen_acc * gzsl_unseen_acc / (gzsl_seen_acc + gzsl_unseen_acc)

        if H > H_max:
            gzsl_seen_acc_max = gzsl_seen_acc
            gzsl_unseen_acc_max = gzsl_unseen_acc
            H_max = H
            gamma_opt = gamma

    print('ZSL: averaged per-class accuracy: {0:.2f}'.format(zsl_unseen_acc * 100))
    print('GZSL Seen: averaged per-class accuracy: {0:.2f}'.format(gzsl_seen_acc_max * 100))
    print('GZSL Unseen: averaged per-class accuracy: {0:.2f}'.format(gzsl_unseen_acc_max * 100))
    print('GZSL: harmonic mean (H): {0:.2f}'.format(H_max * 100))
    print('GZSL: gamma: {0:.2f}'.format(gamma_opt))

    return gamma_opt


def test(model, test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels, attrs_mat, use_cuda, gamma):
    # Representation
    with torch.no_grad():
        seen_reprs = get_reprs(model, test_seen_loader, use_cuda)  # 1764X312
        unseen_reprs = get_reprs(model, test_unseen_loader, use_cuda)  # 2967X312
    # Labels
    uniq_test_seen_labels = np.unique(test_seen_labels)  # 150
    uniq_test_unseen_labels = np.unique(test_unseen_labels)  # 50

    # ZSL
    zsl_unseen_sim = unseen_reprs @ attrs_mat[uniq_test_unseen_labels].T
    predict_labels = np.argmax(zsl_unseen_sim, axis=1)
    zsl_unseen_predict_labels = uniq_test_unseen_labels[predict_labels]
    zsl_unseen_acc = compute_accuracy(zsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

    # Calibrated stacking
    Cs_mat = np.zeros(attrs_mat.shape[0])
    Cs_mat[uniq_test_seen_labels] = gamma

    # GZSL
    # seen classes
    gzsl_seen_sim = softmax(seen_reprs @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_seen_predict_labels = np.argmax(gzsl_seen_sim, axis=1)
    gzsl_seen_acc = compute_accuracy(gzsl_seen_predict_labels, test_seen_labels, uniq_test_seen_labels)

    # unseen classes
    gzsl_unseen_sim = softmax(unseen_reprs @ attrs_mat.T, axis=1) - Cs_mat
    gzsl_unseen_predict_labels = np.argmax(gzsl_unseen_sim, axis=1)
    gzsl_unseen_acc = compute_accuracy(gzsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

    H = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc)

    print('ZSL: averaged per-class accuracy: {0:.2f}'.format(zsl_unseen_acc * 100))
    print('GZSL Seen: averaged per-class accuracy: {0:.2f}'.format(gzsl_seen_acc * 100))
    print('GZSL Unseen: averaged per-class accuracy: {0:.2f}'.format(gzsl_unseen_acc * 100))
    print('GZSL: harmonic mean (H): {0:.2f}'.format(H * 100))
    print('GZSL: gamma: {0:.2f}'.format(gamma))

def main(args):
    DATASET = 'SUN'  # ["AWA2", "CUB", "SUN", "APY", "FLO"]

    # Set Dataset Paths

    # In[ ]:

    if DATASET == 'AWA2':
        ROOT = '/home/kmyh/D/yjq/datasets/AWA2/Animals_with_Attributes2/JPEGImages/'
    elif DATASET == 'CUB':
        # ROOT='E:\python_project\datasets\CUB\CUB_200_2011\images'
        ROOT = '/home/kmyh/D/yjq/datasets/CUB/CUB_200_2011/CUB_200_2011/images/'
    elif DATASET == 'SUN':
        ROOT = '/home/kmyh/D/yjq/datasets/SUN/images/'
    elif DATASET == 'APY':
        ROOT = '/home/kmyh/D/yjq/datasets/APY/APY_all/'
    elif DATASET == 'FLO':
        ROOT = '/home/kmyh/D/yjq/datasets/FLO/102flowers/jpg/'
    else:
        print("Please specify the dataset")

    DATA_DIR = f'/home/kmyh/D/yjq/datasets/xlsa17/xlsa17/data/{DATASET}'
    data = sio.loadmat(f'{DATA_DIR}/res101.mat')
    # data consists of files names
    attrs_mat = sio.loadmat(f'{DATA_DIR}/att_splits.mat')
    # attrs_mat is the attributes (class-level information)
    image_files = data['image_files']

    if DATASET == 'AWA2':
        image_files = np.array([im_f[0][0].split('JPEGImages/')[-1] for im_f in image_files])

    if DATASET == 'FLO':
        image_files = np.array([im_f[0][0].split('jpg/')[-1] for im_f in image_files])

    if DATASET == 'APY':
        image_files = np.array([im_f[0][0].split('JPEGImages/')[-1] for im_f in image_files])
        image_files = np.append(image_files[:12695],
                                np.array([im_f.split('ayahoo_test_images/')[-1] for im_f in image_files[12695:]]))

    if DATASET == 'CUB' or DATASET == 'SUN':
        image_files = np.array([im_f[0][0].split('images/')[-1] for im_f in image_files])
    # labels are indexed from 1 as it was done in Matlab, so 1 subtracted for Python
    labels = data['labels'].squeeze().astype(np.int64) - 1
    train_idx = attrs_mat['train_loc'].squeeze() - 1
    val_idx = attrs_mat['val_loc'].squeeze() - 1
    trainval_idx = attrs_mat['trainval_loc'].squeeze() - 1
    test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
    test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1

    # consider the train_labels and val_labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    # split train_idx to train_idx (used for training) and val_seen_idx
    train_idx, val_seen_idx = train_test_split(train_idx, test_size=0.2, stratify=train_labels)
    # split val_idx to val_idx (not useful) and val_unseen_idx
    val_unseen_idx = train_test_split(val_idx, test_size=0.2, stratify=val_labels)[1]
    # attribute matrix
    attrs_mat = attrs_mat["att"].astype(np.float32).T

    ### used for validation
    # train files and labels
    train_files = image_files[train_idx]
    train_labels = labels[train_idx]
    uniq_train_labels, train_labels_based0, counts_train_labels = np.unique(train_labels, return_inverse=True,
                                                                            return_counts=True)
    # val seen files and labels
    val_seen_files = image_files[val_seen_idx]
    val_seen_labels = labels[val_seen_idx]
    uniq_val_seen_labels = np.unique(val_seen_labels)
    # val unseen files and labels
    val_unseen_files = image_files[val_unseen_idx]
    val_unseen_labels = labels[val_unseen_idx]
    uniq_val_unseen_labels = np.unique(val_unseen_labels)

    ### used for testing
    # trainval files and labels
    trainval_files = image_files[trainval_idx]
    trainval_labels = labels[trainval_idx]
    uniq_trainval_labels, trainval_labels_based0, counts_trainval_labels = np.unique(trainval_labels,
                                                                                     return_inverse=True,
                                                                                     return_counts=True)
    # test seen files and labels
    test_seen_files = image_files[test_seen_idx]
    test_seen_labels = labels[test_seen_idx]
    uniq_test_seen_labels = np.unique(test_seen_labels)
    # test unseen files and labels
    test_unseen_files = image_files[test_unseen_idx]
    test_unseen_labels = labels[test_unseen_idx]
    uniq_test_unseen_labels = np.unique(test_unseen_labels)



    # # Transformations

    # In[ ]:

    # Training Transformations
    trainTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    # Testing Transformations
    testTransform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # ### Average meter

    # In[ ]:

    num_workers = 0


    ### used in testing
    # trainval data loader
    trainval_data = DataLoader(ROOT, trainval_files, trainval_labels_based0, transform=trainTransform)
    weights_ = 1. / counts_trainval_labels
    weights = weights_[trainval_labels_based0]
    trainval_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=trainval_labels_based0.shape[0],
                                                              replacement=True)
    trainval_data_loader = torch.utils.data.DataLoader(trainval_data, batch_size=32, sampler=trainval_sampler,
                                                       num_workers=num_workers)
    # seen test data loader
    test_seen_data = DataLoader(ROOT, test_seen_files, test_seen_labels, transform=testTransform)
    test_seen_data_loader = torch.utils.data.DataLoader(test_seen_data, batch_size=256, shuffle=False,
                                                        num_workers=num_workers)
    # unseen test data loader
    test_unseen_data = DataLoader(ROOT, test_unseen_files, test_unseen_labels, transform=testTransform)
    test_unseen_data_loader = torch.utils.data.DataLoader(test_unseen_data, batch_size=256, shuffle=False,
                                                          num_workers=num_workers)
    if DATASET == 'AWA2':
        attr_length = 85
    elif DATASET == 'CUB':
        attr_length = 1024
    elif DATASET == 'SUN':
        attr_length = 102

    elif DATASET == 'APY':
        attr_length = 64

    elif DATASET == 'FLO':
        attr_length = 1024
    else:
        print("Please specify the dataset, and set {attr_length} equal to the attribute length")

    device = torch.device(args.device)
    # model = initMae(args).to(device)

    vit = initMae(args).to(device)
    mlp_g = nn.Linear(1024, attr_length, bias=False)  # 1024x312

    model = nn.ModuleDict({
        "vit": vit,
        "mlp_g": mlp_g})

    # finetune all the parameters
    for param in model.parameters():
        param.requires_grad = True

    # move model to GPU if CUDA is available
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam([{"params": model.vit.parameters(), "lr": 0.00001, "weight_decay": 0.0001},
                                  {"params": model.mlp_g.parameters(), "lr": 0.001, "weight_decay": 0.00001}])

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)  # 动态调整学习率


    # train attributes
    train_attrbs = attrs_mat[uniq_train_labels]  # 100x312
    train_attrbs_tensor = torch.from_numpy(train_attrbs)
    # trainval attributes
    trainval_attrbs = attrs_mat[uniq_trainval_labels]  # 150x312
    trainval_attrbs_tensor = torch.from_numpy(trainval_attrbs)
    if use_cuda:
        train_attrbs_tensor = train_attrbs_tensor.cuda()
        trainval_attrbs_tensor = trainval_attrbs_tensor.cuda()

    if DATASET == 'AWA2':
        gamma = 0.9
    elif DATASET == 'CUB':
        gamma = 0.9
    elif DATASET == 'SUN':
        gamma = 0.4
    elif DATASET == 'APY':
        gamma = 0.8
    elif DATASET == 'FLO':
        gamma = 0.9
    else:
        print("Please specify the dataset, and set {attr_length} equal to the attribute length")
    print('Dataset:', DATASET, '\nGamma:', gamma)


    for i in range(500):
        print('epoch:',format(i))
        train(model, trainval_data_loader, trainval_attrbs_tensor, optimizer, use_cuda, lamb_1=1.0)
        # print(' .... Saving model ...')
        # print('Epoch: ', i)
        # save_path = str(DATASET) + '__ViT-ZSL__' + 'Epoch_' + str(i) + '.pt'
        # ckpt_path = './checkpoint/' + str(DATASET)
        # path = os.path.join(ckpt_path, save_path)
        # torch.save(model.state_dict(), path)
        lr_scheduler.step()
        test(model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader, test_unseen_labels, attrs_mat,
             use_cuda, gamma)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    


