import os 
import sys 
import argparse 
import logging
import time as t 
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import fpn
from commons import *
from utils import *
from train_picie import *




def initialize_classifier(args, n_query, centroids):
    classifier = nn.Conv2d(args.in_dim, n_query, kernel_size=1, stride=1, padding=0, bias=False)
    classifier = nn.DataParallel(classifier)
    classifier = classifier.cuda()
    if centroids is not None:
        classifier.module.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
    freeze_all(classifier)

    return classifier


def get_testloader(args):
    testset    = EvalDataset(args.data_root, dataset=args.dataset, res=args.res1, split=args.val_type, mode='test', stuff=args.stuff, thing=args.thing)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=args.batch_size_eval,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=collate_eval)

    return testloader


def compute_dist(featmap, metric_function, euclidean_train=True):
    centroids = metric_function.module.weight.data
    if euclidean_train:
        return - (1 - 2*metric_function(featmap)\
                    + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 
    else:
        return metric_function(featmap)


def get_nearest_neighbors(n_query, dataloader, model, classifier, k=10):
    model.eval()
    classifier.eval()

    min_dsts = [[] for _ in range(n_query)]
    min_locs = [[] for _ in range(n_query)]
    min_imgs = [[] for _ in range(n_query)]
    with torch.no_grad():
        for indice, image, label in dataloader:
            image = image.cuda(non_blocking=True)
            feats = model(image)
            feats = F.normalize(feats, dim=1, p=2)
            dists = compute_dist(feats.cpu(), classifier) # (B x C x H x W)
            dists = dists.cpu()
            B, _, H, W = dists.shape
            for c in range(n_query):
                dst, idx = dists[:, c].flatten().topk(1)

                idx = idx.item()
                ib = idx//(H*W)
                ih = idx%(H*W)//W 
                iw = idx%(H*W)%W
                if len(min_dsts[c]) < k:
                    min_dsts[c].append(dst.cpu())
                    min_locs[c].append((ib, ih, iw))
                    min_imgs[c].append(indice[ib])
                    
                    # for i in range(len(min_dsts[c])):
                    #     min_dsts[c][i] = min_dsts[c][i].cpu()
                elif dst.cpu() < max(min_dsts[c]):
                    
                    # for i in range(len(min_dsts[c])):
                    #     min_dsts[c][i] = min_dsts[c][i].cpu()
                    # min_dsts[c][-1] = min_dsts[c][-1].cpu()
                    
                    imax = np.argmax(min_dsts[c])

                    min_dsts[c] = min_dsts[c][:imax] + min_dsts[c][imax+1:]
                    min_locs[c] = min_locs[c][:imax] + min_locs[c][imax+1:]
                    min_imgs[c] = min_imgs[c][:imax] + min_imgs[c][imax+1:]

                    min_dsts[c].append(dst)
                    min_locs[c].append((ib, ih, iw))
                    min_imgs[c].append(indice[ib])
                
    loclist = min_locs 
    dataset = dataloader.dataset
    imglist = [[dataset.transform_data(*dataset.load_data(dataset.imdb[i]), i, raw_image=True) for i in ids] for ids in min_imgs]
    return imglist, loclist

class Arguments:
    def __init__(self, 
                data_root='../../Data/coco',
                save_root='results',
                restart_path='',
                seed=1,
                num_workers=6,
                restart=True,
                num_epoch=10,
                repeats=1,
                arch='resnet18',
                pretrain=True,
                res=320,
                res1=320,
                res2=640,
                batch_size_cluster=4,
                batch_size_train=128,
                batch_size_test=64,
                lr=1e-4,
                weight_decay=0,
                momentum=0.9,
                optim_type='Adam',
                num_init_batches=3,
                num_batches=3,
                kmeans_n_iter=30,
                in_dim=128,
                X=80,
                metric_train='cosine',
                metric_test='cosine',
                K_train=27,
                K_test=5,
                no_balance=False,
                mse=False,
                augment=False,
                equiv=False,
                min_scale=0.5,
                stuff=True,
                thing=True,
                jitter=False,
                grey=False,
                blur=False,
                h_flip=False,
                v_flip=False,
                random_crop=False,
                val_type='train',
                version=7,
                fullcoco=False,
                eval_only=False,
                eval_path='K_train/checkpoint.pth.tar',
                save_model_path='K_train',
                save_eval_path='K_test',
                cityscapes=False,
                faiss_gpu_id=1
                ):

        self.data_root=data_root
        self.save_root=save_root
        self.restart_path=restart_path
        self.seed=seed
        self.num_workers=num_workers
        self.restart=restart
        self.num_epoch=num_epoch
        self.repeats=repeats
        self.arch=arch
        self.pretrain=pretrain
        self.res=res
        self.res1=res1
        self.res2=res2
        self.batch_size_cluster=batch_size_cluster
        self.batch_size_train=batch_size_train
        self.batch_size_test=batch_size_test
        self.lr=lr
        self.weight_decay=weight_decay
        self.momentum=momentum
        self.optim_type=optim_type
        self.num_init_batches=num_init_batches
        self.num_batches=num_batches
        self.kmeans_n_iter=kmeans_n_iter
        self.in_dim=in_dim
        self.X=X
        self.metric_train=metric_train
        self.metric_test=metric_test
        self.K_train=K_train
        self.K_test=K_test
        self.no_balance=no_balance
        self.mse=mse
        self.augment=augment
        self.equiv=equiv
        self.min_scale=min_scale
        self.stuff=stuff
        self.thing=thing
        self.jitter=jitter
        self.grey=grey
        self.blur=blur
        self.h_flip=h_flip
        self.v_flip=v_flip
        self.random_crop=random_crop
        self.val_type=val_type
        self.cityscapes = cityscapes
        self.version=version
        self.fullcoco=fullcoco
        self.eval_only=eval_only
        self.eval_path=eval_path
        self.save_eval_path = save_eval_path
        self.save_model_path = save_model_path
        self.faiss_gpu_id = faiss_gpu_id


if __name__ == '__main__':
    args = Arguments()
    # args = parse_arguments()
    
    # Use random seed.
    fix_seed_for_reproducability(args.seed)

    # Init model. 
    model = fpn.PanopticFPN(args)
    model = nn.DataParallel(model)
    model = model.cuda()

    # Load weights.
    checkpoint = torch.load(args.eval_path) 
    model.load_state_dict(checkpoint['state_dict'])
    
    # Init classifier (for eval only.)
    queries = torch.tensor(np.load('querys.npy')).cuda()
    classifier = initialize_classifier(args, queries.size(0), queries)

    # Prepare dataloader.
    dataset    = get_dataset(args, mode='eval_test')
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=collate_eval,
                                             worker_init_fn=worker_init_fn(args.seed))

    # Retrieve 10-nearest neighbors.
    imglist, loclist = get_nearest_neighbors(queries.size(0), dataloader, model, classifier, k=args.K_test) 

    # Save the result. 
    torch.save([imglist, loclist], args.save_root + '/picie_retrieval_result_coco.pkl')
    print('-Done.', flush=True)

