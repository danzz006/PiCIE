import os 
import sys 
import argparse 
import logging
import time as t 
import numpy as np 
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import fpn
from commons import *
from utils import *
from train_picie import * 

def compute_dist(featmap, metric_function, euclidean_train=True):
    centroids = metric_function.module.weight.data
    if euclidean_train:
        return - (1 - 2*metric_function(featmap)\
                    + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 
    else:
        return metric_function(featmap)


def compute_histogram(args, dataloader, model, classifier):
    histogram = np.zeros((args.K_test, args.K_test))

    model.eval()
    classifier.eval()
    with torch.no_grad():
        for i, (indice, image, label) in enumerate(dataloader):
            image = image.cuda(non_blocking=True)
            feats = model(image)
            feats = F.normalize(feats, dim=1, p=2)

            if i == 0:
                print('Batch image size   : {}'.format(image.size()), flush=True)
                print('Batch label size   : {}'.format(label.size()), flush=True)
                print('Batch feature size : {}\n'.format(feats.size()), flush=True)
            
            probs = compute_dist(feats, classifier)
            probs = F.interpolate(probs, args.res1, mode='bilinear', align_corners=False)
            preds = probs.topk(1, dim=1)[1].view(probs.size(0), -1).cpu().numpy()
            label = label.view(probs.size(0), -1).cpu().numpy()

            histogram += scores(label, preds, args.K_test)
            
    return histogram

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
                K_test=27,
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
                eval_path='K_train/checkpoint_40.pth.tar',
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
    # args = parse_arguments()
    args = Arguments()
    
    # Use random seed.
    fix_seed_for_reproducability(args.seed)

    # Init model. 
    model = fpn.PanopticFPN(args)
    # model = nn.DataParallel(model)
    model = model.cuda()

    # Init classifier (for eval only.)
    classifier = initialize_classifier(args)

    # Load weights.
    checkpoint = torch.load(args.eval_path)
    
    ckpt = {}
    for k in checkpoint:
        if type(checkpoint[k]) == collections.OrderedDict:
            ckpt[k] = {}
            for j in checkpoint[k]:
                if "module" in j:
                    ckpt[k][j[7:]] = checkpoint[k][j]
                else:
                    ckpt[k][j] = checkpoint[k][j]
        else:
            ckpt[k] = checkpoint[k]
    
    
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(ckpt['state_dict'])
    classifier.load_state_dict(checkpoint['classifier1_state_dict'])

    # Prepare dataloader.
    dataset    = get_dataset(args, mode='eval_test')
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=collate_eval,
                                             worker_init_fn=worker_init_fn(args.seed))

    # Compute statistics.
    histogram = compute_histogram(args, dataloader, model, classifier)

    # Save the result. 
    torch.save(histogram, args.save_root + '/picie_histogram_coco.pkl')
    print('-Done.', flush=True)

