

import argparse
import os
import time as t
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from commons import * 
from modules import fpn 
from distributed import Client
from dask_cuda import LocalCUDACluster
from copy import copy


# In[8]:


def train_supervised(args, logger, dataloader, model, classifier1, classifier2, criterion1, criterion2, optimizer, epoch):
    losses = AverageMeter()
    losses_mse = AverageMeter()
    losses_cet = AverageMeter()
    losses_cet_across = AverageMeter()
    losses_cet_within = AverageMeter()

    # switch to train mode
    model.train()
    if args.mse:
        criterion_mse = torch.nn.MSELoss().cuda()

    classifier1.eval()
    classifier2.eval()
    for i, (indice, input1, label1) in enumerate(dataloader):
        input1 = eqv_transform_if_needed(args, dataloader, indice, input1.cuda(non_blocking=True))
        label1 = label1.cuda(non_blocking=True)
        featmap1 = model(input1)
        featmap1 = F.interpolate(featmap1, label1.shape[-2:], mode='bilinear', align_corners=False)
        
        # input2 = input2.cuda(non_blocking=True)
        # label2 = label2.cuda(non_blocking=True)
        # featmap2 = eqv_transform_if_needed(args, dataloader, indice, model(input2))

        B, C, _ = featmap1.size()[:3]
        if i == 0:
            logger.info('Batch input size   : {}'.format(list(input1.shape)))
            logger.info('Batch label size   : {}'.format(list(label1.shape)))
            logger.info('Batch feature size : {}\n'.format(list(featmap1.shape)))
        
        if args.metric_train == 'cosine':
            featmap1 = F.normalize(featmap1, dim=1, p=2)
            # featmap2 = F.normalize(featmap2, dim=1, p=2)

        featmap12_processed, label12_processed = featmap1, label1.flatten()
        # featmap21_processed, label21_processed = featmap2, label1.flatten()

        # Cross-view loss
        output12 = feature_flatten(classifier2(featmap12_processed)) # NOTE: classifier2 is coupled with label2
        # output21 = feature_flatten(classifier1(featmap21_processed)) # NOTE: classifier1 is coupled with label1
        
        loss12  = criterion2(output12, label12_processed)
        # loss21  = criterion1(output21, label21_processed)  

        # loss_across = (loss12 + loss21) / 2.
        loss_across = loss12 
        losses_cet_across.update(loss_across.item(), B)

        featmap11_processed, label11_processed = featmap1, label1.flatten()
        # featmap22_processed, label22_processed = featmap2, label2.flatten()
        
        # Within-view loss
        output11 = feature_flatten(classifier1(featmap11_processed)) # NOTE: classifier1 is coupled with label1
        # output22 = feature_flatten(classifier2(featmap22_processed)) # NOTE: classifier2 is coupled with label2

        loss11 = criterion1(output11, label11_processed)
        # loss22 = criterion2(output22, label22_processed)

        loss_within = loss11
        # loss_within = (loss11 + loss22) / 2. 
        losses_cet_within.update(loss_within.item(), B)
        loss = (loss_across + loss_within) / 2.
        
        losses_cet.update(loss.item(), B)
        
        if args.mse:
            loss_mse = criterion_mse(featmap1, featmap2)
            losses_mse.update(loss_mse.item(), B)

            loss = (loss + loss_mse) / 2. 
        
        # record loss
        losses.update(loss.item(), B)

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % 200) == 0:
            logger.info('{0} / {1}\t'.format(i, len(dataloader)))

    return losses.avg, losses_cet.avg, losses_cet_within.avg, losses_cet_across.avg, losses_mse.avg



def train(args, logger, dataloader, model, classifier1, classifier2, criterion1, criterion2, optimizer, epoch):
    losses = AverageMeter()
    losses_mse = AverageMeter()
    losses_cet = AverageMeter()
    losses_cet_across = AverageMeter()
    losses_cet_within = AverageMeter()

    # switch to train mode
    model.train()
    if args.mse:
        criterion_mse = torch.nn.MSELoss().cuda()

    classifier1.eval()
    classifier2.eval()
    for i, (indice, input1, input2, label1, label2) in enumerate(dataloader):
        input1 = eqv_transform_if_needed(args, dataloader, indice, input1.cuda(non_blocking=True))
        label1 = label1.cuda(non_blocking=True)
        featmap1 = model(input1)
        
        input2 = input2.cuda(non_blocking=True)
        label2 = label2.cuda(non_blocking=True)
        featmap2 = eqv_transform_if_needed(args, dataloader, indice, model(input2))

        B, C, _ = featmap1.size()[:3]
        if i == 0:
            logger.info('Batch input size   : {}'.format(list(input1.shape)))
            logger.info('Batch label size   : {}'.format(list(label1.shape)))
            logger.info('Batch feature size : {}\n'.format(list(featmap1.shape)))
        
        if args.metric_train == 'cosine':
            featmap1 = F.normalize(featmap1, dim=1, p=2)
            featmap2 = F.normalize(featmap2, dim=1, p=2)

        featmap12_processed, label12_processed = featmap1, label2.flatten()
        featmap21_processed, label21_processed = featmap2, label1.flatten()

        # Cross-view loss
        output12 = feature_flatten(classifier2(featmap12_processed)) # NOTE: classifier2 is coupled with label2
        output21 = feature_flatten(classifier1(featmap21_processed)) # NOTE: classifier1 is coupled with label1
        
        loss12  = criterion2(output12, label12_processed)
        loss21  = criterion1(output21, label21_processed)  

        loss_across = (loss12 + loss21) / 2.
        losses_cet_across.update(loss_across.item(), B)

        featmap11_processed, label11_processed = featmap1, label1.flatten()
        featmap22_processed, label22_processed = featmap2, label2.flatten()
        
        # Within-view loss
        output11 = feature_flatten(classifier1(featmap11_processed)) # NOTE: classifier1 is coupled with label1
        output22 = feature_flatten(classifier2(featmap22_processed)) # NOTE: classifier2 is coupled with label2

        loss11 = criterion1(output11, label11_processed)
        loss22 = criterion2(output22, label22_processed)

        loss_within = (loss11 + loss22) / 2. 
        losses_cet_within.update(loss_within.item(), B)
        loss = (loss_across + loss_within) / 2.
        
        losses_cet.update(loss.item(), B)
        
        if args.mse:
            loss_mse = criterion_mse(featmap1, featmap2)
            losses_mse.update(loss_mse.item(), B)

            loss = (loss + loss_mse) / 2. 
        
        # record loss
        losses.update(loss.item(), B)

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % 200) == 0:
            logger.info('{0} / {1}\t'.format(i, len(dataloader)))

    return losses.avg, losses_cet.avg, losses_cet_within.avg, losses_cet_across.avg, losses_mse.avg




# In[9]:


def main(args, logger):
    logger.info(args)

    # Use random seed.
    fix_seed_for_reproducability(args.seed)

    # Start time.
    t_start = t.time()

    # Get model and optimizer.
    model, optimizer, classifier1 = get_model_and_optimizer(args, logger)

    # New trainset inside for-loop.
    inv_list, eqv_list = get_transform_params(args)
    trainset = get_dataset(args, mode='train', inv_list=inv_list, eqv_list=eqv_list)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=args.batch_size_cluster,
                                            shuffle=False, 
                                            num_workers=args.num_workers,
                                            pin_memory=False,
                                            collate_fn=collate_train,
                                            worker_init_fn=worker_init_fn(args.seed))
    
    
    # if True:
    #     trainset_supervised = get_dataset(args, mode='supervised_train', inv_list=inv_list, eqv_list=eqv_list)
    #     trainloader_supervised = torch.utils.data.DataLoader(trainset_supervised, 
    #                                             batch_size=args.batch_size_train,
    #                                             shuffle=False, 
    #                                             num_workers=args.num_workers,
    #                                             pin_memory=False,
    #                                             collate_fn=collate_train,
    #                                             worker_init_fn=worker_init_fn(args.seed))
        
    
    testset    = get_dataset(args, mode='train_val')
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=False,
                                             collate_fn=collate_eval,
                                             worker_init_fn=worker_init_fn(args.seed))
    
    # Before train.
    # _, _ = evaluate(args, logger, testloader, classifier1, model)
    
    if not args.eval_only:
        # Train start.
        for epoch in range(args.start_epoch, args.num_epoch):
            # Assign probs. 
            trainloader.dataset.mode = 'compute'
            trainloader.dataset.reshuffle()

            # Adjust lr if needed. 
            # adjust_learning_rate(optimizer, epoch, args)
            classifier1 = initialize_classifier(args)
            classifier2 = initialize_classifier(args)
            if epoch == 0 or epoch % 4 != 0 :
                logger.info('\n============================= [Epoch {}] =============================\n'.format(epoch))
                logger.info('Start computing centroids.')
                t1 = t.time()
                centroids1, kmloss1 = run_mini_batch_kmeans(args, logger, trainloader, model, view=1)
                centroids2, kmloss2 = run_mini_batch_kmeans(args, logger, trainloader, model, view=2)
                logger.info('-Centroids ready. [Loss: {:.5f}| {:.5f}/ Time: {}]\n'.format(kmloss1, kmloss2, get_datetime(int(t.time())-int(t1))))
                
                # Compute cluster assignment. 
                t2 = t.time()
                weight1 = compute_labels(args, logger, trainloader, model, centroids1, view=1)
                weight2 = compute_labels(args, logger, trainloader, model, centroids2, view=2)
                logger.info('-Cluster labels ready. [{}]\n'.format(get_datetime(int(t.time())-int(t2)))) 
                
                # Criterion.
                if not args.no_balance:
                    criterion1 = torch.nn.CrossEntropyLoss(weight=weight1).cuda()
                    criterion2 = torch.nn.CrossEntropyLoss(weight=weight2).cuda()
                else:
                    criterion1 = torch.nn.CrossEntropyLoss().cuda()
                    criterion2 = torch.nn.CrossEntropyLoss().cuda()

            # Setup nonparametric classifier.
                classifier1.module.weight.data = centroids1.unsqueeze(-1).unsqueeze(-1)
                classifier2.module.weight.data = centroids2.unsqueeze(-1).unsqueeze(-1)
                del centroids1 
                del centroids2
                # Set-up train loader.
                trainset.mode  = 'train'
                trainloader_loop  = torch.utils.data.DataLoader(trainset, 
                                                                batch_size=args.batch_size_train, 
                                                                shuffle=True,
                                                                num_workers=args.num_workers,
                                                                pin_memory=False,
                                                                collate_fn=collate_train,
                                                                worker_init_fn=worker_init_fn(args.seed))
                
            freeze_all(classifier1)
            freeze_all(classifier2)


            
            if epoch != 0 and epoch % 4 == 0:
                
                if not args.no_balance:
                    criterion1 = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda()
                    criterion2 = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda()
                else:
                    criterion1 = torch.nn.CrossEntropyLoss().cuda()
                    criterion2 = torch.nn.CrossEntropyLoss().cuda()
                
                logger.info('Start supervised training ...')
                trainset_supervised = get_dataset(args, mode='supervised_train', inv_list=inv_list, eqv_list=eqv_list)
                trainloader_supervised = torch.utils.data.DataLoader(trainset_supervised, 
                                                batch_size=args.batch_size_train,
                                                shuffle=False, 
                                                num_workers=args.num_workers,
                                                pin_memory=False,
                                                collate_fn=collate_eval,
                                                worker_init_fn=worker_init_fn(args.seed))
                
                train_loss, train_cet, cet_within, cet_across, train_mse = train_supervised(args, logger, trainloader_supervised, model, classifier1, classifier2, criterion1, criterion2, optimizer, epoch)
            else:
                logger.info('Start training ...')
                train_loss, train_cet, cet_within, cet_across, train_mse = train(args, logger, trainloader_loop, model, classifier1, classifier2, criterion1, criterion2, optimizer, epoch) 
            
            acc1, res1 = evaluate(args, logger, testloader, classifier1, model)
            acc2, res2 = evaluate(args, logger, testloader, classifier2, model)
            
            logger.info('============== Epoch [{}] =============='.format(epoch))
            # logger.info('  Time: [{}]'.format(get_datetime(int(t.time())-int(t1))))
            # logger.info('  K-Means loss   : {:.5f} | {:.5f}'.format(kmloss1, kmloss2))
            logger.info('  Training Total Loss  : {:.5f}'.format(train_loss))
            logger.info('  Training CE Loss (Total | Within | Across) : {:.5f} | {:.5f} | {:.5f}'.format(train_cet, cet_within, cet_across))
            logger.info('  Training MSE Loss (Total) : {:.5f}'.format(train_mse))
            logger.info('  [View 1] ACC: {:.4f} | mIoU: {:.4f}'.format(acc1, res1['mean_iou']))
            logger.info('  [View 2] ACC: {:.4f} | mIoU: {:.4f}'.format(acc2, res2['mean_iou']))
            logger.info('========================================\n')
            
            saving_args = copy(args)
            del saving_args.client
            torch.save({'epoch': epoch+1, 
                        'args' : saving_args,
                        'state_dict': model.state_dict(),
                        'classifier1_state_dict' : classifier1.state_dict(),
                        'classifier2_state_dict' : classifier2.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        },
                        os.path.join(args.save_model_path, 'checkpoint_{}.pth.tar'.format(epoch)))
            
            torch.save({'epoch': epoch+1, 
                        'args' : saving_args,
                        'state_dict': model.state_dict(),
                        'classifier1_state_dict' : classifier1.state_dict(),
                        'classifier2_state_dict' : classifier2.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        },
                        os.path.join(args.save_model_path, 'checkpoint.pth.tar'))
        
        # Evaluate.
        trainset    = get_dataset(args, mode='eval_val')
        trainloader = torch.utils.data.DataLoader(trainset, 
                                                    batch_size=args.batch_size_cluster,
                                                    shuffle=True,
                                                    num_workers=args.num_workers,
                                                    pin_memory=False,
                                                    collate_fn=collate_train,
                                                    worker_init_fn=worker_init_fn(args.seed))

        testset    = get_dataset(args, mode='eval_test')
        testloader = torch.utils.data.DataLoader(testset, 
                                                batch_size=args.batch_size_test,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                pin_memory=False,
                                                collate_fn=collate_eval,
                                                worker_init_fn=worker_init_fn(args.seed))

        # Evaluate with fresh clusters.
        acc_list_new = []  
        res_list_new = []                 
        logger.info('Start computing centroids.')
        if args.repeats > 0:
            for _ in range(args.repeats):
                t1 = t.time()
                centroids1, kmloss1 = run_mini_batch_kmeans(args, logger, trainloader, model, view=-1)
                logger.info('-Centroids ready. [Loss: {:.5f}/ Time: {}]\n'.format(kmloss1, get_datetime(int(t.time())-int(t1))))
                
                classifier1 = initialize_classifier(args)
                classifier1.module.weight.data = centroids1.unsqueeze(-1).unsqueeze(-1)
                freeze_all(classifier1)
                
                acc_new, res_new = evaluate(args, logger, testloader, classifier1, model)
                acc_list_new.append(acc_new)
                res_list_new.append(res_new)
        else:
            acc_new, res_new = evaluate(args, logger, testloader, classifier1, model)
            acc_list_new.append(acc_new)
            res_list_new.append(res_new)

        logger.info('Average overall pixel accuracy [NEW] : {:.3f} +/- {:.3f}.'.format(np.mean(acc_list_new), np.std(acc_list_new)))
        logger.info('Average mIoU [NEW] : {:.3f} +/- {:.3f}. '.format(np.mean([res['mean_iou'] for res in res_list_new]), 
                                                                    np.std([res['mean_iou'] for res in res_list_new])))
        logger.info('Experiment done. [{}]\n'.format(get_datetime(int(t.time())-int(t_start))))


# In[12]:


class Arguments:
    def __init__(self, 
                data_root='../../Data/coco',
                supervised_data_root = '../../Data/coco_supervisedset',
                save_root='results',
                restart_path='',
                seed=1,
                num_workers=6,
                restart=True,
                num_epoch=100,
                repeats=1,
                arch='resnet18',
                pretrain=True,
                res=320,
                res1=320,
                res2=640,
                batch_size_cluster=4,
                batch_size_train=32,
                batch_size_test=8,
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
                eval_path='results',
                save_model_path='K_train',
                save_eval_path='K_test',
                cityscapes=False,
                faiss_gpu_id=1
                ):

        self.data_root=data_root
        self.supervised_data_root=supervised_data_root
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


args = Arguments()


# In[13]:


if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=[0,1,2,3],
        )
    
    client = Client(cluster)    
    args.client = client
    main(args, logger)
    client.close()

# In[ ]:




