from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import math
import time
import random
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util.icbhi_dataset import ICBHIDataset, ICBHIPediatricsDataset, ICBHINonPediatricsDataset

from util.snu_dataset import SNUBHDataset
from util.smart_dataset import SMARTDataset
from util.multi_dataset import ICBHISNUBHDataset, ICBHISMARTDataset, SNUBHSMARTDataset, ALLDataset


#from util._dataset import BothDataset
from util.icbhi_util import get_score
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json, save_model_multitask, save_model_multitask_domain
from models import get_backbone_class, Projector
from method import MetaCL, PatchMixLoss, PatchMixConLoss, polarization, equalization, equalization_ver2, AMSoftmaxLoss, CrossEntropyLabelSmooth, AM_Softmax_v2, ArcMarginProduct, AAMSoftmaxLoss
import torch.nn.functional as F
from torch.nn import Parameter
from parser import parse_args



def set_loader(args):
    args.h = int(args.desired_length * 100 - 2)
    args.w = 128
    train_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
    val_transform = [transforms.ToTensor(),
                    transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose(val_transform)
    
    if args.dataset == 'icbhi':
        if not args.eval:
            train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        
        if args.eval_pediatrics:
            val_dataset = ICBHIPediatricsDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        elif args.eval_non_pediatrics:
            val_dataset = ICBHINonPediatricsDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        else:
            val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
            
    elif args.dataset == 'snubh':
        if not args.eval:
            train_dataset = SNUBHDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset = SNUBHDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
    
    elif args.dataset == 'smart':
        if not args.eval:
            train_dataset = SMARTDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset = SMARTDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
    
    elif args.dataset == 'icbhi_snubh':
        if not args.eval:
            train_dataset = ICBHISNUBHDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset_snubh = SNUBHDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_icbhi_child = ICBHIPediatricsDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_icbhi_nonchild = ICBHINonPediatricsDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_icbhi = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        
    
    elif args.dataset == 'icbhi_smart':
        if not args.eval:
            train_dataset = ICBHISMARTDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset_smart = SMARTDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_icbhi_child = ICBHIPediatricsDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_icbhi_nonchild = ICBHINonPediatricsDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_icbhi = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        
    elif args.dataset == 'snubh_smart':
        if not args.eval:
            train_dataset = SNUBHSMARTDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset_snubh = SNUBHDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_smart = SMARTDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
    
    elif args.dataset == 'all':
        if not args.eval:
            train_dataset = ALLDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
        val_dataset_snubh = SNUBHDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_smart = SMARTDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_icbhi_child = ICBHIPediatricsDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_icbhi_nonchild = ICBHINonPediatricsDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        val_dataset_icbhi = ICBHIDataset(train_flag=False, transform=val_transform, args=args, print_flag=True)
        
    if args.multitask or args.multitask_domain:
        args.class_nums = train_dataset.lung_class_nums
        args.disease_nums = train_dataset.disease_class_nums
    else:
        args.class_nums = train_dataset.class_nums if not args.eval else None
    '''
    if args.domain_adaptation or args.domain_adaptation2:
        args.domain_nums = train_dataset.domain_nums
    '''        
    
    if not args.eval:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    if args.dataset in ['icbhi', 'snubh', 'smart']:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader if not args.eval else None, val_loader, args
    
    elif args.dataset == 'icbhi_snubh':
        val_loader_snubh = torch.utils.data.DataLoader(val_dataset_snubh, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_icbhi_child = torch.utils.data.DataLoader(val_dataset_icbhi_child, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_icbhi_nonchild = torch.utils.data.DataLoader(val_dataset_icbhi_nonchild, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_icbhi = torch.utils.data.DataLoader(val_dataset_icbhi, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        return train_loader if not args.eval else None, (val_loader_icbhi, val_loader_snubh, val_loader_icbhi_child, val_loader_icbhi_nonchild), args
    
    elif args.dataset == 'icbhi_smart':
        val_loader_smart = torch.utils.data.DataLoader(val_dataset_smart, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_icbhi_child = torch.utils.data.DataLoader(val_dataset_icbhi_child, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_icbhi_nonchild = torch.utils.data.DataLoader(val_dataset_icbhi_nonchild, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_icbhi = torch.utils.data.DataLoader(val_dataset_icbhi, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        return train_loader if not args.eval else None, (val_loader_icbhi, val_loader_smart, val_loader_icbhi_child, val_loader_icbhi_nonchild), args
    
    elif args.dataset == 'snubh_smart':
        val_loader_snubh = torch.utils.data.DataLoader(val_dataset_snubh, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_smart = torch.utils.data.DataLoader(val_dataset_smart, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        return train_loader if not args.eval else None, (val_loader_snubh, val_loader_smart), args
            
    elif args.dataset == 'all':
        val_loader_snubh = torch.utils.data.DataLoader(val_dataset_snubh, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_smart = torch.utils.data.DataLoader(val_dataset_smart, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_icbhi_child = torch.utils.data.DataLoader(val_dataset_icbhi_child, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_icbhi_nonchild = torch.utils.data.DataLoader(val_dataset_icbhi_nonchild, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_loader_icbhi = torch.utils.data.DataLoader(val_dataset_icbhi, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        return train_loader if not args.eval else None, (val_loader_icbhi, val_loader_snubh, val_loader_smart, val_loader_icbhi_child, val_loader_icbhi_nonchild), args
    
    else:
        raise NotImplemented
        



def set_model(args):
    kwargs = {}
    if args.model == 'ast':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        #kwargs['label_dim'] = args.n_cls
        kwargs['label_dim'] = args.lung_cls if args.multitask or args.multitask_domain else args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        if args.domain_adaptation:
            kwargs['domain_label_dim'] = args.m_cls
        if args.multitask:
            kwargs['multitask_label_dim'] = args.disease_cls
        elif args.multitask_domain:
            kwargs['multitask_label_dim'] = args.disease_cls
            kwargs['multitask_domain_label_dim'] = args.domain_cls

    model = get_backbone_class(args.model)(**kwargs)
    if args.domain_adaptation:
        class_classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
        domain_classifier = nn.Linear(model.final_feat_dim, args.meta_all_cls if args.meta_all else args.m_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.domain_mlp_head)
    elif args.domain_adaptation2:
        class_classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
        domain_classifier = Projector(model.final_feat_dim, args.proj_dim)
    elif args.multitask:
        lung_classifier = nn.Linear(model.final_feat_dim, args.lung_cls) if args.model not in ['ast'] else deepcopy(model.mlp_head)
        disease_classifier = nn.Linear(model.final_feat_dim, args.disease_cls) if args.model not in ['ast'] else deepcopy(model.multitask_mlp_head)
    elif args.multitask_domain:
        lung_classifier = nn.Linear(model.final_feat_dim, args.lung_cls) if args.model not in ['ast'] else deepcopy(model.mlp_head)
        disease_classifier = nn.Linear(model.final_feat_dim, args.disease_cls) if args.model not in ['ast'] else deepcopy(model.multitask_mlp_head)
        domain_classifier = nn.Linear(model.final_feat_dim, args.domain_cls) if args.model not in ['ast'] else deepcopy(model.multitask_domain_mlp_head)
    else:
        classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
    projector = Projector(model.final_feat_dim, args.proj_dim) if args.method in ['patchmix_cl'] or args.domain_adaptation2 else nn.Identity()
    
      
    
    
    if args.weighted_loss:
        weights = torch.tensor(args.class_nums, dtype=torch.float32)
        weights = 1.0 / (weights / weights.sum())
        weights /= weights.sum()
        
        criterion = nn.CrossEntropyLoss(weight=weights)    
    else:
        weights = None
        criterion = nn.CrossEntropyLoss()
    
    if args.multitask: ## check
        if args.weighted_loss_diagnosis_only:
            weights = torch.tensor(args.disease_nums, dtype=torch.float32)
            weights = 1.0 / (weights / weights.sum())
            weights /= weights.sum()        
            criterion2 = nn.CrossEntropyLoss(weight=weights)
            print('weighted_loss_diagnosis_only')
        else:
            weights = None
            criterion2 = nn.CrossEntropyLoss()
            print('no weighted_loss_diagnosis_only')
    
    elif args.multitask_domain:
        if args.weighted_loss_diagnosis_only:
            weights = torch.tensor(args.disease_nums, dtype=torch.float32)
            weights = 1.0 / (weights / weights.sum())
            weights /= weights.sum()        
            criterion2 = nn.CrossEntropyLoss(weight=weights)
            print('weighted_loss_diagnosis_only')
        else:
            weights = None
            criterion2 = nn.CrossEntropyLoss()
            print('no weighted_loss_diagnosis_only')
        
        if args.weighted_loss_domain_only:
            weights = torch.tensor(args.meta_nums, dtype=torch.float32)
            weights = 1.0 / (weights / weights.sum())
            weights /= weights.sum()        
            criterion3 = nn.CrossEntropyLoss(weight=weights)
            print('weighted_loss_domain_only')
        else:
            weights = None
            criterion3 = nn.CrossEntropyLoss()
            print('no weighted_loss_domain_only')
    
    
        
    if args.domain_adaptation:
        if args.meta_all:
            if args.meta_weights:
                #meta_weights = [0.6294, 0.3706, 0.5478, 0.4522, 0.1702, 0.1014, 0.0070, 0.4285, 0.2929, 0.0232, 0.2089, 0.1917, 0.1233, 0.0843, 0.2083, 0.1603, 0.7071, 0.2929]
                #meta_weights = torch.tensor([0.6294, 0.3706, 0.5478, 0.4522, 0.1702, 0.1014, 0.0070, 0.4285, 0.2929, 0.0232, 0.2089, 0.1917, 0.1233, 0.0843, 0.2083, 0.1603, 0.7071, 0.2929], dtype=torch.float32)
                meta_weights = torch.tensor([0.3706, 0.6294, 0.4522, 0.5478, 0.0357, 0.0600, 0.8693, 0.0142, 0.0208, 0.5122, 0.0569, 0.0620, 0.0965, 0.1410, 0.0571, 0.0742, 0.2929, 0.7071], dtype=torch.float32) 
                print('meta_weights type', type(meta_weights))
                print(meta_weights.size())
                
                if args.bce:
                    criterion2 = nn.BCEWithLogitsLoss(weight=meta_weights)
                elif args.mlm:
                    criterion2 = nn.MultiLabelMarginLoss(weight=meta_weights)
                else:
                    criterion2 = nn.MultiLabelSoftMarginLoss(weight=meta_weights)
            else:
                if args.bce:
                    criterion2 = nn.BCEWithLogitsLoss()
                elif args.mlm:
                    criterion2 = nn.MultiLabelMarginLoss()
                else:
                    criterion2 = nn.MultiLabelSoftMarginLoss()
        else:
            criterion2 = nn.CrossEntropyLoss()
    elif args.domain_adaptation2:
        criterion2 = MetaCL(temperature=args.temperature)
    
           
    if args.model not in ['ast', 'ssast'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')

    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        state_dict = ckpt['model']

        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        if args.multitask:
            # for multitask 
            if ckpt.get('lung_classifier', None) is not None:
                lung_classifier.load_state_dict(ckpt['lung_classifier'], strict=True)
            
            if ckpt.get('disease_classifier', None) is not None:
                disease_classifier.load_state_dict(ckpt['disease_classifier'], strict=True)
        
        elif args.multitask_domain:
            # for multitask 
            if ckpt.get('lung_classifier', None) is not None:
                lung_classifier.load_state_dict(ckpt['lung_classifier'], strict=True)
            
            if ckpt.get('disease_classifier', None) is not None:
                disease_classifier.load_state_dict(ckpt['disease_classifier'], strict=True)
            
            if ckpt.get('domain_classifier', None) is not None:
                domain_classifier.load_state_dict(ckpt['domain_classifier'], strict=True)
        
        else:
            if ckpt.get('classifier', None) is not None:
                classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
    
    #if args.method == 'ce':
    if args.method == 'ce':
        if args.domain_adaptation or args.domain_adaptation2:
            criterion = [criterion.cuda(), criterion2.cuda()]
        elif args.multitask:
            criterion = [criterion.cuda(), criterion2.cuda()]
        elif args.multitask_domain:
            criterion = [criterion.cuda(), criterion2.cuda(), criterion3.cuda()]
        else:
            criterion = [criterion.cuda()]
    elif args.method == 'patchmix':
        
        if args.domain_adaptation:
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda(), PatchMixLoss(criterion=criterion2).cuda()]
        elif args.domain_adaptation2:
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda(), PatchMixConLoss(temperature=args.temperature).cuda()]
        elif args.multitask:
            criterion = [criterion.cuda(), criterion2.cuda()]
        elif args.multitask_domain:
            criterion = [criterion.cuda(), criterion2.cuda(), criterion3.cuda()]
        else:
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda()]
    elif args.method == 'patchmix_cl':
        criterion = [criterion.cuda(), PatchMixConLoss(temperature=args.temperature).cuda()]
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.cuda()
    if args.domain_adaptation or args.domain_adaptation2:
        classifier = [class_classifier.cuda(), domain_classifier.cuda()]
    elif args.multitask:
        classifier = [lung_classifier.cuda(), disease_classifier.cuda()]
    elif args.multitask_domain:
        classifier = [lung_classifier.cuda(), disease_classifier.cuda(), domain_classifier.cuda()]
    else:
        classifier.cuda()
    projector.cuda()
    
    
    if args.domain_adaptation or args.domain_adaptation2:
        optim_params = list(model.parameters()) + list(classifier[0].parameters()) + list(classifier[-1].parameters()) + list(projector.parameters())
    elif args.multitask:
        optim_params = list(model.parameters()) + list(classifier[0].parameters()) + list(classifier[-1].parameters()) + list(projector.parameters())
    elif args.multitask_domain:
        optim_params = list(model.parameters()) + list(classifier[0].parameters()) + list(classifier[1].parameters()) + list(classifier[2].parameters()) + list(projector.parameters())
    else:
        optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, projector, criterion, optimizer


def train_multitask(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    if args.multitask:
        classifier[0].train()
        classifier[1].train()
    elif args.multitask_domain:
        classifier[0].train()
        classifier[1].train()
        classifier[2].train()
    projector.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lung_accs = AverageMeter()
    disease_accs = AverageMeter()
    domain_accs = AverageMeter()
    
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        # data load
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        #print('images', images.size())
        
        if args.multitask:
            lung_class_labels = labels[0].cuda(non_blocking=True)
            disease_class_labels = labels[1].cuda(non_blocking=True)
        elif args.multitask_domain:
            lung_class_labels = labels[0].cuda(non_blocking=True)
            disease_class_labels = labels[1].cuda(non_blocking=True)
            domain_class_labels = labels[2].cuda(non_blocking=True)
            
        bsz = lung_class_labels.shape[0]
                
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                if args.multitask:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier[0].state_dict()), deepcopy(classifier[1].state_dict()), deepcopy(projector.state_dict())]
                elif args.multitask_domain:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier[0].state_dict()), deepcopy(classifier[1].state_dict()), deepcopy(classifier[2].state_dict()), deepcopy(projector.state_dict())]
                p = float(idx + epoch * len(train_loader)) / args.epochs + 1 / len(train_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                features = model(args.transforms(images), args=args, alpha=alpha, training=True)
                
                if args.multitask:
                    lung_output = classifier[0](features)
                    disease_output = classifier[1](features)
                    loss = (args.weight1 * criterion[0](lung_output, lung_class_labels)) + (args.weight2 * criterion[1](disease_output, disease_class_labels))
                elif args.multitask_domain:
                    lung_output = classifier[0](features)
                    disease_output = classifier[1](features)
                    domain_output = classifier[2](features)
                    loss = (args.weight1 * criterion[0](lung_output, lung_class_labels)) + (args.weight2 * criterion[1](disease_output, disease_class_labels)) + (args.weight3 * criterion[2](domain_output, domain_class_labels))

        losses.update(loss.item(), bsz)
        [lung_acc], _ = accuracy(lung_output[:bsz], lung_class_labels, topk=(1,))
        [disease_acc], _ = accuracy(disease_output[:bsz], disease_class_labels, topk=(1,))
        lung_accs.update(lung_acc[0], bsz)
        disease_accs.update(disease_acc[0], bsz)
        
        if args.multitask_domain:
            [domain_acc], _ = accuracy(domain_output[:bsz], domain_class_labels, topk=(1,))
            domain_accs.update(domain_acc[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                if args.multitask:
                    classifier[0] = update_moving_average(args.ma_beta, classifier[0], ma_ckpt[1])
                    classifier[1] = update_moving_average(args.ma_beta, classifier[1], ma_ckpt[2])
                elif args.multitask_domain:
                    classifier[0] = update_moving_average(args.ma_beta, classifier[0], ma_ckpt[1])
                    classifier[1] = update_moving_average(args.ma_beta, classifier[1], ma_ckpt[2])
                    classifier[2] = update_moving_average(args.ma_beta, classifier[2], ma_ckpt[3])
                projector = update_moving_average(args.ma_beta, projector, ma_ckpt[2])

        # print info
        
        if args.multitask:
            if (idx + 1) % args.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Lung_Acc {lung_accs.val:.3f} ({lung_accs.avg:.3f})\t'
                      'Disease_Acc {disease_accs.val:.3f} ({disease_accs.avg:.3f})\t'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, lung_accs=lung_accs, disease_accs=disease_accs))
                sys.stdout.flush()
        elif args.multitask_domain:
            if (idx + 1) % args.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Lung_Acc {lung_accs.val:.3f} ({lung_accs.avg:.3f})\t'
                      'Disease_Acc {disease_accs.val:.3f} ({disease_accs.avg:.3f})\t'
                      'Domain_Acc {domain_accs.val:.3f} ({domain_accs.avg:.3f})\t'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, lung_accs=lung_accs, disease_accs=disease_accs, domain_accs=domain_accs))
                sys.stdout.flush()

    if args.multitask:
        return losses.avg, lung_accs.avg, disease_accs.avg
    elif args.multitask_domain:
        return losses.avg, lung_accs.avg, disease_accs.avg, domain_accs.avg


def train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None):
    model.train()
    if args.domain_adaptation or args.domain_adaptation2:
        classifier[0].train()
        classifier[1].train()
    else:
        classifier.train()
    projector.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    end = time.time()
    optimizer.zero_grad()
    
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        if args.domain_adaptation or args.domain_adaptation2:
            class_labels = labels[0].cuda(non_blocking=True)
            if args.meta_all:
                meta_labels = torch.stack(labels[1], dim=-1)
                meta_labels = meta_labels.cuda(non_blocking=True)
                if args.bce:
                    meta_labels = meta_labels.half()
            elif args.meta_all_supcon:
                age_meta_labels = labels[1][0].cuda(non_blocking=True)
                sex_meta_labels = labels[1][1].cuda(non_blocking=True)
                dev_meta_labels = labels[1][2].cuda(non_blocking=True)
                loc_meta_labels = labels[1][3].cuda(non_blocking=True)
                data_meta_labels = labels[1][4].cuda(non_blocking=True)
                
            else:
                meta_labels = labels[1].cuda(non_blocking=True)
        else:
            labels = labels.cuda(non_blocking=True)
        bsz = class_labels.shape[0] if args.domain_adaptation or args.domain_adaptation2 else labels.shape[0]
        
        
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                if args.domain_adaptation or args.domain_adaptation2:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier[0].state_dict()), deepcopy(classifier[1].state_dict()), deepcopy(projector.state_dict())]
                    p = float(idx + epoch * len(train_loader)) / args.epochs + 1 / len(train_loader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]
                    alpha = None

        if args.accum:
            #effective_len = len(train_loader) // args.accum_steps
            #global_step = (epoch * effective_len) + (idx // args.accum_steps)
            effective_idx = idx // args.accum_steps
            effective_len = len(train_loader) // args.accum_steps
            #warmup_learning_rate(args, global_step, effective_len, optimizer)
            warmup_learning_rate(args, epoch, effective_idx, effective_len, optimizer)
        else:
            warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast():
            if args.method == 'ce':
                features = model(args.transforms(images), args=args, alpha=alpha, training=True)
                if args.domain_adaptation:
                    #features = (features, domain_features) # domain_features -> ReverseLayerF
                    output = classifier[0](features[0])
                    class_loss = criterion[0](output, class_labels)                                   
                    meta_output = classifier[1](features[1])
                    meta_loss = criterion[1](meta_output, meta_labels)
                    loss = class_loss + (args.alpha * meta_loss)                
                elif args.domain_adaptation2:
                    output = classifier[0](features[0])
                    class_loss = criterion[0](output, class_labels)
                    
                    features2 = model(args.transforms(images), args=args, alpha=alpha, training=True)
                    feat1 = features[0]
                    feat2 = features2[0]
                
                    if args.target_type == 'project_flow_all':
                        proj1, proj2 = classifier[1](feat1), classifier[1](feat2) # classifier[1] = projector #
                        
                    elif args.target_type == 'representation_all':
                        proj1, proj2 = feat1, feat2
                    
                    elif args.target_type == 'z1block_project':
                        proj1 = deepcopy(feat1.detach())
                        proj2 = classifier[1](feat2)
                    
                    elif args.target_type == 'z1_project2':
                        proj1 = feat1
                        proj2 = classifier[1](feat2)
                    
                    elif args.target_type == 'project1block_project2':
                        proj1 = deepcopy(classifier[1](feat1).detach())
                        proj2 = classifier[1](feat2)
                    
                    elif args.target_type == 'project1_r2block':
                        proj1 = classifier[1](feat1)
                        proj2 = deepcopy(feat2.detach())
                    
                    elif args.target_type == 'project1_r2':
                        proj1 = classifier[1](feat1)
                        proj2 = feat2
                    
                    elif args.target_type == 'project1_project2block':
                        proj1 = classifier[1](feat1)
                        proj2 = deepcopy(classifier[1](feat2).detach())
                    
                    elif args.target_type == 'project_block_all':
                        proj1 = deepcopy(classifier[1](feat1).detach())
                        proj2 = deepcopy(classifier[1](feat2).detach())
                    
                    if args.meta_all_supcon:
                        meta_age_loss = criterion[1](proj1, proj2, age_meta_labels)
                        meta_sex_loss = criterion[1](proj1, proj2, sex_meta_labels)
                        meta_dev_loss = criterion[1](proj1, proj2, dev_meta_labels)
                        meta_loc_loss = criterion[1](proj1, proj2, loc_meta_labels)
                        meta_data_loss = criterion[1](proj1, proj2, data_meta_labels)
                        
                        if args.polarization:
                            meta_losses = polarization(meta_age_loss, meta_sex_loss, meta_dev_loss, meta_loc_loss, meta_data_loss, args)
                            loss = class_loss + meta_losses
                        elif args.equalization:
                            meta_losses = equalization(meta_age_loss, meta_sex_loss, meta_dev_loss, meta_loc_loss, meta_data_loss, args)
                            loss = class_loss + meta_losses
                        elif args.equalization_ver2:
                            meta_losses = equalization_ver2(meta_age_loss, meta_sex_loss, meta_dev_loss, meta_loc_loss, meta_data_loss, args)
                            loss = class_loss + meta_losses
                        else:
                            loss = class_loss + (args.alpha * meta_age_loss / 5) + (args.alpha * meta_sex_loss / 5) + (args.alpha * meta_dev_loss / 5) + (args.alpha * meta_loc_loss / 5) + (args.alpha * meta_data_loss / 5)
                        #print('loss', loss)
                    else:
                        meta_loss = criterion[1](proj1, proj2, meta_labels) #meta cl loss
                        loss = class_loss + (args.alpha * meta_loss)
                    
                
                else:
                    output = classifier(features)
                    loss = criterion[0](output, labels)
                    

            
        #print('loss', loss)
        
        if args.accum:
            loss = loss / args.accum_steps
            losses.update(loss.item() * args.accum_steps, bsz)
        else:
            losses.update(loss.item(), bsz)
            
        [acc1], _ = accuracy(output[:bsz], class_labels if args.domain_adaptation or args.domain_adaptation2 else labels, topk=(1,))
        top1.update(acc1[0], bsz)
        
        scaler.scale(loss).backward()
        
        if args.accum:
            if (idx + 1) % args.accum_steps == 0 or (idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        else:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                if args.domain_adaptation or args.domain_adaptation2:
                    classifier[0] = update_moving_average(args.ma_beta, classifier[0], ma_ckpt[1])
                    classifier[1] = update_moving_average(args.ma_beta, classifier[1], ma_ckpt[2])
                else:
                    classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])
                projector = update_moving_average(args.ma_beta, projector, ma_ckpt[2])

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg

def validate_multitask_domain(val_loader, model, classifier, criterion, args, best_average, best_lung, best_disease, best_domain, best_model=None):
    save_bool = False
    model.eval()
    
    classifier[0].eval()
    classifier[1].eval()
    classifier[2].eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    lung_accs = AverageMeter()
    disease_accs = AverageMeter()
    domain_accs = AverageMeter()
    
    lung_hits, lung_counts = [0.0] * args.lung_cls, [0.0] * args.lung_cls
    disease_hits, disease_counts = [0.0] * args.disease_cls, [0.0] * args.disease_cls
    domain_hits, domain_counts = [0.0] * args.domain_cls, [0.0] * args.domain_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            lung_class_labels = labels[0].cuda(non_blocking=True)
            disease_class_labels = labels[1].cuda(non_blocking=True)
            domain_class_labels = labels[2].cuda(non_blocking=True)
            
            images = images.cuda(non_blocking=True)
            
            lung_class_labels = lung_class_labels.cuda(non_blocking=True)
            disease_class_labels = disease_class_labels.cuda(non_blocking=True)
            domain_class_labels = domain_class_labels.cuda(non_blocking=True)
            
            bsz = lung_class_labels.shape[0]
            with torch.cuda.amp.autocast():
                features = model(images, args=args, training=False)
                #output = classifier(features)
                #loss = criterion[0](output, labels)
                lung_output = classifier[0](features)
                disease_output = classifier[1](features)
                domain_output = classifier[2](features)
                #loss = criterion[0](lung_output, lung_class_labels) + criterion[1](disease_output, disease_class_labels)
                loss = (args.weight1 * criterion[0](lung_output, lung_class_labels)) + (args.weight2 * criterion[1](disease_output, disease_class_labels)) + (args.weight3 * criterion[2](domain_output, domain_class_labels))

            losses.update(loss.item(), bsz)
            [lung_acc], _ = accuracy(lung_output, lung_class_labels, topk=(1,))
            [disease_acc], _ = accuracy(disease_output, disease_class_labels, topk=(1,))
            [domain_acc], _ = accuracy(domain_output, domain_class_labels, topk=(1,))
            lung_accs.update(lung_acc[0], bsz)
            disease_accs.update(disease_acc[0], bsz)
            domain_accs.update(domain_acc[0], bsz)

            
            # for lung class
            _, lung_preds = torch.max(lung_output, 1)
            for idx in range(lung_preds.shape[0]):
                lung_counts[lung_class_labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if lung_preds[idx].item() == lung_class_labels[idx].item():
                        lung_hits[lung_class_labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if lung_class_labels[idx].item() == 0 and lung_preds[idx].item() == lung_class_labels[idx].item():
                        lung_hits[lung_class_labels[idx].item()] += 1.0
                    elif lung_class_labels[idx].item() != 0 and lung_preds[idx].item() > 0:  # abnormal
                        lung_hits[lung_class_labels[idx].item()] += 1.0

            lung_sp, lung_se, lung_sc = get_score(lung_hits, lung_counts)
            
            # for disease class
            _, disease_preds = torch.max(disease_output, 1)
            for idx in range(disease_preds.shape[0]):
                disease_counts[disease_class_labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if disease_preds[idx].item() == disease_class_labels[idx].item():
                        disease_hits[disease_class_labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if disease_class_labels[idx].item() == 0 and disease_preds[idx].item() == disease_class_labels[idx].item():
                        disease_hits[disease_class_labels[idx].item()] += 1.0
                    elif disease_class_labels[idx].item() != 0 and disease_preds[idx].item() > 0:  # abnormal
                        disease_hits[lung_class_labels[idx].item()] += 1.0

            disease_sp, disease_se, disease_sc = get_score(disease_hits, disease_counts)
            
            # for domain class
            _, domain_preds = torch.max(domain_output, 1)
            for idx in range(domain_preds.shape[0]):
                domain_counts[domain_class_labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if domain_preds[idx].item() == domain_class_labels[idx].item():
                        domain_hits[domain_class_labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if domain_class_labels[idx].item() == 0 and domain_preds[idx].item() == domain_class_labels[idx].item():
                        domain_hits[domain_class_labels[idx].item()] += 1.0
                    elif domain_class_labels[idx].item() != 0 and domain_preds[idx].item() > 0:  # abnormal
                        domain_hits[domain_class_labels[idx].item()] += 1.0

            domain_sp, domain_se, domain_sc = get_score(domain_hits, domain_counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {lung_accs.val:.3f} ({lung_accs.avg:.3f})\t'
                      'Acc@1 {disease_accs.val:.3f} ({disease_accs.avg:.3f})\t'
                      'Acc@1 {domain_accs.val:.3f} ({domain_accs.avg:.3f})\t'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, lung_accs=lung_accs, disease_accs=disease_accs, domain_accs=domain_accs))
    
    '''
    average_sc = (lung_sc + disease_sc + domain_sc)/3
    average_sp = (lung_sp + disease_sp + domain_sp)/3
    average_se = (lung_se + disease_se + domain_se)/3
    '''
    average_sc = (lung_sc + disease_sc)/2
    average_sp = (lung_sp + disease_sp)/2
    average_se = (lung_se + disease_se)/2
    
    if args.lung_first:
        if lung_sc > best_lung[-1] and lung_se > 5:
            save_bool = True
            best_average = [average_sp, average_se, average_sc]
            best_lung = [lung_sp, lung_se, lung_sc]
            best_disease = [disease_sp, disease_se, disease_sc]
            best_domain = [domain_sp, domain_se, domain_sc]
            best_model = [deepcopy(model.state_dict()), deepcopy(classifier[0].state_dict()), deepcopy(classifier[1].state_dict()), deepcopy(classifier[2].state_dict())]
    
    else:
        if average_sc > best_average[-1] and average_se > 5:
            save_bool = True
            best_average = [average_sp, average_se, average_sc]
            best_lung = [lung_sp, lung_se, lung_sc]
            best_disease = [disease_sp, disease_se, disease_sc]
            #best_domain = [domain_sp, domain_se, domain_sc]
            best_domain = [float(domain_accs.avg), float(domain_accs.avg), float(domain_accs.avg)]
            best_model = [deepcopy(model.state_dict()), deepcopy(classifier[0].state_dict()), deepcopy(classifier[1].state_dict()), deepcopy(classifier[2].state_dict())]

    print(' * Average *')
    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(average_sp, average_se, average_sc, best_average[0], best_average[1], best_average[-1]))
    
    print(' * Lung *')
    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(lung_sp, lung_se, lung_sc, best_lung[0], best_lung[1], best_lung[-1]))
    print(' * Lung Acc@1 {lung_accs.avg:.2f}'.format(lung_accs=lung_accs))
    
    print(' * Disease *')
    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(disease_sp, disease_se, disease_sc, best_disease[0], best_disease[1], best_disease[-1]))
    print(' * Disease Acc@1 {disease_accs.avg:.2f}'.format(disease_accs=disease_accs))
    
    print(' * Domain *')
    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(domain_sp, domain_se, domain_sc, best_domain[0], best_domain[1], best_domain[-1]))
    print(' * Domain Acc@1 {domain_accs.avg:.2f}'.format(domain_accs=domain_accs))
    
    
    return best_average, best_lung, best_disease, best_domain, best_model, save_bool

def validate_multitask(val_loader, model, classifier, criterion, args, best_average, best_lung, best_disease, best_model=None):
    save_bool = False
    model.eval()
    
    if args.multitask:
        classifier[0].eval()
        classifier[1].eval()
    else:
        if args.domain_adaptation or args.domain_adaptation2:
            classifier = classifier[0]
        classifier.eval()
    

    batch_time = AverageMeter()
    losses = AverageMeter()
    lung_accs = AverageMeter()
    disease_accs = AverageMeter()
    lung_hits, lung_counts = [0.0] * args.lung_cls, [0.0] * args.lung_cls
    disease_hits, disease_counts = [0.0] * args.disease_cls, [0.0] * args.disease_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            if args.multitask:
                lung_class_labels = labels[0].cuda(non_blocking=True)
                disease_class_labels = labels[1].cuda(non_blocking=True)
            
            images = images.cuda(non_blocking=True)
            
            lung_class_labels = lung_class_labels.cuda(non_blocking=True)
            disease_class_labels = disease_class_labels.cuda(non_blocking=True)
            bsz = lung_class_labels.shape[0]
            with torch.cuda.amp.autocast():
                #if args.model == 'ast':
                features = model(images, args=args, training=False)
                #output = classifier(features)
                #loss = criterion[0](output, labels)
                lung_output = classifier[0](features)
                disease_output = classifier[1](features)
                #loss = criterion[0](lung_output, lung_class_labels) + criterion[1](disease_output, disease_class_labels)
                loss = (args.weight1 * criterion[0](lung_output, lung_class_labels)) + (args.weight2 * criterion[1](disease_output, disease_class_labels))

            losses.update(loss.item(), bsz)
            [lung_acc], _ = accuracy(lung_output, lung_class_labels, topk=(1,))
            [disease_acc], _ = accuracy(disease_output, disease_class_labels, topk=(1,))
            lung_accs.update(lung_acc[0], bsz)
            disease_accs.update(disease_acc[0], bsz)

            
            # for lung class
            _, lung_preds = torch.max(lung_output, 1)
            for idx in range(lung_preds.shape[0]):
                lung_counts[lung_class_labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if lung_preds[idx].item() == lung_class_labels[idx].item():
                        lung_hits[lung_class_labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if lung_class_labels[idx].item() == 0 and lung_preds[idx].item() == lung_class_labels[idx].item():
                        lung_hits[lung_class_labels[idx].item()] += 1.0
                    elif lung_class_labels[idx].item() != 0 and lung_preds[idx].item() > 0:  # abnormal
                        lung_hits[lung_class_labels[idx].item()] += 1.0

            lung_sp, lung_se, lung_sc = get_score(lung_hits, lung_counts)
            
            # for disease class
            _, disease_preds = torch.max(disease_output, 1)
            for idx in range(disease_preds.shape[0]):
                disease_counts[disease_class_labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if disease_preds[idx].item() == disease_class_labels[idx].item():
                        disease_hits[disease_class_labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if disease_class_labels[idx].item() == 0 and disease_preds[idx].item() == disease_class_labels[idx].item():
                        disease_hits[disease_class_labels[idx].item()] += 1.0
                    elif disease_class_labels[idx].item() != 0 and disease_preds[idx].item() > 0:  # abnormal
                        disease_hits[lung_class_labels[idx].item()] += 1.0

            disease_sp, disease_se, disease_sc = get_score(disease_hits, disease_counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {lung_accs.val:.3f} ({lung_accs.avg:.3f})\t'
                      'Acc@1 {disease_accs.val:.3f} ({disease_accs.avg:.3f})\t'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, lung_accs=lung_accs, disease_accs=disease_accs))
    
    average_sc = (lung_sc + disease_sc)/2
    average_sp = (lung_sp + disease_sp)/2
    average_se = (lung_se + disease_se)/2
    
    
    #if sc > best_acc[-1] and se > 5:
    if args.lung_first:
        if lung_sc > best_lung[-1] and lung_se > 5:
            save_bool = True
            best_average = [average_sp, average_se, average_sc]
            best_lung = [lung_sp, lung_se, lung_sc]
            best_disease = [disease_sp, disease_se, disease_sc]
            best_model = [deepcopy(model.state_dict()), deepcopy(classifier[0].state_dict()), deepcopy(classifier[1].state_dict())]
    
    else:
        if average_sc > best_average[-1] and average_se > 5:
            save_bool = True
            best_average = [average_sp, average_se, average_sc]
            best_lung = [lung_sp, lung_se, lung_sc]
            best_disease = [disease_sp, disease_se, disease_sc]
            best_model = [deepcopy(model.state_dict()), deepcopy(classifier[0].state_dict()), deepcopy(classifier[1].state_dict())]

    print(' * Average *')
    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(average_sp, average_se, average_sc, best_average[0], best_average[1], best_average[-1]))
    
    print(' * Lung *')
    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(lung_sp, lung_se, lung_sc, best_lung[0], best_lung[1], best_lung[-1]))
    print(' * Lung Acc@1 {lung_accs.avg:.2f}'.format(lung_accs=lung_accs))
    
    print(' * Disease *')
    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(disease_sp, disease_se, disease_sc, best_disease[0], best_disease[1], best_disease[-1]))
    print(' * Disease Acc@1 {disease_accs.avg:.2f}'.format(disease_accs=disease_accs))
    
    
    return best_average, best_lung, best_disease, best_model, save_bool


def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    
    if args.domain_adaptation or args.domain_adaptation2:
        classifier = classifier[0]
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]            

            with torch.cuda.amp.autocast():
                features = model(images, args=args, training=False)
                output = classifier(features)
                loss = criterion[0](output, labels)                

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if labels[idx].item() == 0 and preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                    elif labels[idx].item() != 0 and preds[idx].item() > 0:  # abnormal
                        hits[labels[idx].item()] += 1.0

            sp, se, sc = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    
    if sc > best_acc[-1] and se > 5:
        save_bool = True
        best_acc = [sp, se, sc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[-1]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool

def validate_each(val_loader, model, classifier, criterion, args):
    model.eval()
    best_acc = [0.0, 0.0, 0.0]
    
    if args.domain_adaptation or args.domain_adaptation2:
        classifier = classifier[0]
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]            

            with torch.cuda.amp.autocast():
                features = model(images, args=args, training=False)
                output = classifier(features)
                loss = criterion[0](output, labels)

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if not args.two_cls_eval:
                    if preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                else:  # only when args.n_cls == 4
                    if labels[idx].item() == 0 and preds[idx].item() == labels[idx].item():
                        hits[labels[idx].item()] += 1.0
                    elif labels[idx].item() != 0 and preds[idx].item() > 0:  # abnormal
                        hits[labels[idx].item()] += 1.0

            sp, se, sc = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    
    if se > 5:
        best_acc = [sp, se, sc]
    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f}'.format(sp, se, sc))
    return best_acc


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    best_model_icbhi = None
    best_model_snubh = None
    best_model_smart = None
    best_model_icbhi_snubh = None
    best_model_icbhi_smart = None
    best_model_snubh_smart = None
    best_model_all = None    
    best_model = None
    
    # for mtl
    
    best_average = [0, 0, 0]
    best_lung = [0, 0, 0]
    best_disease = [0, 0, 0]
    best_domain = [0, 0, 0]
    
    
    best_average_icbhi = [0, 0, 0]
    best_lung_icbhi = [0, 0, 0]
    best_disease_icbhi = [0, 0, 0]
    best_domain_icbhi = [0, 0, 0]
    best_average_icbhi_child = [0, 0, 0]
    best_lung_icbhi_child = [0, 0, 0]
    best_disease_icbhi_child = [0, 0, 0]
    best_domain_icbhi_child = [0, 0, 0]
    best_average_icbhi_nonchild = [0, 0, 0]
    best_lung_icbhi_nonchild = [0, 0, 0]
    best_disease_icbhi_nonchild = [0, 0, 0]
    best_domain_icbhi_nonchild = [0, 0, 0]
    
    best_average_snubh = [0, 0, 0]
    best_lung_snubh = [0, 0, 0]
    best_disease_snubh = [0, 0, 0]
    best_domain_snubh = [0, 0, 0]
    
    best_average_smart = [0, 0, 0]
    best_lung_smart = [0, 0, 0]
    best_disease_smart = [0, 0, 0]
    best_domain_smart = [0, 0, 0]
    
    
    if args.dataset == 'all':
        best_acc_all = [0, 0, 0]
        best_acc_icbhi = [0, 0, 0]  # Specificity, Sensitivity, Score
        best_acc_snubh = [0, 0, 0]  # Specificity, Sensitivity, Score
        best_acc_smart = [0, 0, 0]  # Specificity, Sensitivity, Score
        if args.measure_all:
            best_acc_icbhi_child = [0, 0, 0]
            best_acc_icbhi_nonchild = [0, 0, 0]
    
    elif args.dataset == 'icbhi_snubh':
        best_acc_all = [0, 0, 0]
        best_acc_icbhi = [0, 0, 0]  # Specificity, Sensitivity, Score
        best_acc_snubh = [0, 0, 0]  # Specificity, Sensitivity, Score
        if args.measure_all:
            best_acc_icbhi_child = [0, 0, 0]
            best_acc_icbhi_nonchild = [0, 0, 0]
    
    elif args.dataset == 'icbhi_smart':
        best_acc_all = [0, 0, 0]
        best_acc_icbhi = [0, 0, 0]  # Specificity, Sensitivity, Score
        best_acc_smart = [0, 0, 0]  # Specificity, Sensitivity, Score
        if args.measure_all:
            best_acc_icbhi_child = [0, 0, 0]
            best_acc_icbhi_nonchild = [0, 0, 0]
    
    elif args.dataset == 'snubh_smart':
        best_acc_all = [0, 0, 0]
        best_acc_snubh = [0, 0, 0]  # Specificity, Sensitivity, Score
        best_acc_smart = [0, 0, 0]  # Specificity, Sensitivity, Score
    
    else:
        best_acc = [0, 0, 0]
        
    
    args.transforms = SpecAugment(args)
    train_loader, val_loader, args = set_loader(args)
    model, classifier, projector, criterion, optimizer = set_model(args)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
    
    
    
    
    if args.dataset == 'icbhi_snubh':
        val_loader_icbhi = val_loader[0] #2760
        val_loader_snubh = val_loader[1]
        val_loader_icbhi_child = val_loader[2]
        val_loader_icbhi_nonchild = val_loader[3]
    elif args.dataset == 'icbhi_smart':
        val_loader_icbhi = val_loader[0] #2760
        val_loader_smart = val_loader[1]
        val_loader_icbhi_child = val_loader[2]
        val_loader_icbhi_nonchild = val_loader[3]
    elif args.dataset == 'snubh_smart':
        val_loader_snubh = val_loader[0] #2760
        val_loader_smart = val_loader[1]
    elif args.dataset == 'all':
        val_loader_icbhi = val_loader[0] #2760
        val_loader_snubh = val_loader[1]
        val_loader_smart = val_loader[2]
        val_loader_icbhi_child = val_loader[3]
        val_loader_icbhi_nonchild = val_loader[4]
    
    first_data = args.dataset.split('_')[0]
    second_data = args.dataset.split('_')[-1]
    print('first {} second {}'.format(first_data, second_data))
    
    if not args.eval:
        print('Experiments {} start'.format(args.model_name))
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            time1 = time.time()
            
            
            if args.multitask:
                loss, lung_acc, disease_acc = train_multitask(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
                time2 = time.time()
                print('Train epoch {}, total time {:.2f}, lung_accuracy:{:.2f} disease_accraucy:{:.2f}'.format(epoch, time2-time1, lung_acc, disease_acc))
            elif args.multitask_domain:
                loss, lung_acc, disease_acc, domain_acc = train_multitask(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
                time2 = time.time()
                print('Train epoch {}, total time {:.2f}, lung_accuracy:{:.2f} disease_accraucy:{:.2f}, domain_accraucy:{:.2f}'.format(epoch, time2-time1, lung_acc, disease_acc, domain_acc))
            else:
                loss, acc = train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler)
                time2 = time.time()
                print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, acc))
                        
            
            if args.dataset in ['icbhi_snubh', 'icbhi_smart']:
                # eval for one epoch
                if args.multitask:
                    best_average_icbhi, best_lung_icbhi, best_disease_icbhi, best_model_icbhi, save_bool_icbhi = validate_multitask(val_loader_icbhi, model, classifier, criterion, args, best_average, best_lung, best_disease, best_model)
                elif args.multitask_domain:
                    best_average_icbhi, best_lung_icbhi, best_disease_icbhi, best_domain_icbhi, best_model_icbhi, save_bool_icbhi = validate_multitask_domain(val_loader_icbhi, model, classifier, criterion, args, best_average, best_lung, best_disease, best_domain, best_model)
                else:                    
                    best_acc_icbhi, best_model_icbhi, save_bool_icbhi = validate(val_loader_icbhi, model, classifier, criterion, args, best_acc_icbhi, best_model_icbhi)
                
                
                # save a checkpoint of model and classifier when the best score is updated
                if save_bool_icbhi:            
                    if args.multitask:
                        print('ICBHI Best')
                        print('(Average) ICBHI Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi[2], epoch))
                        print('(Lung) ICBHI Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi[2], epoch))
                        print('(Disease) ICBHI Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi[2], epoch))
                        
                        print('ICBHI-Child test results')
                        best_average_icbhi_child, best_lung_icbhi_child, best_disease_icbhi_child, _, _ = validate_multitask(val_loader_icbhi_child, model, classifier, criterion, args, best_average_icbhi_child, best_lung_icbhi_child, best_disease_icbhi_child, best_model)
                        print('(Average) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi_child[2], epoch))
                        print('(Lung) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi_child[2], epoch))
                        print('(Disease) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi_child[2], epoch))
                        
                        print('ICBHI-Non Child test results')
                        best_average_icbhi_nonchild, best_lung_icbhi_nonchild, best_disease_icbhi_nonchild, _, _ = validate_multitask(val_loader_icbhi_nonchild, model, classifier, criterion, args, best_average_icbhi_nonchild, best_lung_icbhi_nonchild, best_disease_icbhi_nonchild, best_model)
                        print('(Average) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi_nonchild[2], epoch))
                        print('(Lung) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi_nonchild[2], epoch))
                        print('(Disease) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi_nonchild[2], epoch))                        
                    elif args.multitask_domain:
                        print('ICBHI Best')
                        print('(Average) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi[2], epoch))
                        print('(Lung) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi[2], epoch))
                        print('(Disease) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi[2], epoch))
                        print('(Domain) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain_icbhi[2], epoch))
                        
                        print('ICBHI-Child test results')
                        best_average_icbhi_child, best_lung_icbhi_child, best_disease_icbhi_child, _, _ = validate_multitask_domain(val_loader_icbhi_child, model, classifier, criterion, args, best_average_icbhi_child, best_lung_icbhi_child, best_disease_icbhi_child, best_domain_icbhi_child, best_model)
                        print('(Average) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi_child[2], epoch))
                        print('(Lung) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi_child[2], epoch))
                        print('(Disease) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi_child[2], epoch))
                        print('(Domain) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi_child[2], epoch))
                        
                        print('ICBHI-Non Child test results')
                        best_average_icbhi_nonchild, best_lung_icbhi_nonchild, best_disease_icbhi_nonchild, best_domain_icbhi_nonchild, _, _ = validate_multitask_domain(val_loader_icbhi_nonchild, model, classifier, criterion, args, best_average_icbhi_nonchild, best_lung_icbhi_nonchild, best_disease_icbhi_nonchild, best_domain_icbhi_nonchild, best_model)
                        print('(Average) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi_nonchild[2], epoch))
                        print('(Lung) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi_nonchild[2], epoch))
                        print('(Disease) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi_nonchild[2], epoch))
                        print('(Domain) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain_icbhi_nonchild[2], epoch))
                    else:
                        #print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_icbhi[2], epoch))
                        print('ICBHI Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_icbhi[2], epoch))
                        print('ICBHI-Child test results')
                        best_acc_icbhi_child = validate_each(val_loader_icbhi_child, model, classifier, criterion, args)
                        print('ICBHI-Non Child test results')
                        best_acc_icbhi_nonchild = validate_each(val_loader_icbhi_nonchild, model, classifier, criterion, args)
                        print('ICBHI-Child {} Non-Child {}'.format(best_acc_icbhi_child[2], best_acc_icbhi_nonchild[2]))
                    
                    save_file_icbhi = os.path.join(args.save_folder, 'icbhi_best_epoch_{}.pth'.format(epoch))
                    if args.multitask:
                        save_model_multitask(model, optimizer, args, epoch, save_file_icbhi, classifier[0], classifier[1])
                    elif args.multitask_domain:
                        save_model_multitask_domain(model, optimizer, args, epoch, save_file_icbhi, classifier[0], classifier[1], classifier[2])
                    else:
                        save_model(model, optimizer, args, epoch, save_file_icbhi, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
                
                if second_data == 'snubh':
                    if args.multitask:
                        best_average_snubh, best_lung_snubh, best_disease_snubh, best_model_snubh, save_bool_snubh = validate_multitask(val_loader_snubh, model, classifier, criterion, args, best_average_snubh, best_lung_snubh, best_disease_snubh, best_model_snubh)
                    elif args.multitask_domain:
                        best_average_snubh, best_lung_snubh, best_disease_snubh, best_domain_snubh, best_model_snubh, save_bool_snubh = validate_multitask_domain(val_loader_snubh, model, classifier, criterion, args, best_average_snubh, best_lung_snubh, best_disease_snubh, best_domain_snubh, best_model_snubh)
                    else:                    
                        best_acc_snubh, best_model_snubh, save_bool_snubh = validate(val_loader_snubh, model, classifier, criterion, args, best_acc_snubh, best_model_snubh)
                                                            
                    if save_bool_snubh:    
                        if args.multitask:
                            print('(Average) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_snubh[2], epoch))
                            print('(Lung) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_snubh[2], epoch))
                            print('(Disease) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_snubh[2], epoch))
                        elif args.multitask_domain:
                            print('(Average) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_snubh[2], epoch))
                            print('(Lung) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_snubh[2], epoch))
                            print('(Disease) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_snubh[2], epoch))
                            print('(Domain) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain_snubh[2], epoch))
                        else:
                            print('SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_snubh[2], epoch))                            
                        save_file_snubh = os.path.join(args.save_folder, 'snubh_best_epoch_{}.pth'.format(epoch))
                        if args.multitask:
                            save_model_multitask(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1])
                        elif args.multitask_domain:
                            save_model_multitask_domain(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1], classifier[2])
                        else:
                            save_model(model, optimizer, args, epoch, save_file_snubh, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
                elif second_data == 'smart':
                    if args.multitask:
                        best_average_smart, best_lung_smart, best_disease_smart, best_model_smart, save_bool_smart = validate_multitask(val_loader_smart, model, classifier, criterion, args, best_average_smart, best_lung_smart, best_disease_smart, best_model_smart)
                    elif args.multitask_domain:
                        best_average_smart, best_lung_smart, best_disease_smart, best_domain_smart, best_model_smart, save_bool_smart = validate_multitask_domain(val_loader_smart, model, classifier, criterion, args, best_average_smart, best_lung_smart, best_disease_smart, best_domain_smart, best_model_smart)
                    else:                    
                        best_acc_smart, best_model_smart, save_bool_smart = validate(val_loader_smart, model, classifier, criterion, args, best_acc_smart, best_model_smart)
                                                            
                    if save_bool_smart:    
                        if args.multitask:
                            print('(Average) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_smart[2], epoch))
                            print('(Lung) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_smart[2], epoch))
                            print('(Disease) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_smart[2], epoch))
                        elif args.multitask_domain:
                            print('(Average) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_smart[2], epoch))
                            print('(Lung) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_smart[2], epoch))
                            print('(Disease) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_smart[2], epoch))
                            print('(Domain) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain_smart[2], epoch))
                        else:
                            print('SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_smart[2], epoch))                            
                        save_file_smart = os.path.join(args.save_folder, 'smart_best_epoch_{}.pth'.format(epoch))
                        if args.multitask:
                            save_model_multitask(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1])
                        elif args.multitask_domain:
                            save_model_multitask_domain(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1], classifier[2])
                        else:
                            save_model(model, optimizer, args, epoch, save_file_smart, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
                                    
            
            elif args.dataset == 'snubh_smart':
                if args.multitask:
                    best_average_snubh, best_lung_snubh, best_disease_snubh, best_model_snubh, save_bool_snubh = validate_multitask(val_loader_snubh, model, classifier, criterion, args, best_average_snubh, best_lung_snubh, best_disease_snubh, best_model_snubh)
                elif args.multitask_domain:
                    best_average_snubh, best_lung_snubh, best_disease_snubh, best_domain_snubh, best_model_snubh, save_bool_snubh = validate_multitask_domain(val_loader_snubh, model, classifier, criterion, args, best_average_snubh, best_lung_snubh, best_disease_snubh, best_domain_snubh, best_model_snubh)
                else:                    
                    best_acc_snubh, best_model_snubh, save_bool_snubh = validate(val_loader_snubh, model, classifier, criterion, args, best_acc_snubh, best_model_snubh)
                                                        
                if save_bool_snubh:    
                    if args.multitask:
                        print('(Average) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_snubh[2], epoch))
                        print('(Lung) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_snubh[2], epoch))
                        print('(Disease) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_snubh[2], epoch))
                    elif args.multitask_domain:
                        print('(Average) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_snubh[2], epoch))
                        print('(Lung) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_snubh[2], epoch))
                        print('(Disease) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_snubh[2], epoch))
                        print('(Domain) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain_snubh[2], epoch))
                    else:
                        print('SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_snubh[2], epoch))                            
                    save_file_snubh = os.path.join(args.save_folder, 'snubh_best_epoch_{}.pth'.format(epoch))
                    if args.multitask:
                        save_model_multitask(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1])
                    elif args.multitask_domain:
                        save_model_multitask_domain(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1], classifier[2])
                    else:
                        save_model(model, optimizer, args, epoch, save_file_snubh, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
                
                if args.multitask:
                    best_average_smart, best_lung_smart, best_disease_smart, best_model_smart, save_bool_smart = validate_multitask(val_loader_smart, model, classifier, criterion, args, best_average_smart, best_lung_smart, best_disease_smart, best_model_smart)
                elif args.multitask_domain:
                    best_average_smart, best_lung_smart, best_disease_smart, best_domain_smart, best_model_smart, save_bool_smart = validate_multitask_domain(val_loader_smart, model, classifier, criterion, args, best_average_smart, best_lung_smart, best_disease_smart, best_domain_smart, best_model_smart)
                else:                    
                    best_acc_smart, best_model_smart, save_bool_smart = validate(val_loader_smart, model, classifier, criterion, args, best_acc_smart, best_model_smart)
                                                        
                if save_bool_smart:    
                    if args.multitask:
                        print('(Average) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_smart[2], epoch))
                        print('(Lung) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_smart[2], epoch))
                        print('(Disease) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_smart[2], epoch))
                    elif args.multitask_domain:
                        print('(Average) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_smart[2], epoch))
                        print('(Lung) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_smart[2], epoch))
                        print('(Disease) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_smart[2], epoch))
                        print('(Domain) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain_smart[2], epoch))
                    else:
                        print('SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_smart[2], epoch))                            
                    save_file_smart = os.path.join(args.save_folder, 'smart_best_epoch_{}.pth'.format(epoch))
                    if args.multitask:
                        save_model_multitask(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1])
                    elif args.multitask_domain:
                        save_model_multitask_domain(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1], classifier[2])
                    else:
                        save_model(model, optimizer, args, epoch, save_file_smart, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
            
            elif args.dataset == 'all':
                if args.multitask:
                    best_average_icbhi, best_lung_icbhi, best_disease_icbhi, best_model_icbhi, save_bool_icbhi = validate_multitask(val_loader_icbhi, model, classifier, criterion, args, best_average, best_lung, best_disease, best_model)
                elif args.multitask_domain:
                    best_average_icbhi, best_lung_icbhi, best_disease_icbhi, best_domain_icbhi, best_model_icbhi, save_bool_icbhi = validate_multitask_domain(val_loader_icbhi, model, classifier, criterion, args, best_average, best_lung, best_disease, best_domain, best_model)
                else:                    
                    best_acc_icbhi, best_model_icbhi, save_bool_icbhi = validate(val_loader_icbhi, model, classifier, criterion, args, best_acc_icbhi, best_model_icbhi)
                
                
                # save a checkpoint of model and classifier when the best score is updated
                if save_bool_icbhi:            
                    if args.multitask:
                        print('ICBHI Best')
                        print('(Average) ICBHI Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi[2], epoch))
                        print('(Lung) ICBHI Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi[2], epoch))
                        print('(Disease) ICBHI Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi[2], epoch))
                        
                        print('ICBHI-Child test results')
                        best_average_icbhi_child, best_lung_icbhi_child, best_disease_icbhi_child, _, _ = validate_multitask(val_loader_icbhi_child, model, classifier, criterion, args, best_average_icbhi_child, best_lung_icbhi_child, best_disease_icbhi_child, best_model)
                        print('(Average) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi_child[2], epoch))
                        print('(Lung) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi_child[2], epoch))
                        print('(Disease) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi_child[2], epoch))
                        
                        print('ICBHI-Non Child test results')
                        best_average_icbhi_nonchild, best_lung_icbhi_nonchild, best_disease_icbhi_nonchild, _, _ = validate_multitask(val_loader_icbhi_nonchild, model, classifier, criterion, args, best_average_icbhi_nonchild, best_lung_icbhi_nonchild, best_disease_icbhi_nonchild, best_model)
                        print('(Average) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi_nonchild[2], epoch))
                        print('(Lung) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi_nonchild[2], epoch))
                        print('(Disease) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi_nonchild[2], epoch))                        
                    elif args.multitask_domain:
                        print('ICBHI Best')
                        print('(Average) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi[2], epoch))
                        print('(Lung) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi[2], epoch))
                        print('(Disease) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi[2], epoch))
                        print('(Domain) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain_icbhi[2], epoch))
                        
                        print('ICBHI-Child test results')
                        best_average_icbhi_child, best_lung_icbhi_child, best_disease_icbhi_child, _, _ = validate_multitask_domain(val_loader_icbhi_child, model, classifier, criterion, args, best_average_icbhi_child, best_lung_icbhi_child, best_disease_icbhi_child, best_domain_icbhi_child, best_model)
                        print('(Average) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi_child[2], epoch))
                        print('(Lung) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi_child[2], epoch))
                        print('(Disease) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi_child[2], epoch))
                        print('(Domain) ICBHI-Child Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi_child[2], epoch))
                        
                        print('ICBHI-Non Child test results')
                        best_average_icbhi_nonchild, best_lung_icbhi_nonchild, best_disease_icbhi_nonchild, best_domain_icbhi_nonchild, _, _ = validate_multitask_domain(val_loader_icbhi_nonchild, model, classifier, criterion, args, best_average_icbhi_nonchild, best_lung_icbhi_nonchild, best_disease_icbhi_nonchild, best_domain_icbhi_nonchild, best_model)
                        print('(Average) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_icbhi_nonchild[2], epoch))
                        print('(Lung) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_icbhi_nonchild[2], epoch))
                        print('(Disease) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_icbhi_nonchild[2], epoch))
                        print('(Domain) ICBHI-NonChild Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain_icbhi_nonchild[2], epoch))
                    else:
                        #print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_icbhi[2], epoch))
                        print('ICBHI Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_icbhi[2], epoch))
                        print('ICBHI-Child test results')
                        best_acc_icbhi_child = validate_each(val_loader_icbhi_child, model, classifier, criterion, args)
                        print('ICBHI-Non Child test results')
                        best_acc_icbhi_nonchild = validate_each(val_loader_icbhi_nonchild, model, classifier, criterion, args)
                        print('ICBHI-Child {} Non-Child {}'.format(best_acc_icbhi_child[2], best_acc_icbhi_nonchild[2]))
                    
                    save_file_icbhi = os.path.join(args.save_folder, 'icbhi_best_epoch_{}.pth'.format(epoch))
                    if args.multitask:
                        save_model_multitask(model, optimizer, args, epoch, save_file_icbhi, classifier[0], classifier[1])
                    elif args.multitask_domain:
                        save_model_multitask_domain(model, optimizer, args, epoch, save_file_icbhi, classifier[0], classifier[1], classifier[2])
                    else:
                        save_model(model, optimizer, args, epoch, save_file_icbhi, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
                
                if args.multitask:
                    best_average_snubh, best_lung_snubh, best_disease_snubh, best_model_snubh, save_bool_snubh = validate_multitask(val_loader_snubh, model, classifier, criterion, args, best_average_snubh, best_lung_snubh, best_disease_snubh, best_model_snubh)
                elif args.multitask_domain:
                    best_average_snubh, best_lung_snubh, best_disease_snubh, best_domain_snubh, best_model_snubh, save_bool_snubh = validate_multitask_domain(val_loader_snubh, model, classifier, criterion, args, best_average_snubh, best_lung_snubh, best_disease_snubh, best_domain_snubh, best_model_snubh)
                else:                    
                    best_acc_snubh, best_model_snubh, save_bool_snubh = validate(val_loader_snubh, model, classifier, criterion, args, best_acc_snubh, best_model_snubh)
                                                        
                if save_bool_snubh:    
                    if args.multitask:
                        print('(Average) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_snubh[2], epoch))
                        print('(Lung) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_snubh[2], epoch))
                        print('(Disease) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_snubh[2], epoch))
                    elif args.multitask_domain:
                        print('(Average) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_snubh[2], epoch))
                        print('(Lung) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_snubh[2], epoch))
                        print('(Disease) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_snubh[2], epoch))
                        print('(Domain) SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain_snubh[2], epoch))
                    else:
                        print('SNUBH Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_snubh[2], epoch))                            
                    save_file_snubh = os.path.join(args.save_folder, 'snubh_best_epoch_{}.pth'.format(epoch))
                    if args.multitask:
                        save_model_multitask(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1])
                    elif args.multitask_domain:
                        save_model_multitask_domain(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1], classifier[2])
                    else:
                        save_model(model, optimizer, args, epoch, save_file_snubh, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
                
                if args.multitask:
                    best_average_smart, best_lung_smart, best_disease_smart, best_model_smart, save_bool_smart = validate_multitask(val_loader_smart, model, classifier, criterion, args, best_average_smart, best_lung_smart, best_disease_smart, best_model_smart)
                elif args.multitask_domain:
                    best_average_smart, best_lung_smart, best_disease_smart, best_domain_smart, best_model_smart, save_bool_smart = validate_multitask_domain(val_loader_smart, model, classifier, criterion, args, best_average_smart, best_lung_smart, best_disease_smart, best_domain_smart, best_model_smart)
                else:                    
                    best_acc_smart, best_model_smart, save_bool_smart = validate(val_loader_smart, model, classifier, criterion, args, best_acc_smart, best_model_smart)
                                                        
                if save_bool_smart:    
                    if args.multitask:
                        print('(Average) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_smart[2], epoch))
                        print('(Lung) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_smart[2], epoch))
                        print('(Disease) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_smart[2], epoch))
                    elif args.multitask_domain:
                        print('(Average) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average_smart[2], epoch))
                        print('(Lung) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung_smart[2], epoch))
                        print('(Disease) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease_smart[2], epoch))
                        print('(Domain) SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain_smart[2], epoch))
                    else:
                        print('SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_smart[2], epoch))                            
                    save_file_smart = os.path.join(args.save_folder, 'smart_best_epoch_{}.pth'.format(epoch))
                    if args.multitask:
                        save_model_multitask(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1])
                    elif args.multitask_domain:
                        save_model_multitask_domain(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1], classifier[2])
                    else:
                        save_model(model, optimizer, args, epoch, save_file_smart, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
                
                
            
            else:
                if args.multitask:
                    best_average, best_lung, best_disease, best_model, save_bool = validate_multitask(val_loader, model, classifier, criterion, args, best_average, best_lung, best_disease, best_model)
                elif args.multitask_domain:
                    best_average, best_lung, best_disease, best_domain, best_model, save_bool = validate_multitask_domain(val_loader, model, classifier, criterion, args, best_average, best_lung, best_disease, best_domain, best_model)
                else:
                    best_acc, best_model, save_bool = validate(val_loader, model, classifier, criterion, args, best_acc, best_model)
                
                if save_bool:            
                    if args.multitask:
                        print('(Average) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average[2], epoch))
                        print('(Lung) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung[2], epoch))
                        print('(Disease) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease[2], epoch))
                    elif args.multitask_domain:
                        print('(Average) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_average[2], epoch))
                        print('(Lung) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_lung[2], epoch))
                        print('(Disease) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_disease[2], epoch))
                        print('(Domain) Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_domain[2], epoch))
                    else:
                        print('SMART Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc_smart[2], epoch))
                    
                    save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
                    
                    if args.multitask:
                        save_model_multitask(model, optimizer, args, epoch, save_file, classifier[0], classifier[1])
                    elif args.multitask_domain:
                        save_model_multitask_domain(model, optimizer, args, epoch, save_file, classifier[0], classifier[1], classifier[2])
                    else:
                        save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
                    
            

        # save a checkpoint of classifier with the best accuracy or score
        if args.dataset in ['icbhi_snubh', 'icbhi_smart']:
            save_file_icbhi = os.path.join(args.save_folder, 'icbhi_best.pth')
            model.load_state_dict(best_model_icbhi[0])
            if args.multitask:
                classifier[0].load_state_dict(best_model_icbhi[1])
                classifier[1].load_state_dict(best_model_icbhi[2])
                save_model_multitask(model, optimizer, args, epoch, save_file_icbhi, classifier[0], classifier[1])
            elif args.multitask_domain:
                classifier[0].load_state_dict(best_model_icbhi[1])
                classifier[1].load_state_dict(best_model_icbhi[2])
                classifier[2].load_state_dict(best_model_icbhi[3])
                save_model_multitask_domain(model, optimizer, args, epoch, save_file_icbhi, classifier[0], classifier[1], classifier[2])
            else:
                #classifier.load_state_dict(best_model[1])
                classifier[0].load_state_dict(best_model_icbhi[1]) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(best_model_icbhi[1])
                save_model(model, optimizer, args, epoch, save_file_icbhi, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
            
            
            if second_data == 'snubh':
                save_file_snubh = os.path.join(args.save_folder, 'snubh_best.pth')
                model.load_state_dict(best_model_snubh[0])
                if args.multitask:
                    classifier[0].load_state_dict(best_model_snubh[1])
                    classifier[1].load_state_dict(best_model_snubh[2])
                    save_model_multitask(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1])
                elif args.multitask_domain:                    
                    classifier[0].load_state_dict(best_model_snubh[1])
                    classifier[1].load_state_dict(best_model_snubh[2])
                    classifier[2].load_state_dict(best_model_snubh[3])
                    save_model_multitask_domain(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1], classifier[2])
                else:
                    #classifier.load_state_dict(best_model[1])
                    classifier[0].load_state_dict(best_model_snubh[1]) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(best_model_snubh[1])
                    save_model(model, optimizer, args, epoch, save_file_snubh, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
            
            elif second_data == 'smart':
                save_file_smart = os.path.join(args.save_folder, 'smart_best.pth')
                model.load_state_dict(best_model_smart[0])              
                if args.multitask:
                    classifier[0].load_state_dict(best_model_smart[1])
                    classifier[1].load_state_dict(best_model_smart[2])
                    save_model_multitask(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1])
                elif args.multitask_domain:                    
                    classifier[0].load_state_dict(best_model_smart[1])
                    classifier[1].load_state_dict(best_model_smart[2])
                    classifier[2].load_state_dict(best_model_smart[3])
                    save_model_multitask_domain(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1], classifier[2])
                else:
                    classifier[0].load_state_dict(best_model_smart[1]) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(best_model_smart[1])
                    save_model(model, optimizer, args, epoch, save_file_smart, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
        
        elif args.dataset == 'snubh_smart':
            save_file_snubh = os.path.join(args.save_folder, 'snubh_best.pth')
            model.load_state_dict(best_model_snubh[0])
            if args.multitask:
                classifier[0].load_state_dict(best_model_snubh[1])
                classifier[1].load_state_dict(best_model_snubh[2])
                save_model_multitask(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1])
            elif args.multitask_domain:                    
                classifier[0].load_state_dict(best_model_snubh[1])
                classifier[1].load_state_dict(best_model_snubh[2])
                classifier[2].load_state_dict(best_model_snubh[3])
                save_model_multitask_domain(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1], classifier[2])
            else:
                classifier[0].load_state_dict(best_model_snubh[1]) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(best_model_snubh[1])
                save_model(model, optimizer, args, epoch, save_file_snubh, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
            
            save_file_smart = os.path.join(args.save_folder, 'smart_best.pth')
            model.load_state_dict(best_model_smart[0])
            if args.multitask:
                classifier[0].load_state_dict(best_model_smart[1])
                classifier[1].load_state_dict(best_model_smart[2])
                save_model_multitask(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1])
            elif args.multitask_domain:                    
                classifier[0].load_state_dict(best_model_smart[1])
                classifier[1].load_state_dict(best_model_smart[2])
                classifier[2].load_state_dict(best_model_smart[3])
                save_model_multitask_domain(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1], classifier[2])
            else:
                classifier[0].load_state_dict(best_model_smart[1]) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(best_model_smart[1])
                save_model(model, optimizer, args, epoch, save_file_smart, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
        
        elif args.dataset == 'all':
            save_file_icbhi = os.path.join(args.save_folder, 'icbhi_best.pth')
            model.load_state_dict(best_model_icbhi[0])
            if args.multitask:
                classifier[0].load_state_dict(best_model_icbhi[1])
                classifier[1].load_state_dict(best_model_icbhi[2])
                save_model_multitask(model, optimizer, args, epoch, save_file_icbhi, classifier[0], classifier[1])
            elif args.multitask_domain:                    
                classifier[0].load_state_dict(best_model_icbhi[1])
                classifier[1].load_state_dict(best_model_icbhi[2])
                classifier[2].load_state_dict(best_model_icbhi[3])
                save_model_multitask_domain(model, optimizer, args, epoch, save_file_icbhi, classifier[0], classifier[1], classifier[2])
            else:
                classifier[0].load_state_dict(best_model_icbhi[1]) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(best_model_icbhi[1])
                save_model(model, optimizer, args, epoch, save_file_icbhi, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
            
            save_file_snubh = os.path.join(args.save_folder, 'snubh_best.pth')
            model.load_state_dict(best_model_snubh[0])
            if args.multitask:
                classifier[0].load_state_dict(best_model_snubh[1])
                classifier[1].load_state_dict(best_model_snubh[2])
                save_model_multitask(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1])
            elif args.multitask_domain:                    
                classifier[0].load_state_dict(best_model_snubh[1])
                classifier[1].load_state_dict(best_model_snubh[2])
                classifier[2].load_state_dict(best_model_snubh[3])
                save_model_multitask_domain(model, optimizer, args, epoch, save_file_snubh, classifier[0], classifier[1], classifier[2])
            else:
                classifier[0].load_state_dict(best_model_snubh[1]) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(best_model_snubh[1])
                save_model(model, optimizer, args, epoch, save_file_snubh, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
            
            save_file_smart = os.path.join(args.save_folder, 'smart_best.pth')
            model.load_state_dict(best_model_smart[0])
            if args.multitask:
                classifier[0].load_state_dict(best_model_smart[1])
                classifier[1].load_state_dict(best_model_smart[2])
                save_model_multitask(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1])
            elif args.multitask_domain:                    
                classifier[0].load_state_dict(best_model_smart[1])
                classifier[1].load_state_dict(best_model_smart[2])
                classifier[2].load_state_dict(best_model_smart[3])
                save_model_multitask_domain(model, optimizer, args, epoch, save_file_smart, classifier[0], classifier[1], classifier[2])
            else:
                classifier[0].load_state_dict(best_model_smart[1]) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(best_model_smart[1])
                save_model(model, optimizer, args, epoch, save_file_smart, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
        
        else:
            save_file = os.path.join(args.save_folder, 'best.pth')
            model.load_state_dict(best_model[0])
            if args.multitask:
                classifier[0].load_state_dict(best_model[1])
                classifier[1].load_state_dict(best_model[2])
                save_model_multitask(model, optimizer, args, epoch, save_file, classifier[0], classifier[1])
            elif args.multitask_domain:                    
                classifier[0].load_state_dict(best_model[1])
                classifier[1].load_state_dict(best_model[2])
                classifier[2].load_state_dict(best_model[3])
                save_model_multitask_domain(model, optimizer, args, epoch, save_file, classifier[0], classifier[1], classifier[2])
            else:
                classifier[0].load_state_dict(best_model[1]) if args.domain_adaptation or args.domain_adaptation2 else classifier.load_state_dict(best_model[1])
                save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.domain_adaptation or args.domain_adaptation2 else classifier)
            
                
    #eval code. TBD later
    else: # have to modified
        if args.dataset == 'icbhi_snubh':
            val_loader_icbhi = val_loader[0]
            val_loader_snubh = val_loader[1]
            val_loader_icbhi_child = val_loader[2]
            val_loader_icbhi_nonchild = val_loader[3]
            #best_acc_both, best_model, save_bool = validate(val_loader_both, model, classifier, criterion, args, best_acc_both, best_model)
            best_acc_icbhi = validate_each(val_loader_icbhi, model, classifier, criterion, args)
            best_acc_snubh = validate_each(val_loader_snubh, model, classifier, criterion, args)
            best_acc_icbhi_child = validate_each(val_loader_icbhi_child, model, classifier, criterion, args)
            best_acc_icbhi_nonchild = validate_each(val_loader_icbhi_nonchild, model, classifier, criterion, args)
        else:
            best_acc, _, _  = validate(val_loader, model, classifier, criterion, args, best_acc)
    
    print('{} finished'.format(args.model_name))
    
    if args.dataset in ['icbhi_snubh', 'icbhi_smart']:
        if args.domain_adaptation:            
            update_json('%s' % args.model_name+'_icbhi', best_acc_icbhi, path=os.path.join(args.save_dir, 'results_da.json'))
            if second_data == 'snubh':
                update_json('%s' % args.model_name+'_snubh', best_acc_snubh, path=os.path.join(args.save_dir, 'results_da.json'))
            elif second_data == 'smart':
                update_json('%s' % args.model_name+'_smart', best_acc_smart, path=os.path.join(args.save_dir, 'results_da.json'))
            update_json('%s' % args.model_name+'_icbhi_child', best_acc_icbhi_child, path=os.path.join(args.save_dir, 'results_da.json'))
            update_json('%s' % args.model_name+'_icbhi_nonchild', best_acc_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_da.json'))
        elif args.domain_adaptation2:            
            update_json('%s' % args.model_name+'_icbhi', best_acc_icbhi, path=os.path.join(args.save_dir, 'results_da2.json'))
            if second_data == 'snubh':
                update_json('%s' % args.model_name+'_snubh', best_acc_snubh, path=os.path.join(args.save_dir, 'results_da2.json'))
            elif second_data == 'smart':
                update_json('%s' % args.model_name+'_smart', best_acc_smart, path=os.path.join(args.save_dir, 'results_da2.json'))
            update_json('%s' % args.model_name+'_icbhi_child', best_acc_icbhi_child, path=os.path.join(args.save_dir, 'results_da2.json'))
            update_json('%s' % args.model_name+'_icbhi_nonchild', best_acc_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_da2.json'))
        
        elif args.multitask:                
            update_json('%s' % args.model_name+'_icbhi_avg', best_average_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_icbhi_lung', best_lung_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_disease', best_disease_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            if second_data == 'snubh':
                update_json('%s' % args.model_name+'_snubh_avg', best_average_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
                update_json('%s' % args.model_name+'_snubh_lung', best_lung_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
                update_json('%s' % args.model_name+'_snubh_disease', best_disease_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
                                
            elif second_data == 'smart':
                update_json('%s' % args.model_name+'_smart_avg', best_average_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
                update_json('%s' % args.model_name+'_smart_lung', best_lung_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
                update_json('%s' % args.model_name+'_smart_disease', best_disease_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_icbhi_child_avg', best_average_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE            
            update_json('%s' % args.model_name+'_icbhi_child_lung', best_lung_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_child_disease', best_disease_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_nonchild_avg', best_average_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE            
            update_json('%s' % args.model_name+'_icbhi_nonchild_lung', best_lung_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_nonchild_disease', best_disease_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            
            
        elif args.multitask_domain:
            update_json('%s' % args.model_name+'_icbhi_avg', best_average_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_icbhi_lung', best_lung_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_disease', best_disease_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_domain', best_domain_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            if second_data == 'snubh':
                update_json('%s' % args.model_name+'_snubh_avg', best_average_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
                update_json('%s' % args.model_name+'_snubh_lung', best_lung_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
                update_json('%s' % args.model_name+'_snubh_disease', best_disease_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
                update_json('%s' % args.model_name+'_snubh_domain', best_domain_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
                                
            if second_data == 'snubh':
                update_json('%s' % args.model_name+'_smart_avg', best_average_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
                update_json('%s' % args.model_name+'_smart_lung', best_lung_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
                update_json('%s' % args.model_name+'_smart_disease', best_disease_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
                update_json('%s' % args.model_name+'_smart_domain', best_domain_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_icbhi_child_avg', best_average_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE            
            update_json('%s' % args.model_name+'_icbhi_child_lung', best_lung_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_child_disease', best_disease_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_child_domain', best_domain_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            
            update_json('%s' % args.model_name+'_icbhi_nonchild_avg', best_average_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE            
            update_json('%s' % args.model_name+'_icbhi_nonchild_lung', best_lung_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_nonchild_disease', best_disease_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_nonchild_domain', best_domain_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
        
        
        else:
            update_json('%s' % args.model_name+'_icbhi', best_acc_icbhi, path=os.path.join(args.save_dir, 'results.json'))
            if second_data == 'snubh':
                update_json('%s' % args.model_name+'_snubh', best_acc_snubh, path=os.path.join(args.save_dir, 'results.json'))
            elif second_data == 'smart':
                update_json('%s' % args.model_name+'_smart', best_acc_smart, path=os.path.join(args.save_dir, 'results.json'))
            update_json('%s' % args.model_name+'_icbhi_child', best_acc_icbhi_child, path=os.path.join(args.save_dir, 'results.json'))
            update_json('%s' % args.model_name+'_icbhi_nonchild', best_acc_icbhi_nonchild, path=os.path.join(args.save_dir, 'results.json'))
    
    elif args.dataset == 'snubh_smart':
        if args.domain_adaptation:
            update_json('%s' % args.model_name+'_snubh', best_acc_snubh, path=os.path.join(args.save_dir, 'results_da.json'))
            update_json('%s' % args.model_name+'_smart', best_acc_smart, path=os.path.join(args.save_dir, 'results_da.json'))
        elif args.domain_adaptation2:
            update_json('%s' % args.model_name+'_snubh', best_acc_snubh, path=os.path.join(args.save_dir, 'results_da2.json'))
            update_json('%s' % args.model_name+'_smart', best_acc_smart, path=os.path.join(args.save_dir, 'results_da2.json'))
        
        
        elif args.multitask:                
            update_json('%s' % args.model_name+'_snubh_avg', best_average_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_snubh_lung', best_lung_snubh, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_snubh_disease', best_disease_snubh, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_smart_avg', best_average_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_lung', best_lung_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_disease', best_disease_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            
            
        elif args.multitask_domain:
            update_json('%s' % args.model_name+'_snubh_avg', best_average_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_snubh_lung', best_lung_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_snubh_disease', best_disease_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_snubh_domain', best_domain_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_avg', best_average_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_lung', best_lung_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_disease', best_disease_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_domain', best_domain_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            
        
        else:
            update_json('%s' % args.model_name+'_snubh', best_acc_snubh, path=os.path.join(args.save_dir, 'results.json'))
            update_json('%s' % args.model_name+'_smart', best_acc_smart, path=os.path.join(args.save_dir, 'results.json'))
    
    elif args.dataset == 'all':
        if args.domain_adaptation:
            update_json('%s' % args.model_name+'_icbhi', best_acc_icbhi, path=os.path.join(args.save_dir, 'results_da.json'))
            update_json('%s' % args.model_name+'_icbhi_child', best_acc_icbhi_child, path=os.path.join(args.save_dir, 'results_da.json'))
            update_json('%s' % args.model_name+'_icbhi_nonchild', best_acc_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_da.json'))
            update_json('%s' % args.model_name+'_snubh', best_acc_snubh, path=os.path.join(args.save_dir, 'results_da.json'))
            update_json('%s' % args.model_name+'_smart', best_acc_smart, path=os.path.join(args.save_dir, 'results_da.json'))
        
        elif args.domain_adaptation2:
            update_json('%s' % args.model_name+'_icbhi', best_acc_icbhi, path=os.path.join(args.save_dir, 'results_da2.json'))
            update_json('%s' % args.model_name+'_icbhi_child', best_acc_icbhi_child, path=os.path.join(args.save_dir, 'results_da2.json'))
            update_json('%s' % args.model_name+'_icbhi_nonchild', best_acc_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_da2.json'))
            update_json('%s' % args.model_name+'_snubh', best_acc_snubh, path=os.path.join(args.save_dir, 'results_da2.json'))
            update_json('%s' % args.model_name+'_smart', best_acc_smart, path=os.path.join(args.save_dir, 'results_da2.json'))
        
        elif args.multitask:                
            update_json('%s' % args.model_name+'_icbhi_avg', best_average_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_icbhi_lung', best_lung_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_disease', best_disease_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            
            update_json('%s' % args.model_name+'_snubh_avg', best_average_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_snubh_lung', best_lung_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_snubh_disease', best_disease_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
                                
            update_json('%s' % args.model_name+'_smart_avg', best_average_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_lung', best_lung_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_disease', best_disease_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            
            update_json('%s' % args.model_name+'_icbhi_child_avg', best_average_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE            
            update_json('%s' % args.model_name+'_icbhi_child_lung', best_lung_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_child_disease', best_disease_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            
            update_json('%s' % args.model_name+'_icbhi_nonchild_avg', best_average_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE            
            update_json('%s' % args.model_name+'_icbhi_nonchild_lung', best_lung_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_nonchild_disease', best_disease_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            
            
        elif args.multitask_domain:
            update_json('%s' % args.model_name+'_icbhi_avg', best_average_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_icbhi_lung', best_lung_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_disease', best_disease_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_domain', best_domain_icbhi, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            
            update_json('%s' % args.model_name+'_snubh_avg', best_average_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_snubh_lung', best_lung_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_snubh_disease', best_disease_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_snubh_domain', best_domain_snubh, path=os.path.join(args.save_dir, 'results_multitask.json'))
            
            update_json('%s' % args.model_name+'_smart_avg', best_average_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_lung', best_lung_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_disease', best_disease_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            update_json('%s' % args.model_name+'_smart_domain', best_domain_smart, path=os.path.join(args.save_dir, 'results_multitask.json'))
            
            update_json('%s' % args.model_name+'_icbhi_child_avg', best_average_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE            
            update_json('%s' % args.model_name+'_icbhi_child_lung', best_lung_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_child_disease', best_disease_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_child_domain', best_domain_icbhi_child, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            
            update_json('%s' % args.model_name+'_icbhi_nonchild_avg', best_average_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE            
            update_json('%s' % args.model_name+'_icbhi_nonchild_lung', best_lung_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_nonchild_disease', best_disease_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
            update_json('%s' % args.model_name+'_icbhi_nonchild_domain', best_domain_icbhi_nonchild, path=os.path.join(args.save_dir, 'results_multitask.json')) #CE
        
        else:
            update_json('%s' % args.model_name+'_icbhi', best_acc_icbhi, path=os.path.join(args.save_dir, 'results.json'))
            update_json('%s' % args.model_name+'_icbhi_child', best_acc_icbhi_child, path=os.path.join(args.save_dir, 'results.json'))
            update_json('%s' % args.model_name+'_icbhi_nonchild', best_acc_icbhi_nonchild, path=os.path.join(args.save_dir, 'results.json'))
            update_json('%s' % args.model_name+'_snubh', best_acc_snubh, path=os.path.join(args.save_dir, 'results.json'))
            update_json('%s' % args.model_name+'_smart', best_acc_smart, path=os.path.join(args.save_dir, 'results.json'))
    
    
    else:
        if args.domain_adaptation:
            update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results_da.json'))
        elif args.domain_adaptation2:
            update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results_da2.json'))
        else:
            update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))
    
    
if __name__ == '__main__':
    main()
