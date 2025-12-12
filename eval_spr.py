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
from util.spr_dataset import SPRSoundDataset


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
    
    if args.dataset == 'spr':
        val_dataset = SPRSoundDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)
    elif args.dataset == 'hflung_v1':
        val_dataset = HFDataset(train_flag=True, transform=train_transform, args=args, print_flag=True)    
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return val_loader, args
    
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
    projector = Projector(model.final_feat_dim, args.proj_dim) if args.method in ['patchmix_cl'] or args.domain_adaptation2 else nn.Identity()
    classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
           
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
        
    return model, classifier, projector




def validate(val_loader, model, classifier, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    
    if args.domain_adaptation or args.domain_adaptation2:
        classifier = classifier[0]
    classifier.eval()

    batch_time = AverageMeter()
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
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time, top1=top1))
    
    if sc > best_acc[-1] and se > 5:
        save_bool = True
        best_acc = [sp, se, sc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[-1]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool



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
    
    best_acc = [0, 0, 0]
    test_loader, args = set_loader(args)
    model, classifier, projector = set_model(args)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler()
    
    print('*' * 20)
    
    best_acc, _, _ = validate(test_loader, model, classifier, args, best_acc)
    
    
    print('{} finished'.format(args.model_name))
    print('best_acc', best_acc)
    
    
    
if __name__ == '__main__':
    main()
