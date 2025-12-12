import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--name', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    parser.add_argument('--two_cls_eval', action='store_true',
                        help='evaluate with two classes')
    parser.add_argument('--eval_pediatrics', action='store_true',
                        help='divide test set in terms of children or not')
    parser.add_argument('--eval_non_pediatrics', action='store_true',
                        help='divide test set in terms of children or not')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--measure_all', action='store_true', help='measure inclduing icbhi child-nonchild or not')
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='wheeze',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases), wheeze: (wheeze vs others)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--m_cls', type=int, default=0,
                        help='set k-way classification problem for domain (meta)')
    parser.add_argument('--multitask', action='store_true')
    parser.add_argument('--multitask_domain', action='store_true')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=8, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0, 
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])
    parser.add_argument('--weight1', type=float, default=1.0)
    parser.add_argument('--weight2', type=float, default=1.0)
    parser.add_argument('--weight3', type=float, default=1.0)
    parser.add_argument('--lung_first', action='store_true')
    parser.add_argument('--lung_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--disease_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--domain_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--mtl_meta', type=str, default='none',
                        help='the meta information for selecting', choices=['age', 'sex', 'loc', 'dev', 'age_sex', 'age_loc', 'age_dev', 
                            'sex_loc', 'sex_dev', 'loc_dev', 'age_sex_loc', 'age_sex_dev', 'age_loc_dev', 'sex_loc_dev', 'all'])
    
    parser.add_argument('--weighted_loss_diagnosis_only', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    parser.add_argument('--weighted_loss_domain_only', action='store_true',
                        help='weighted cross entropy loss (higher weights on abnormal class)')
    parser.add_argument('--accum', action='store_true')
    parser.add_argument('--accum_steps', type=int, default=2)

    # model
    parser.add_argument('--model', type=str, default='ast')
    parser.add_argument('--pretrained', action='store_true')
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    # for AST
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')
    # for SSAST
    parser.add_argument('--ssast_task', type=str, default='ft_avgtok', 
                        help='pretraining or fine-tuning task', choices=['ft_avgtok', 'ft_cls'])
    parser.add_argument('--fshape', type=int, default=16, 
                        help='fshape of SSAST')
    parser.add_argument('--tshape', type=int, default=16, 
                        help='tshape of SSAST')
    parser.add_argument('--ssast_pretrained_type', type=str, default='Patch', 
                        help='pretrained ckpt version of SSAST model')

    parser.add_argument('--method', type=str, default='ce')
    
    parser.add_argument('--arc1', action='store_true')
    parser.add_argument('--arc2', action='store_true')
    parser.add_argument('--domain_adaptation', action='store_true')
    parser.add_argument('--domain_adaptation2', action='store_true')
    # Meta Domain CL & Patch-Mix CL loss
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--mix_beta', default=1.0, type=float,
                        help='patch-mix interpolation coefficient')
    parser.add_argument('--time_domain', action='store_true',
                        help='patchmix for the specific time domain')

    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--negative_pair', type=str, default='all',
                        help='the method for selecting negative pair', choices=['all', 'diff_label'])
    parser.add_argument('--target_type', type=str, default='project1_project2block', help='how to make target representation',
                        choices=['project_flow_all', 'representation_all', 'z1block_project', 'z1_project2', 'project1block_project2', 'project1_r2block', 'project1_r2', 'project1_project2block', 'project_block_all'])
    
    # Meta for SCL
    parser.add_argument('--meta_mode', type=str, default='none', help='the meta information for selecting', 
        choices=['none', 'age', 'data', 'dev', 'loc', 'sex', 
        'age_data', 'age_dev', 'age_loc', 'age_sex', 'data_dev', 'data_loc', 'data_sex', 'dev_loc', 'dev_sex', 'loc_sex', 
        'age_data_dev', 'age_data_loc', 'age_data_sex', 'age_dev_loc', 'age_dev_sex', 'age_loc_sex', 'data_dev_loc', 'data_dev_sex', 'data_loc_sex', 'dev_loc_sex',
        'age_data_dev_loc', 'age_data_dev_sex', 'age_dev_loc_sex', 'data_dev_loc_sex',
        'age_data_dev_loc_sex'])
    
    parser.add_argument('--meta_all', action='store_true')
    parser.add_argument('--meta_all_supcon', action='store_true')
    parser.add_argument('--meta_all_cls', type=int, default=18)
    parser.add_argument('--bce', action='store_true')
    parser.add_argument('--mlm', action='store_true')
    parser.add_argument('--meta_weights', action='store_true')
    
    parser.add_argument('--no_reverse', action='store_true')
    parser.add_argument('--polarization', action='store_true')
    parser.add_argument('--equalization', action='store_true')
    parser.add_argument('--equalization_ver2', action='store_true')
    parser.add_argument('--meta_scale', action='store_true')
                        
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method) if args.meta_mode == 'none' else '{}_{}_{}_{}'.format(args.dataset, args.model, args.method, args.meta_mode)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    if args.method in ['patchmix', 'patchmix_cl']:
        assert args.model in ['ast', 'ssast']
    if args.domain_adaptation:
        args.save_folder = os.path.join(args.save_dir, 'da', args.model_name)
    elif args.domain_adaptation2:
        args.save_folder =  os.path.join(args.save_dir, 'da2', args.model_name)
    else:
        args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
        
            
    if args.class_split == 'wheeze':
        if args.domain_adaptation or args.domain_adaptation2:
            if args.meta_mode == 'age':
                args.meta_cls_list = ['Adult', 'Child']
                args.m_cls = 2
            elif args.meta_mode == 'sex':
                args.meta_cls_list = ['Male', 'Female']
                args.m_cls = 2
            elif args.meta_mode == 'loc':
                args.meta_cls_list = ['Tc', 'Al', 'Ar', 'Pl', 'Pr', 'Ll', 'Lr']
                args.m_cls = 7
            elif args.meta_mode == 'dev':
                args.meta_cls_list = ['Meditron', 'LittC2SE', 'Litt3200', 'AKGC417L', 'Jabes', 'SM-300']
                args.m_cls = 6
            elif args.meta_mode == 'data':
                args.meta_cls_list = ['ICBHI', 'SNUBH', 'SMART']
                args.m_cls = 3
        
        
        args.cls_list = ['Others', 'Wheeze']
    
    elif args.class_split == 'diagnosis':
        if args.n_cls == 3:
            args.cls_list = ['healthy', 'airway disease', 'lung parenchymal disease']
        elif args.n_cls == 2:
            args.cls_list = ['healthy', 'unhealthy']
        else:
            raise NotImplementedError
    
    elif args.class_split == 'multitask':
        if args.lung_cls == 2:
            args.lung_list = ['Others','Wheeze']        
        if args.disease_cls == 3:
            args.disease_list = ['healthy', 'airway disease', 'lung parenchymal disease']
        elif args.disease_cls == 2:
            args.disease_list = ['healthy', 'unhealthy']
    
    
    elif args.class_split == 'multitask_domain':
        if args.lung_cls == 2:
            args.lung_list = ['Others','Wheeze']        
        if args.disease_cls == 3:
            args.disease_list = ['healthy', 'airway disease', 'lung parenchymal disease']
        elif args.disease_cls == 2:
            args.disease_list = ['healthy', 'unhealthy']
        
        if args.mtl_meta == 'age':
            args.domain_list = ['Adult', 'Child']
            args.domain_cls = 2
        elif args.mtl_meta == 'sex':
            args.domain_list = ['Male', 'Female']
            args.domain_cls = 2
        elif args.mtl_meta == 'loc':
            args.domain_list = ['Tc', 'Al', 'Ar', 'Pl', 'Pr', 'Ll', 'Lr']
            args.domain_cls = 7
        elif args.mtl_meta == 'dev':
            args.domain_list = ['Meditron', 'LittC2SE', 'Litt3200', 'AKGC417L', 'Jabes', 'SM-300']
            args.domain_cls = 6
        elif args.mtl_meta == 'data':
            args.domain_cls = ['ICBHI', 'SNUBH', 'SMART']
            args.domain_cls = 3
    
        
    else:
        raise NotImplementedError
    
    if args.n_cls == 0 and args.m_cls !=0:
        args.n_cls = args.m_cls
        args.cls_list = args.meta_cls_list

    return args