from collections import namedtuple
import os
import math
import random
from tkinter import W
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torchaudio
from torchaudio import transforms as T

from .augmentation import augment_raw_audio

__all__ = ['get_annotations', 'save_image', 'get_mean_and_std', 'get_individual_cycles_torchaudio', 'generate_mel_spectrogram', 'generate_fbank', 'get_score']


# ==========================================================================
""" ICBHI dataset information """
def _extract_lungsound_annotation(file_name, data_folder):
    tokens = file_name.strip().split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient Number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(data_folder, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')

    return recording_info, recording_annotations


def get_annotations(args, data_folder):
    if args.class_split == 'lungsound' or args.class_split in ['lungsound_meta', 'meta', 'wheeze']: # 4-class
        filenames = sorted(glob(data_folder+'/*')) #--> 1840: 920 wav + 920 txt
        filenames = set(f.strip().split('/')[-1].split('.')[0] for f in filenames if '.txt' in f)
        filenames = sorted(list(set(filenames))) #--> 920

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            annotation_dict[f] = ann

    elif args.class_split == 'diagnosis':
        filenames = sorted(glob(data_folder+'/*')) #--> 1840: 920 wav + 920 txt
        filenames = set(f.strip().split('/')[-1].split('.')[0] for f in filenames if '.txt' in f)
        filenames = sorted(list(set(filenames))) #--> 920
        tmp = pd.read_csv(os.path.join(args.data_folder, 'icbhi_dataset/patient_diagnosis.txt'), names=['Disease'], delimiter='\t')

        annotation_dict = {}
        for f in filenames:
            info, ann = _extract_lungsound_annotation(f, data_folder)
            ann.drop(['Crackles', 'Wheezes'], axis=1, inplace=True)

            disease = tmp.loc[int(f.strip().split('_')[0]), 'Disease']
            ann['Disease'] = disease

            annotation_dict[f] = ann
            
    return annotation_dict


def save_image(image, fpath):
    save_dir = os.path.join(fpath, 'image.jpg')
    cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    

def get_mean_and_std(dataset):
    """ Compute the mean and std value of mel-spectrogram """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

    cnt = 0
    fst_moment = torch.zeros(1)
    snd_moment = torch.zeros(1)
    for inputs, _, _ in dataloader:
        b, c, h, w = inputs.shape
        nb_pixels = b * h * w

        fst_moment += torch.sum(inputs, dim=[0,2,3])
        snd_moment += torch.sum(inputs**2, dim=[0,2,3])
        cnt += nb_pixels

    mean = fst_moment / cnt
    std = torch.sqrt(snd_moment/cnt - mean**2)

    return mean, std
# ==========================================================================


# ==========================================================================
""" data preprocessing """

def _get_lungsound_label(crackle, wheeze, n_cls, args):
    if args.class_split == 'wheeze':
        if wheeze == 1:
            return 1
        else:
            return 0
    else:
        
        if n_cls == 4:
            if crackle == 0 and wheeze == 0:
                return 0
            elif crackle == 1 and wheeze == 0:
                return 1
            elif crackle == 0 and wheeze == 1:
                return 2
            elif crackle == 1 and wheeze == 1:
                return 3
        
        elif n_cls == 2:
            if crackle == 0 and wheeze == 0:
                return 0
            else:
                return 1
    
    


def _get_diagnosis_label(disease, n_cls):
    if n_cls == 3:
        if disease in ['COPD', 'Bronchiectasis', 'Asthma']:
            return 1
        elif disease in ['URTI', 'LRTI', 'Pneumonia', 'Bronchiolitis']:
            return 2
        else:
            return 0

    elif n_cls == 2:
        if disease == 'Healthy':
            return 0
        else:
            return 1


def _slice_data_torchaudio(start, end, data, sample_rate):
    """
    SCL paper..
    sample_rate denotes how many sample points for one second
    """
    max_ind = data.shape[1]
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)

    return data[:, start_ind: end_ind]


def cut_pad_sample_torchaudio(data, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = args.desired_length * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
        if data.dim() == 1:
            data = data.unsqueeze(0)
    else:
        if args.pad_types == 'zero':
            tmp = torch.zeros(1, target_duration, dtype=torch.float32)
            diff = target_duration - data.shape[-1]
            tmp[..., diff//2:data.shape[-1]+diff//2] = data
            data = tmp
        elif args.pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data

def get_meta_infor(metadata, args, label):
    #print('meta_infor_len', label)
    meta_infor_len = len(metadata)
    #print('metadata', len(metadata), metadata)
    if meta_infor_len == 7:
        age = int(metadata[0])
        sex = int(metadata[1])
        loc = int(metadata[-2])
        dev = int(metadata[-1])
        data = 0
    
    elif meta_infor_len == 5:
        label = int(metadata[0])
        age = int(metadata[2])
        sex = int(metadata[1])
        dev = int(metadata[3])
        loc = int(metadata[-1])
        data = 1
    
    if args.meta_all:
        meta_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Adult, Pediatric, Male, Female, Meditron, LittC2SE, Litt3200, AKGC417L, Jabes, Tc, Al, Ar, Pl, Pr, Ll, Lr, ICBHI, SNUBH
        if meta_infor_len == 7:
            age = int(metadata[0])
            sex = int(metadata[1])
            loc = int(metadata[-2])
            dev = int(metadata[-1])
            data = 0
        elif meta_infor_len == 5:
            label = int(metadata[0])
            age = int(metadata[2])
            sex = int(metadata[1])
            dev = int(metadata[3])
            loc = int(metadata[-1])
            data = 1
        
        if age >= 18:
            meta_label[0] = 1
        else:
            meta_label[1] = 1
        
        if sex == 0:
            meta_label[2] = 1
        else:
            meta_label[3] = 1
        
        if dev == 0:
            meta_label[4] = 1
        elif dev == 1:
            meta_label[5] = 1
        elif dev == 2:
            meta_label[6] = 1
        elif dev == 3:
            meta_label[7] = 1
        elif dev == 4:
            meta_label[8] = 1
        
        if loc == 0:
            meta_label[9] = 1
        elif loc == 1:
            meta_label[10] = 1
        elif loc == 2:
            meta_label[11] = 1
        elif loc == 3:
            meta_label[12] = 1
        elif loc == 4:
            meta_label[13] = 1
        elif loc == 5:
            meta_label[14] = 1
        elif loc == 6:
            meta_label[15] = 1
        
        if data == 0:
            meta_label[16] = 1
        else:
            meta_label[17] = 1
        
        return meta_label
    
    if args.meta_all_supcon:
        if age >= 18:
            age_meta = 0
        else:
            age_meta = 1
        
        if sex == 0:
            sex_meta = 0
        else:
            sex_meta = 1
        
        dev_meta = dev
        loc_meta = loc
        data_meta = data
        
        return (age_meta, sex_meta, dev_meta, loc_meta, data_meta)
        
    if args.meta_mode == 'none':
        return None
    
    elif args.meta_mode == 'age': # Age Meta-SCL
        if age >= 18:
            meta_label = 0
        else:
            meta_label = 1
        
    elif args.meta_mode == 'sex':
        if sex == 0:
            meta_label = 0
        else:
            meta_label = 1
    
    elif args.meta_mode == 'loc':
        meta_label = loc
    
    elif args.meta_mode == 'dev':
        meta_label = dev
        #print('meta_label', meta_label)
    
    elif args.meta_mode == 'label':
        meta_label = label
    
    elif args.meta_mode == 'data':
        meta_label = data
    
    elif args.meta_mode == 'age_data':
        if age >= 19 and data == 0:
            meta_label = 0
        elif age >= 19 and data == 1:
            meta_label = 1
        elif age < 19 and data == 0:
            meta_label = 2
        elif age < 19 and data == 1:
            meta_label = 3
        else:
            raise ValueError
    
    elif args.meta_mode == 'age_dev':
        if age >= 19 and dev == 0: #Adult_Meditron
            meta_label = 0
        elif age >= 19 and dev == 1: #Adult_LittC2SE
            meta_label = 1
        elif age >= 19 and dev == 2: #Adult_Litt3200
            meta_label = 2
        elif age >= 19 and dev == 3: #Adult_AKGC417L
            meta_label = 3
        elif age >= 19 and dev == 4: #Adult_Jabes
            meta_label = 4
        elif age < 19 and dev == 0: #Child_Meditron
            meta_label = 5
        elif age < 19 and dev == 1: #Child_LittC2SE
            meta_label = 6
        elif age < 19 and dev == 2: #Child_Litt3200
            meta_label = 7
        elif age < 19 and dev == 3: #Child_AKGC417L
            meta_label = 8
        elif age < 19 and dev == 4: #Child_Jabes
            meta_label = 9
        else:
            raise ValueError
    
    elif args.meta_mode == 'age_loc':
        if age >= 19 and loc == 0:
            meta_label = 0
        elif age >= 19 and loc == 1:
            meta_label = 1
        elif age >= 19 and loc == 2:
            meta_label = 2
        elif age >= 19 and loc == 3:
            meta_label = 3
        elif age >= 19 and loc == 4:
            meta_label = 4
        elif age >= 19 and loc == 5:
            meta_label = 5
        elif age >= 19 and loc == 6:
            meta_label = 6
        elif age < 19 and loc == 0:
            meta_label = 7
        elif age < 19 and loc == 1:
            meta_label = 8
        elif age < 19 and loc == 2:
            meta_label = 9
        elif age < 19 and loc == 3:
            meta_label = 10
        elif age < 19 and loc == 4:
            meta_label = 11
        elif age < 19 and loc == 5:
            meta_label = 12
        elif age < 19 and loc == 6:
            meta_label = 13
        else:
            raise ValueError
    
    elif args.meta_mode == 'age_sex': ######### Age_Sex Meta-SCL
        if age >= 19 and sex == 0: #Adult_Male
            meta_label = 0
        elif age >= 19 and sex == 1: # Adult_Female
            meta_label = 1
        elif age < 19 and sex == 0: #Child_Male
            meta_label = 2
        elif age < 19 and sex == 1: #Child_Female
            meta_label = 3
        else:
            raise ValueError
    
    elif args.meta_mode == 'data_dev':
        if data == 0 and dev == 0:
            meta_label = 0
        elif data == 0 and dev == 1:
            meta_label = 1
        elif data == 0 and dev == 2:
            meta_label = 2
        elif data == 0 and dev == 3:
            meta_label = 3
        elif data == 0 and dev == 4:
            meta_label = 4
        elif data == 1 and dev == 0:
            meta_label = 5
        elif data == 1 and dev == 1:
            meta_label = 6
        elif data == 1 and dev == 2:
            meta_label = 7
        elif data == 1 and dev == 3:
            meta_label = 8
        elif data == 1 and dev == 4:
            meta_label = 9
        else:
            raise ValueError
    
    elif args.meta_mode == 'data_loc':
        if data == 0 and loc == 0:
            meta_label = 0
        elif data == 0 and loc == 1:
            meta_label = 1
        elif data == 0 and loc == 2:
            meta_label = 2
        elif data == 0 and loc == 3:
            meta_label = 3
        elif data == 0 and loc == 4:
            meta_label = 4
        elif data == 0 and loc == 5:
            meta_label = 5
        elif data == 0 and loc == 6:
            meta_label = 6
        elif data == 1 and loc == 0:
            meta_label = 7
        elif data == 1 and loc == 1:
            meta_label = 8
        elif data == 1 and loc == 2:
            meta_label = 9
        elif data == 1 and loc == 3:
            meta_label = 10
        elif data == 1 and loc == 4:
            meta_label = 11
        elif data == 1 and loc == 5:
            meta_label = 12
        elif data == 1 and loc == 6:
            meta_label = 13
        else:
            raise ValueError
    
    elif args.meta_mode == 'data_sex':
        if data == 0 and sex == 0: # ICBHI, Male
            meta_label = 0
        elif data == 0 and sex == 1: # ICBHI, Female
            meta_label = 1
        elif data == 1 and sex == 0: # SNUBH, Male
            meta_label = 2
        elif data == 1 and sex == 1: # SNUBH, Female
            meta_label = 3
        else:
            raise ValueError
    
    elif args.meta_mode == 'dev_loc':
        if dev == 0 and loc == 0:
            meta_label = 0
        elif dev == 0 and loc == 1:
            meta_label = 1
        elif dev == 0 and loc == 2:
            meta_label = 2
        elif dev == 0 and loc == 3:
            meta_label = 3
        elif dev == 0 and loc == 4:
            meta_label = 4
        elif dev == 0 and loc == 5:
            meta_label = 5
        elif dev == 0 and loc == 6:
            meta_label = 6
        elif dev == 1 and loc == 0:
            meta_label = 7
        elif dev == 1 and loc == 1:
            meta_label = 8
        elif dev == 1 and loc == 2:
            meta_label = 9
        elif dev == 1 and loc == 3:
            meta_label = 10
        elif dev == 1 and loc == 4:
            meta_label = 11
        elif dev == 1 and loc == 5:
            meta_label = 12
        elif dev == 1 and loc == 6:
            meta_label = 13
        elif dev == 2 and loc == 0:
            meta_label = 14
        elif dev == 2 and loc == 1:
            meta_label = 15
        elif dev == 2 and loc == 2:
            meta_label = 16
        elif dev == 2 and loc == 3:
            meta_label = 17
        elif dev == 2 and loc == 4:
            meta_label = 18
        elif dev == 2 and loc == 5:
            meta_label = 19
        elif dev == 2 and loc == 6:
            meta_label = 20
        elif dev == 3 and loc == 0:
            meta_label = 21
        elif dev == 3 and loc == 1:
            meta_label = 22
        elif dev == 3 and loc == 2:
            meta_label = 23
        elif dev == 3 and loc == 3:
            meta_label = 24
        elif dev == 3 and loc == 4:
            meta_label = 25
        elif dev == 3 and loc == 5:
            meta_label = 26
        elif dev == 3 and loc == 6:
            meta_label = 27
        elif dev == 4 and loc == 0:
            meta_label = 28
        elif dev == 4 and loc == 1:
            meta_label = 29
        elif dev == 4 and loc == 2:
            meta_label = 30
        elif dev == 4 and loc == 3:
            meta_label = 31
        elif dev == 4 and loc == 4:
            meta_label = 32
        elif dev == 4 and loc == 5:
            meta_label = 33
        elif dev == 4 and loc == 6:
            meta_label = 34
        else:
            raise ValueError
    
    
    
    elif args.meta_mode == 'dev_sex':
        if dev == 0 and sex == 0: #
            meta_label = 0
        elif dev == 0 and sex == 1: #
            meta_label = 1
        elif dev == 1 and sex == 0: #
            meta_label = 2
        elif dev == 1 and sex == 1: #
            meta_label = 3
        elif dev == 2 and sex == 0: #
            meta_label = 4
        elif dev == 2 and sex == 1: #
            meta_label = 5
        elif dev == 3 and sex == 0: #
            meta_label = 6
        elif dev == 3 and sex == 1: #
            meta_label = 7
        elif dev == 4 and sex == 0: #
            meta_label = 8
        elif dev == 4 and sex == 1: #
            meta_label = 9
        else:
            raise ValueError
            
    elif args.meta_mode == 'loc_sex':
        if loc == 0 and sex == 0: #
            meta_label = 0
        elif loc == 0 and sex == 1: #
            meta_label = 1
        elif loc == 1 and sex == 0: #
            meta_label = 2
        elif loc == 1 and sex == 1: #
            meta_label = 3
        elif loc == 2 and sex == 0: #
            meta_label = 4
        elif loc == 2 and sex == 1: #
            meta_label = 5
        elif loc == 3 and sex == 0: #
            meta_label = 6
        elif loc == 3 and sex == 1: #
            meta_label = 7
        elif loc == 4 and sex == 0: #
            meta_label = 8
        elif loc == 4 and sex == 1: #
            meta_label = 9
        elif loc == 5 and sex == 0: #
            meta_label = 10
        elif loc == 5 and sex == 1: #
            meta_label = 11
        elif loc == 6 and sex == 0: #
            meta_label = 12
        elif loc == 6 and sex == 1: #
            meta_label = 13
        else:
            raise ValueError
    
    elif args.meta_mode == 'age_data_dev':
        if age >= 19 and data == 0 and dev == 0:
            meta_label = 0
        elif age >= 19 and data == 0 and dev == 1:
            meta_label = 1
        elif age >= 19 and data == 0 and dev == 2:
            meta_label = 2
        elif age >= 19 and data == 0 and dev == 3:
            meta_label = 3
        elif age >= 19 and data == 0 and dev == 4:
            meta_label = 4
        elif age >= 19 and data == 1 and dev == 0:
            meta_label = 5
        elif age >= 19 and data == 1 and dev == 1:
            meta_label = 6
        elif age >= 19 and data == 1 and dev == 2:
            meta_label = 7
        elif age >= 19 and data == 1 and dev == 3:
            meta_label = 8
        elif age >= 19 and data == 1 and dev == 4:
            meta_label = 9
        elif age < 19 and data == 0 and dev == 0:
            meta_label = 10
        elif age < 19 and data == 0 and dev == 1:
            meta_label = 11
        elif age < 19 and data == 0 and dev == 2:
            meta_label = 12
        elif age < 19 and data == 0 and dev == 3:
            meta_label = 13
        elif age < 19 and data == 0 and dev == 4:
            meta_label = 14
        elif age < 19 and data == 1 and dev == 0:
            meta_label = 15
        elif age < 19 and data == 1 and dev == 1:
            meta_label = 16
        elif age < 19 and data == 1 and dev == 2:
            meta_label = 17
        elif age < 19 and data == 1 and dev == 3:
            meta_label = 18
        elif age < 19 and data == 1 and dev == 4:
            meta_label = 19
    
    elif args.meta_mode == 'age_data_loc':
        if age >= 19 and data == 0 and loc == 0:
            meta_label = 0
        elif age >= 19 and data == 0 and loc == 1:
            meta_label = 1
        elif age >= 19 and data == 0 and loc == 2:
            meta_label = 2
        elif age >= 19 and data == 0 and loc == 3:
            meta_label = 3
        elif age >= 19 and data == 0 and loc == 4:
            meta_label = 4
        elif age >= 19 and data == 0 and loc == 5:
            meta_label = 5
        elif age >= 19 and data == 0 and loc == 6:
            meta_label = 6
        elif age >= 19 and data == 1 and loc == 0:
            meta_label = 7
        elif age >= 19 and data == 1 and loc == 1:
            meta_label = 8
        elif age >= 19 and data == 1 and loc == 2:
            meta_label = 9
        elif age >= 19 and data == 1 and loc == 3:
            meta_label = 10
        elif age >= 19 and data == 1 and loc == 4:
            meta_label = 11
        elif age >= 19 and data == 1 and loc == 5:
            meta_label = 12
        elif age >= 19 and data == 1 and loc == 6:
            meta_label = 13
        elif age < 19 and data == 0 and loc == 0:
            meta_label = 14
        elif age < 19 and data == 0 and loc == 1:
            meta_label = 15
        elif age < 19 and data == 0 and loc == 2:
            meta_label = 16
        elif age < 19 and data == 0 and loc == 3:
            meta_label = 17
        elif age < 19 and data == 0 and loc == 4:
            meta_label = 18
        elif age < 19 and data == 0 and loc == 5:
            meta_label = 19
        elif age < 19 and data == 0 and loc == 6:
            meta_label = 20
        elif age < 19 and data == 1 and loc == 0:
            meta_label = 21
        elif age < 19 and data == 1 and loc == 1:
            meta_label = 22
        elif age < 19 and data == 1 and loc == 2:
            meta_label = 23
        elif age < 19 and data == 1 and loc == 3:
            meta_label = 24
        elif age < 19 and data == 1 and loc == 4:
            meta_label = 25
        elif age < 19 and data == 1 and loc == 5:
            meta_label = 26
        elif age < 19 and data == 1 and loc == 6:
            meta_label = 27
    
    elif args.meta_mode == 'age_data_sex':
        if age >= 19 and data == 0 and sex == 0:
            meta_label = 0
        elif age >= 19 and data == 0 and sex == 1:
            meta_label = 1
        elif age >= 19 and data == 1 and sex == 0:
            meta_label = 2
        elif age >= 19 and data == 1 and sex == 1:
            meta_label = 3
        elif age < 19 and data == 0 and sex == 0:
            meta_label = 4
        elif age < 19 and data == 0 and sex == 1:
            meta_label = 5
        elif age < 19 and data == 1 and sex == 0:
            meta_label = 6
        elif age < 19 and data == 1 and sex == 1:
            meta_label = 7
    
    elif args.meta_mode == 'age_dev_loc':
        if age >= 19 and dev == 0 and loc == 0:
            meta_label = 0
        elif age >= 19 and dev == 0 and loc == 1:
            meta_label = 1
        elif age >= 19 and dev == 0 and loc == 2:
            meta_label = 2
        elif age >= 19 and dev == 0 and loc == 3:
            meta_label = 3
        elif age >= 19 and dev == 0 and loc == 4:
            meta_label = 4
        elif age >= 19 and dev == 0 and loc == 5:
            meta_label = 5
        elif age >= 19 and dev == 0 and loc == 6:
            meta_label = 6
        elif age >= 19 and dev == 1 and loc == 0:
            meta_label = 7
        elif age >= 19 and dev == 1 and loc == 1:
            meta_label = 8
        elif age >= 19 and dev == 1 and loc == 2:
            meta_label = 9
        elif age >= 19 and dev == 1 and loc == 3:
            meta_label = 10
        elif age >= 19 and dev == 1 and loc == 4:
            meta_label = 11
        elif age >= 19 and dev == 1 and loc == 5:
            meta_label = 12
        elif age >= 19 and dev == 1 and loc == 6:
            meta_label = 13
        elif age >= 19 and dev == 2 and loc == 0:
            meta_label = 14
        elif age >= 19 and dev == 2 and loc == 1:
            meta_label = 15
        elif age >= 19 and dev == 2 and loc == 2:
            meta_label = 16
        elif age >= 19 and dev == 2 and loc == 3:
            meta_label = 17
        elif age >= 19 and dev == 2 and loc == 4:
            meta_label = 18
        elif age >= 19 and dev == 2 and loc == 5:
            meta_label = 19
        elif age >= 19 and dev == 2 and loc == 6:
            meta_label = 20
        elif age >= 19 and dev == 3 and loc == 0:
            meta_label = 21
        elif age >= 19 and dev == 3 and loc == 1:
            meta_label = 22
        elif age >= 19 and dev == 3 and loc == 2:
            meta_label = 23
        elif age >= 19 and dev == 3 and loc == 3:
            meta_label = 24
        elif age >= 19 and dev == 3 and loc == 4:
            meta_label = 25
        elif age >= 19 and dev == 3 and loc == 5:
            meta_label = 26
        elif age >= 19 and dev == 3 and loc == 6:
            meta_label = 27
        elif age >= 19 and dev == 4 and loc == 0:
            meta_label = 28
        elif age >= 19 and dev == 4 and loc == 1:
            meta_label = 28
        elif age >= 19 and dev == 4 and loc == 2:
            meta_label = 30
        elif age >= 19 and dev == 4 and loc == 3:
            meta_label = 31
        elif age >= 19 and dev == 4 and loc == 4:
            meta_label = 32
        elif age >= 19 and dev == 4 and loc == 5:
            meta_label = 33
        elif age >= 19 and dev == 4 and loc == 6:
            meta_label = 34
        elif age < 19 and dev == 0 and loc == 0:
            meta_label = 35
        elif age < 19 and dev == 0 and loc == 1:
            meta_label = 36
        elif age < 19 and dev == 0 and loc == 2:
            meta_label = 37
        elif age < 19 and dev == 0 and loc == 3:
            meta_label = 38
        elif age < 19 and dev == 0 and loc == 4:
            meta_label = 39
        elif age < 19 and dev == 0 and loc == 5:
            meta_label = 40
        elif age < 19 and dev == 0 and loc == 6:
            meta_label = 41
        elif age < 19 and dev == 1 and loc == 0:
            meta_label = 42
        elif age < 19 and dev == 1 and loc == 1:
            meta_label = 43
        elif age < 19 and dev == 1 and loc == 2:
            meta_label = 44
        elif age < 19 and dev == 1 and loc == 3:
            meta_label = 45
        elif age < 19 and dev == 1 and loc == 4:
            meta_label = 46
        elif age < 19 and dev == 1 and loc == 5:
            meta_label = 47
        elif age < 19 and dev == 1 and loc == 6:
            meta_label = 48
        elif age < 19 and dev == 2 and loc == 0:
            meta_label = 49
        elif age < 19 and dev == 2 and loc == 1:
            meta_label = 50
        elif age < 19 and dev == 2 and loc == 2:
            meta_label = 51
        elif age < 19 and dev == 2 and loc == 3:
            meta_label = 52
        elif age < 19 and dev == 2 and loc == 4:
            meta_label = 53
        elif age < 19 and dev == 2 and loc == 5:
            meta_label = 54
        elif age < 19 and dev == 2 and loc == 6:
            meta_label = 55
        elif age < 19 and dev == 3 and loc == 0:
            meta_label = 56
        elif age < 19 and dev == 3 and loc == 1:
            meta_label = 57
        elif age < 19 and dev == 3 and loc == 2:
            meta_label = 58
        elif age < 19 and dev == 3 and loc == 3:
            meta_label = 59
        elif age < 19 and dev == 3 and loc == 4:
            meta_label = 60
        elif age < 19 and dev == 3 and loc == 5:
            meta_label = 61
        elif age < 19 and dev == 3 and loc == 6:
            meta_label = 62
        elif age < 19 and dev == 4 and loc == 0:
            meta_label = 63
        elif age < 19 and dev == 4 and loc == 1:
            meta_label = 64
        elif age < 19 and dev == 4 and loc == 2:
            meta_label = 65
        elif age < 19 and dev == 4 and loc == 3:
            meta_label = 66
        elif age < 19 and dev == 4 and loc == 4:
            meta_label = 67
        elif age < 19 and dev == 4 and loc == 5:
            meta_label = 68
        elif age < 19 and dev == 4 and loc == 6:
            meta_label = 69
    
    elif args.meta_mode == 'age_dev_sex':
        if age >= 19 and dev == 0 and sex == 0:
            meta_label = 0
        elif age >= 19 and dev == 0 and sex == 1:
            meta_label = 1
        elif age >= 19 and dev == 1 and sex == 0:
            meta_label = 2
        elif age >= 19 and dev == 1 and sex == 1:
            meta_label = 3
        elif age >= 19 and dev == 2 and sex == 0:
            meta_label = 4
        elif age >= 19 and dev == 2 and sex == 1:
            meta_label = 5
        elif age >= 19 and dev == 3 and sex == 0:
            meta_label = 6
        elif age >= 19 and dev == 3 and sex == 1:
            meta_label = 7
        elif age >= 19 and dev == 4 and sex == 0:
            meta_label = 8
        elif age >= 19 and dev == 4 and sex == 1:
            meta_label = 9
        elif age < 19 and dev == 0 and sex == 0:
            meta_label = 10
        elif age < 19 and dev == 0 and sex == 1:
            meta_label = 11
        elif age < 19 and dev == 1 and sex == 0:
            meta_label = 12
        elif age < 19 and dev == 1 and sex == 1:
            meta_label = 13
        elif age < 19 and dev == 2 and sex == 0:
            meta_label = 14
        elif age < 19 and dev == 2 and sex == 1:
            meta_label = 15
        elif age < 19 and dev == 3 and sex == 0:
            meta_label = 16
        elif age < 19 and dev == 3 and sex == 1:
            meta_label = 17
        elif age < 19 and dev == 4 and sex == 0:
            meta_label = 18
        elif age < 19 and dev == 4 and sex == 1:
            meta_label = 19
    
    elif args.meta_mode == 'age_loc_sex':
        if age >= 19 and loc == 0 and sex == 0:
            meta_label = 0
        elif age >= 19 and loc == 0 and sex == 1:
            meta_label = 1
        elif age >= 19 and loc == 1 and sex == 0:
            meta_label = 2
        elif age >= 19 and loc == 1 and sex == 1:
            meta_label = 3
        elif age >= 19 and loc == 2 and sex == 0:
            meta_label = 4
        elif age >= 19 and loc == 2 and sex == 1:
            meta_label = 5
        elif age >= 19 and loc == 3 and sex == 0:
            meta_label = 6
        elif age >= 19 and loc == 3 and sex == 1:
            meta_label = 7
        elif age >= 19 and loc == 4 and sex == 0:
            meta_label = 8
        elif age >= 19 and loc == 4 and sex == 1:
            meta_label = 9
        elif age >= 19 and loc == 5 and sex == 0:
            meta_label = 10
        elif age >= 19 and loc == 5 and sex == 1:
            meta_label = 11
        elif age >= 19 and loc == 6 and sex == 0:
            meta_label = 12
        elif age >= 19 and loc == 6 and sex == 1:
            meta_label = 13
        elif age < 19 and loc == 0 and sex == 0:
            meta_label = 14
        elif age < 19 and loc == 0 and sex == 1:
            meta_label = 15
        elif age < 19 and loc == 1 and sex == 0:
            meta_label = 16
        elif age < 19 and loc == 1 and sex == 1:
            meta_label = 17
        elif age < 19 and loc == 2 and sex == 0:
            meta_label = 18
        elif age < 19 and loc == 2 and sex == 1:
            meta_label = 19
        elif age < 19 and loc == 3 and sex == 0:
            meta_label = 20
        elif age < 19 and loc == 3 and sex == 1:
            meta_label = 21
        elif age < 19 and loc == 4 and sex == 0:
            meta_label = 22
        elif age < 19 and loc == 4 and sex == 1:
            meta_label = 23
        elif age < 19 and loc == 5 and sex == 0:
            meta_label = 24
        elif age < 19 and loc == 5 and sex == 1:
            meta_label = 25
        elif age < 19 and loc == 6 and sex == 0:
            meta_label = 26
        elif age < 19 and loc == 6 and sex == 1:
            meta_label = 27
    
    elif args.meta_mode == 'data_dev_loc':
        if data == 0 and dev == 0 and loc == 0:
            meta_label = 0
        elif data == 0 and dev == 0 and loc == 1:
            meta_label = 1
        elif data == 0 and dev == 0 and loc == 2:
            meta_label = 2
        elif data == 0 and dev == 0 and loc == 3:
            meta_label = 3
        elif data == 0 and dev == 0 and loc == 4:
            meta_label = 4
        elif data == 0 and dev == 0 and loc == 5:
            meta_label = 5
        elif data == 0 and dev == 0 and loc == 6:
            meta_label = 6
        elif data == 0 and dev == 1 and loc == 0:
            meta_label = 7
        elif data == 0 and dev == 1 and loc == 1:
            meta_label = 8
        elif data == 0 and dev == 1 and loc == 2:
            meta_label = 9
        elif data == 0 and dev == 1 and loc == 3:
            meta_label = 10
        elif data == 0 and dev == 1 and loc == 4:
            meta_label = 11
        elif data == 0 and dev == 1 and loc == 5:
            meta_label = 12
        elif data == 0 and dev == 1 and loc == 6:
            meta_label = 13
        elif data == 0 and dev == 2 and loc == 0:
            meta_label = 14
        elif data == 0 and dev == 2 and loc == 1:
            meta_label = 15
        elif data == 0 and dev == 2 and loc == 2:
            meta_label = 16
        elif data == 0 and dev == 2 and loc == 3:
            meta_label = 17
        elif data == 0 and dev == 2 and loc == 4:
            meta_label = 18
        elif data == 0 and dev == 2 and loc == 5:
            meta_label = 19
        elif data == 0 and dev == 2 and loc == 6:
            meta_label = 20
        elif data == 0 and dev == 3 and loc == 0:
            meta_label = 21
        elif data == 0 and dev == 3 and loc == 1:
            meta_label = 22
        elif data == 0 and dev == 3 and loc == 2:
            meta_label = 23
        elif data == 0 and dev == 3 and loc == 3:
            meta_label = 24
        elif data == 0 and dev == 3 and loc == 4:
            meta_label = 25
        elif data == 0 and dev == 3 and loc == 5:
            meta_label = 26
        elif data == 0 and dev == 3 and loc == 6:
            meta_label = 27
        elif data == 0 and dev == 4 and loc == 0:
            meta_label = 28
        elif data == 0 and dev == 4 and loc == 1:
            meta_label = 28
        elif data == 0 and dev == 4 and loc == 2:
            meta_label = 30
        elif data == 0 and dev == 4 and loc == 3:
            meta_label = 31
        elif data == 0 and dev == 4 and loc == 4:
            meta_label = 32
        elif data == 0 and dev == 4 and loc == 5:
            meta_label = 33
        elif data == 0 and dev == 4 and loc == 6:
            meta_label = 34
        elif data == 1 and dev == 0 and loc == 0:
            meta_label = 35
        elif data == 1 and dev == 0 and loc == 1:
            meta_label = 36
        elif data == 1 and dev == 0 and loc == 2:
            meta_label = 37
        elif data == 1 and dev == 0 and loc == 3:
            meta_label = 38
        elif data == 1 and dev == 0 and loc == 4:
            meta_label = 39
        elif data == 1 and dev == 0 and loc == 5:
            meta_label = 40
        elif data == 1 and dev == 0 and loc == 6:
            meta_label = 41
        elif data == 1 and dev == 1 and loc == 0:
            meta_label = 42
        elif data == 1 and dev == 1 and loc == 1:
            meta_label = 43
        elif data == 1 and dev == 1 and loc == 2:
            meta_label = 44
        elif data == 1 and dev == 1 and loc == 3:
            meta_label = 45
        elif data == 1 and dev == 1 and loc == 4:
            meta_label = 46
        elif data == 1 and dev == 1 and loc == 5:
            meta_label = 47
        elif data == 1 and dev == 1 and loc == 6:
            meta_label = 48
        elif data == 1 and dev == 2 and loc == 0:
            meta_label = 49
        elif data == 1 and dev == 2 and loc == 1:
            meta_label = 50
        elif data == 1 and dev == 2 and loc == 2:
            meta_label = 51
        elif data == 1 and dev == 2 and loc == 3:
            meta_label = 52
        elif data == 1 and dev == 2 and loc == 4:
            meta_label = 53
        elif data == 1 and dev == 2 and loc == 5:
            meta_label = 54
        elif data == 1 and dev == 2 and loc == 6:
            meta_label = 55
        elif data == 1 and dev == 3 and loc == 0:
            meta_label = 56
        elif data == 1 and dev == 3 and loc == 1:
            meta_label = 57
        elif data == 1 and dev == 3 and loc == 2:
            meta_label = 58
        elif data == 1 and dev == 3 and loc == 3:
            meta_label = 59
        elif data == 1 and dev == 3 and loc == 4:
            meta_label = 60
        elif data == 1 and dev == 3 and loc == 5:
            meta_label = 61
        elif data == 1 and dev == 3 and loc == 6:
            meta_label = 62
        elif data == 1 and dev == 4 and loc == 0:
            meta_label = 63
        elif data == 1 and dev == 4 and loc == 1:
            meta_label = 64
        elif data == 1 and dev == 4 and loc == 2:
            meta_label = 65
        elif data == 1 and dev == 4 and loc == 3:
            meta_label = 66
        elif data == 1 and dev == 4 and loc == 4:
            meta_label = 67
        elif data == 1 and dev == 4 and loc == 5:
            meta_label = 68
        elif data == 1 and dev == 4 and loc == 6:
            meta_label = 69
    
    elif args.meta_mode == 'data_dev_sex':
        if data == 0 and dev == 0 and sex == 0:
            meta_label = 0
        elif data == 0 and dev == 0 and sex == 1:
            meta_label = 1
        elif data == 0 and dev == 1 and sex == 0:
            meta_label = 2
        elif data == 0 and dev == 1 and sex == 1:
            meta_label = 3
        elif data == 0 and dev == 2 and sex == 0:
            meta_label = 4
        elif data == 0 and dev == 2 and sex == 1:
            meta_label = 5
        elif data == 0 and dev == 3 and sex == 0:
            meta_label = 6
        elif data == 0 and dev == 3 and sex == 1:
            meta_label = 7
        elif data == 0 and dev == 4 and sex == 0:
            meta_label = 8
        elif data == 0 and dev == 4 and sex == 1:
            meta_label = 9
        elif data == 1 and dev == 0 and sex == 0:
            meta_label = 10
        elif data == 1 and dev == 0 and sex == 1:
            meta_label = 11
        elif data == 1 and dev == 1 and sex == 0:
            meta_label = 12
        elif data == 1 and dev == 1 and sex == 1:
            meta_label = 13
        elif data == 1 and dev == 2 and sex == 0:
            meta_label = 14
        elif data == 1 and dev == 2 and sex == 1:
            meta_label = 15
        elif data == 1 and dev == 3 and sex == 0:
            meta_label = 16
        elif data == 1 and dev == 3 and sex == 1:
            meta_label = 17
        elif data == 1 and dev == 4 and sex == 0:
            meta_label = 18
        elif data == 1 and dev == 4 and sex == 1:
            meta_label = 19
    
    elif args.meta_mode == 'data_loc_sex':
        if data == 0 and loc == 0 and sex == 0:
            meta_label = 0
        elif data == 0 and loc == 0 and sex == 1:
            meta_label = 1
        elif data == 0 and loc == 1 and sex == 0:
            meta_label = 2
        elif data == 0 and loc == 1 and sex == 1:
            meta_label = 3
        elif data == 0 and loc == 2 and sex == 0:
            meta_label = 4
        elif data == 0 and loc == 2 and sex == 1:
            meta_label = 5
        elif data == 0 and loc == 3 and sex == 0:
            meta_label = 6
        elif data == 0 and loc == 3 and sex == 1:
            meta_label = 7
        elif data == 0 and loc == 4 and sex == 0:
            meta_label = 8
        elif data == 0 and loc == 4 and sex == 1:
            meta_label = 9
        elif data == 0 and loc == 5 and sex == 0:
            meta_label = 10
        elif data == 0 and loc == 5 and sex == 1:
            meta_label = 11
        elif data == 0 and loc == 6 and sex == 0:
            meta_label = 12
        elif data == 0 and loc == 6 and sex == 1:
            meta_label = 13
        elif data == 1 and loc == 0 and sex == 0:
            meta_label = 14
        elif data == 1 and loc == 0 and sex == 1:
            meta_label = 15
        elif data == 1 and loc == 1 and sex == 0:
            meta_label = 16
        elif data == 1 and loc == 1 and sex == 1:
            meta_label = 17
        elif data == 1 and loc == 2 and sex == 0:
            meta_label = 18
        elif data == 1 and loc == 2 and sex == 1:
            meta_label = 19
        elif data == 1 and loc == 3 and sex == 0:
            meta_label = 20
        elif data == 1 and loc == 3 and sex == 1:
            meta_label = 21
        elif data == 1 and loc == 4 and sex == 0:
            meta_label = 22
        elif data == 1 and loc == 4 and sex == 1:
            meta_label = 23
        elif data == 1 and loc == 5 and sex == 0:
            meta_label = 24
        elif data == 1 and loc == 5 and sex == 1:
            meta_label = 25
        elif data == 1 and loc == 6 and sex == 0:
            meta_label = 26
        elif data == 1 and loc == 6 and sex == 1:
            meta_label = 27
    
    elif args.meta_mode == 'dev_loc_sex':
        if dev == 0 and loc == 0 and sex == 0:
            meta_label = 0
        elif dev == 0 and loc == 0 and sex == 1:
            meta_label = 1
        elif dev == 0 and loc == 1 and sex == 0:
            meta_label = 2
        elif dev == 0 and loc == 1 and sex == 1:
            meta_label = 3
        elif dev == 0 and loc == 2 and sex == 0:
            meta_label = 4
        elif dev == 0 and loc == 2 and sex == 1:
            meta_label = 5
        elif dev == 0 and loc == 3 and sex == 0:
            meta_label = 6
        elif dev == 0 and loc == 3 and sex == 1:
            meta_label = 7
        elif dev == 0 and loc == 4 and sex == 0:
            meta_label = 8
        elif dev == 0 and loc == 4 and sex == 1:
            meta_label = 9
        elif dev == 0 and loc == 5 and sex == 0:
            meta_label = 10
        elif dev == 0 and loc == 5 and sex == 1:
            meta_label = 11
        elif dev == 0 and loc == 6 and sex == 0:
            meta_label = 12
        elif dev == 0 and loc == 6 and sex == 1:
            meta_label = 13
        
        elif dev == 1 and loc == 0 and sex == 0:
            meta_label = 14
        elif dev == 1 and loc == 0 and sex == 1:
            meta_label = 15
        elif dev == 1 and loc == 1 and sex == 0:
            meta_label = 16
        elif dev == 1 and loc == 1 and sex == 1:
            meta_label = 17
        elif dev == 1 and loc == 2 and sex == 0:
            meta_label = 18
        elif dev == 1 and loc == 2 and sex == 1:
            meta_label = 19
        elif dev == 1 and loc == 3 and sex == 0:
            meta_label = 20
        elif dev == 1 and loc == 3 and sex == 1:
            meta_label = 21
        elif dev == 1 and loc == 4 and sex == 0:
            meta_label = 22
        elif dev == 1 and loc == 4 and sex == 1:
            meta_label = 23
        elif dev == 1 and loc == 5 and sex == 0:
            meta_label = 24
        elif dev == 1 and loc == 5 and sex == 1:
            meta_label = 25
        elif dev == 1 and loc == 6 and sex == 0:
            meta_label = 26
        elif dev == 1 and loc == 6 and sex == 1:
            meta_label = 27
        
        elif dev == 2 and loc == 0 and sex == 0:
            meta_label = 28
        elif dev == 2 and loc == 0 and sex == 1:
            meta_label = 29
        elif dev == 2 and loc == 1 and sex == 0:
            meta_label = 30
        elif dev == 2 and loc == 1 and sex == 1:
            meta_label = 31
        elif dev == 2 and loc == 2 and sex == 0:
            meta_label = 32
        elif dev == 2 and loc == 2 and sex == 1:
            meta_label = 33
        elif dev == 2 and loc == 3 and sex == 0:
            meta_label = 34
        elif dev == 2 and loc == 3 and sex == 1:
            meta_label = 35
        elif dev == 2 and loc == 4 and sex == 0:
            meta_label = 36
        elif dev == 2 and loc == 4 and sex == 1:
            meta_label = 37
        elif dev == 2 and loc == 5 and sex == 0:
            meta_label = 38
        elif dev == 2 and loc == 5 and sex == 1:
            meta_label = 39
        elif dev == 2 and loc == 6 and sex == 0:
            meta_label = 40
        elif dev == 2 and loc == 6 and sex == 1:
            meta_label = 41
        
        elif dev == 3 and loc == 0 and sex == 0:
            meta_label = 42
        elif dev == 3 and loc == 0 and sex == 1:
            meta_label = 43
        elif dev == 3 and loc == 1 and sex == 0:
            meta_label = 44
        elif dev == 3 and loc == 1 and sex == 1:
            meta_label = 45
        elif dev == 3 and loc == 2 and sex == 0:
            meta_label = 46
        elif dev == 3 and loc == 2 and sex == 1:
            meta_label = 47
        elif dev == 3 and loc == 3 and sex == 0:
            meta_label = 48
        elif dev == 3 and loc == 3 and sex == 1:
            meta_label = 49
        elif dev == 3 and loc == 4 and sex == 0:
            meta_label = 50
        elif dev == 3 and loc == 4 and sex == 1:
            meta_label = 51
        elif dev == 3 and loc == 5 and sex == 0:
            meta_label = 52
        elif dev == 3 and loc == 5 and sex == 1:
            meta_label = 53
        elif dev == 3 and loc == 6 and sex == 0:
            meta_label = 54
        elif dev == 3 and loc == 6 and sex == 1:
            meta_label = 55
        
        elif dev == 4 and loc == 0 and sex == 0:
            meta_label = 56
        elif dev == 4 and loc == 0 and sex == 1:
            meta_label = 57
        elif dev == 4 and loc == 1 and sex == 0:
            meta_label = 58
        elif dev == 4 and loc == 1 and sex == 1:
            meta_label = 59
        elif dev == 4 and loc == 2 and sex == 0:
            meta_label = 60
        elif dev == 4 and loc == 2 and sex == 1:
            meta_label = 61
        elif dev == 4 and loc == 3 and sex == 0:
            meta_label = 62
        elif dev == 4 and loc == 3 and sex == 1:
            meta_label = 63
        elif dev == 4 and loc == 4 and sex == 0:
            meta_label = 64
        elif dev == 4 and loc == 4 and sex == 1:
            meta_label = 65
        elif dev == 4 and loc == 5 and sex == 0:
            meta_label = 66
        elif dev == 4 and loc == 5 and sex == 1:
            meta_label = 67
        elif dev == 4 and loc == 6 and sex == 0:
            meta_label = 68
        elif dev == 4 and loc == 6 and sex == 1:
            meta_label = 69
    
    
    elif args.meta_mode == 'age_data_dev_loc':
        if age >= 19 and data == 0 and dev == 0 and loc == 0:
            meta_label = 0
        elif age >= 19 and data == 0 and dev == 0 and loc == 1:
            meta_label = 1
        elif age >= 19 and data == 0 and dev == 0 and loc == 2:
            meta_label = 2
        elif age >= 19 and data == 0 and dev == 0 and loc == 3:
            meta_label = 3
        elif age >= 19 and data == 0 and dev == 0 and loc == 4:
            meta_label = 4
        elif age >= 19 and data == 0 and dev == 0 and loc == 5:
            meta_label = 5
        elif age >= 19 and data == 0 and dev == 0 and loc == 6:
            meta_label = 6
        elif age >= 19 and data == 0 and dev == 1 and loc == 0:
            meta_label = 7
        elif age >= 19 and data == 0 and dev == 1 and loc == 1:
            meta_label = 8
        elif age >= 19 and data == 0 and dev == 1 and loc == 2:
            meta_label = 9
        elif age >= 19 and data == 0 and dev == 1 and loc == 3:
            meta_label = 10
        elif age >= 19 and data == 0 and dev == 1 and loc == 4:
            meta_label = 11
        elif age >= 19 and data == 0 and dev == 1 and loc == 5:
            meta_label = 12
        elif age >= 19 and data == 0 and dev == 1 and loc == 6:
            meta_label = 13
        elif age >= 19 and data == 0 and dev == 2 and loc == 0:
            meta_label = 14
        elif age >= 19 and data == 0 and dev == 2 and loc == 1:
            meta_label = 15
        elif age >= 19 and data == 0 and dev == 2 and loc == 2:
            meta_label = 16
        elif age >= 19 and data == 0 and dev == 2 and loc == 3:
            meta_label = 17
        elif age >= 19 and data == 0 and dev == 2 and loc == 4:
            meta_label = 18
        elif age >= 19 and data == 0 and dev == 2 and loc == 5:
            meta_label = 19
        elif age >= 19 and data == 0 and dev == 2 and loc == 6:
            meta_label = 20
        elif age >= 19 and data == 0 and dev == 3 and loc == 0:
            meta_label = 21
        elif age >= 19 and data == 0 and dev == 3 and loc == 1:
            meta_label = 22
        elif age >= 19 and data == 0 and dev == 3 and loc == 2:
            meta_label = 23
        elif age >= 19 and data == 0 and dev == 3 and loc == 3:
            meta_label = 24
        elif age >= 19 and data == 0 and dev == 3 and loc == 4:
            meta_label = 25
        elif age >= 19 and data == 0 and dev == 3 and loc == 5:
            meta_label = 26
        elif age >= 19 and data == 0 and dev == 3 and loc == 6:
            meta_label = 27
        elif age >= 19 and data == 0 and dev == 4 and loc == 0:
            meta_label = 28
        elif age >= 19 and data == 0 and dev == 4 and loc == 1:
            meta_label = 28
        elif age >= 19 and data == 0 and dev == 4 and loc == 2:
            meta_label = 30
        elif age >= 19 and data == 0 and dev == 4 and loc == 3:
            meta_label = 31
        elif age >= 19 and data == 0 and dev == 4 and loc == 4:
            meta_label = 32
        elif age >= 19 and data == 0 and dev == 4 and loc == 5:
            meta_label = 33
        elif age >= 19 and data == 0 and dev == 4 and loc == 6:
            meta_label = 34
        elif age >= 19 and data == 1 and dev == 0 and loc == 0:
            meta_label = 35
        elif age >= 19 and data == 1 and dev == 0 and loc == 1:
            meta_label = 36
        elif age >= 19 and data == 1 and dev == 0 and loc == 2:
            meta_label = 37
        elif age >= 19 and data == 1 and dev == 0 and loc == 3:
            meta_label = 38
        elif age >= 19 and data == 1 and dev == 0 and loc == 4:
            meta_label = 39
        elif age >= 19 and data == 1 and dev == 0 and loc == 5:
            meta_label = 40
        elif age >= 19 and data == 1 and dev == 0 and loc == 6:
            meta_label = 41
        elif age >= 19 and data == 1 and dev == 1 and loc == 0:
            meta_label = 42
        elif age >= 19 and data == 1 and dev == 1 and loc == 1:
            meta_label = 43
        elif age >= 19 and data == 1 and dev == 1 and loc == 2:
            meta_label = 44
        elif age >= 19 and data == 1 and dev == 1 and loc == 3:
            meta_label = 45
        elif age >= 19 and data == 1 and dev == 1 and loc == 4:
            meta_label = 46
        elif age >= 19 and data == 1 and dev == 1 and loc == 5:
            meta_label = 47
        elif age >= 19 and data == 1 and dev == 1 and loc == 6:
            meta_label = 48
        elif age >= 19 and data == 1 and dev == 2 and loc == 0:
            meta_label = 49
        elif age >= 19 and data == 1 and dev == 2 and loc == 1:
            meta_label = 50
        elif age >= 19 and data == 1 and dev == 2 and loc == 2:
            meta_label = 51
        elif age >= 19 and data == 1 and dev == 2 and loc == 3:
            meta_label = 52
        elif age >= 19 and data == 1 and dev == 2 and loc == 4:
            meta_label = 53
        elif age >= 19 and data == 1 and dev == 2 and loc == 5:
            meta_label = 54
        elif age >= 19 and data == 1 and dev == 2 and loc == 6:
            meta_label = 55
        elif age >= 19 and data == 1 and dev == 3 and loc == 0:
            meta_label = 56
        elif age >= 19 and data == 1 and dev == 3 and loc == 1:
            meta_label = 57
        elif age >= 19 and data == 1 and dev == 3 and loc == 2:
            meta_label = 58
        elif age >= 19 and data == 1 and dev == 3 and loc == 3:
            meta_label = 59
        elif age >= 19 and data == 1 and dev == 3 and loc == 4:
            meta_label = 60
        elif age >= 19 and data == 1 and dev == 3 and loc == 5:
            meta_label = 61
        elif age >= 19 and data == 1 and dev == 3 and loc == 6:
            meta_label = 62
        elif age >= 19 and data == 1 and dev == 4 and loc == 0:
            meta_label = 63
        elif age >= 19 and data == 1 and dev == 4 and loc == 1:
            meta_label = 64
        elif age >= 19 and data == 1 and dev == 4 and loc == 2:
            meta_label = 65
        elif age >= 19 and data == 1 and dev == 4 and loc == 3:
            meta_label = 66
        elif age >= 19 and data == 1 and dev == 4 and loc == 4:
            meta_label = 67
        elif age >= 19 and data == 1 and dev == 4 and loc == 5:
            meta_label = 68
        elif age >= 19 and data == 1 and dev == 4 and loc == 6:
            meta_label = 69
        
        elif age < 19 and data == 0 and dev == 0 and loc == 0:
            meta_label = 70
        elif age < 19 and data == 0 and dev == 0 and loc == 1:
            meta_label = 71
        elif age < 19 and data == 0 and dev == 0 and loc == 2:
            meta_label = 72
        elif age < 19 and data == 0 and dev == 0 and loc == 3:
            meta_label = 73
        elif age < 19 and data == 0 and dev == 0 and loc == 4:
            meta_label = 74
        elif age < 19 and data == 0 and dev == 0 and loc == 5:
            meta_label = 75
        elif age < 19 and data == 0 and dev == 0 and loc == 6:
            meta_label = 76
        elif age < 19 and data == 0 and dev == 1 and loc == 0:
            meta_label = 77
        elif age < 19 and data == 0 and dev == 1 and loc == 1:
            meta_label = 78
        elif age < 19 and data == 0 and dev == 1 and loc == 2:
            meta_label = 79
        elif age < 19 and data == 0 and dev == 1 and loc == 3:
            meta_label = 80
        elif age < 19 and data == 0 and dev == 1 and loc == 4:
            meta_label = 81
        elif age < 19 and data == 0 and dev == 1 and loc == 5:
            meta_label = 82
        elif age < 19 and data == 0 and dev == 1 and loc == 6:
            meta_label = 83
        elif age < 19 and data == 0 and dev == 2 and loc == 0:
            meta_label = 84
        elif age < 19 and data == 0 and dev == 2 and loc == 1:
            meta_label = 85
        elif age < 19 and data == 0 and dev == 2 and loc == 2:
            meta_label = 86
        elif age < 19 and data == 0 and dev == 2 and loc == 3:
            meta_label = 87
        elif age < 19 and data == 0 and dev == 2 and loc == 4:
            meta_label = 88
        elif age < 19 and data == 0 and dev == 2 and loc == 5:
            meta_label = 89
        elif age < 19 and data == 0 and dev == 2 and loc == 6:
            meta_label = 90
        elif age < 19 and data == 0 and dev == 3 and loc == 0:
            meta_label = 91
        elif age < 19 and data == 0 and dev == 3 and loc == 1:
            meta_label = 92
        elif age < 19 and data == 0 and dev == 3 and loc == 2:
            meta_label = 93
        elif age < 19 and data == 0 and dev == 3 and loc == 3:
            meta_label = 94
        elif age < 19 and data == 0 and dev == 3 and loc == 4:
            meta_label = 95
        elif age < 19 and data == 0 and dev == 3 and loc == 5:
            meta_label = 96
        elif age < 19 and data == 0 and dev == 3 and loc == 6:
            meta_label = 97
        elif age < 19 and data == 0 and dev == 4 and loc == 0:
            meta_label = 98
        elif age < 19 and data == 0 and dev == 4 and loc == 1:
            meta_label = 99
        elif age < 19 and data == 0 and dev == 4 and loc == 2:
            meta_label = 100
        elif age < 19 and data == 0 and dev == 4 and loc == 3:
            meta_label = 101
        elif age < 19 and data == 0 and dev == 4 and loc == 4:
            meta_label = 102
        elif age < 19 and data == 0 and dev == 4 and loc == 5:
            meta_label = 103
        elif age < 19 and data == 0 and dev == 4 and loc == 6:
            meta_label = 104
        elif age < 19 and data == 1 and dev == 0 and loc == 0:
            meta_label = 105
        elif age < 19 and data == 1 and dev == 0 and loc == 1:
            meta_label = 106
        elif age < 19 and data == 1 and dev == 0 and loc == 2:
            meta_label = 107
        elif age < 19 and data == 1 and dev == 0 and loc == 3:
            meta_label = 108
        elif age < 19 and data == 1 and dev == 0 and loc == 4:
            meta_label = 109
        elif age < 19 and data == 1 and dev == 0 and loc == 5:
            meta_label = 110
        elif age < 19 and data == 1 and dev == 0 and loc == 6:
            meta_label = 111
        elif age < 19 and data == 1 and dev == 1 and loc == 0:
            meta_label = 112
        elif age < 19 and data == 1 and dev == 1 and loc == 1:
            meta_label = 113
        elif age < 19 and data == 1 and dev == 1 and loc == 2:
            meta_label = 114
        elif age < 19 and data == 1 and dev == 1 and loc == 3:
            meta_label = 115
        elif age < 19 and data == 1 and dev == 1 and loc == 4:
            meta_label = 116
        elif age < 19 and data == 1 and dev == 1 and loc == 5:
            meta_label = 117
        elif age < 19 and data == 1 and dev == 1 and loc == 6:
            meta_label = 118
        elif age < 19 and data == 1 and dev == 2 and loc == 0:
            meta_label = 119
        elif age < 19 and data == 1 and dev == 2 and loc == 1:
            meta_label = 120
        elif age < 19 and data == 1 and dev == 2 and loc == 2:
            meta_label = 121
        elif age < 19 and data == 1 and dev == 2 and loc == 3:
            meta_label = 122
        elif age < 19 and data == 1 and dev == 2 and loc == 4:
            meta_label = 123
        elif age < 19 and data == 1 and dev == 2 and loc == 5:
            meta_label = 124
        elif age < 19 and data == 1 and dev == 2 and loc == 6:
            meta_label = 125
        elif age < 19 and data == 1 and dev == 3 and loc == 0:
            meta_label = 126
        elif age < 19 and data == 1 and dev == 3 and loc == 1:
            meta_label = 127
        elif age < 19 and data == 1 and dev == 3 and loc == 2:
            meta_label = 128
        elif age < 19 and data == 1 and dev == 3 and loc == 3:
            meta_label = 129
        elif age < 19 and data == 1 and dev == 3 and loc == 4:
            meta_label = 130
        elif age < 19 and data == 1 and dev == 3 and loc == 5:
            meta_label = 131
        elif age < 19 and data == 1 and dev == 3 and loc == 6:
            meta_label = 132
        elif age < 19 and data == 1 and dev == 4 and loc == 0:
            meta_label = 133
        elif age < 19 and data == 1 and dev == 4 and loc == 1:
            meta_label = 134
        elif age < 19 and data == 1 and dev == 4 and loc == 2:
            meta_label = 135
        elif age < 19 and data == 1 and dev == 4 and loc == 3:
            meta_label = 136
        elif age < 19 and data == 1 and dev == 4 and loc == 4:
            meta_label = 137
        elif age < 19 and data == 1 and dev == 4 and loc == 5:
            meta_label = 138
        elif age < 19 and data == 1 and dev == 4 and loc == 6:
            meta_label = 139
    
    
    elif args.meta_mode == 'age_data_dev_sex':
        if age >= 19 and data == 0 and dev == 0 and sex == 0:
            meta_label = 0
        elif age >= 19 and data == 0 and dev == 0 and sex == 1:
            meta_label = 1
        elif age >= 19 and data == 0 and dev == 1 and sex == 0:
            meta_label = 2
        elif age >= 19 and data == 0 and dev == 1 and sex == 1:
            meta_label = 3
        elif age >= 19 and data == 0 and dev == 2 and sex == 0:
            meta_label = 4
        elif age >= 19 and data == 0 and dev == 2 and sex == 1:
            meta_label = 5
        elif age >= 19 and data == 0 and dev == 3 and sex == 0:
            meta_label = 6
        elif age >= 19 and data == 0 and dev == 3 and sex == 1:
            meta_label = 7
        elif age >= 19 and data == 0 and dev == 4 and sex == 0:
            meta_label = 8
        elif age >= 19 and data == 0 and dev == 4 and sex == 1:
            meta_label = 9
        elif age >= 19 and data == 1 and dev == 0 and sex == 0:
            meta_label = 10
        elif age >= 19 and data == 1 and dev == 0 and sex == 1:
            meta_label = 11
        elif age >= 19 and data == 1 and dev == 1 and sex == 0:
            meta_label = 12
        elif age >= 19 and data == 1 and dev == 1 and sex == 1:
            meta_label = 13
        elif age >= 19 and data == 1 and dev == 2 and sex == 0:
            meta_label = 14
        elif age >= 19 and data == 1 and dev == 2 and sex == 1:
            meta_label = 15
        elif age >= 19 and data == 1 and dev == 3 and sex == 0:
            meta_label = 16
        elif age >= 19 and data == 1 and dev == 3 and sex == 1:
            meta_label = 17
        elif age >= 19 and data == 1 and dev == 4 and sex == 0:
            meta_label = 18
        elif age >= 19 and data == 1 and dev == 4 and sex == 1:
            meta_label = 19
        elif age < 19 and data == 0 and dev == 0 and sex == 0:
            meta_label = 20
        elif age < 19 and data == 0 and dev == 0 and sex == 1:
            meta_label = 21
        elif age < 19 and data == 0 and dev == 1 and sex == 0:
            meta_label = 22
        elif age < 19 and data == 0 and dev == 1 and sex == 1:
            meta_label = 23
        elif age < 19 and data == 0 and dev == 2 and sex == 0:
            meta_label = 24
        elif age < 19 and data == 0 and dev == 2 and sex == 1:
            meta_label = 25
        elif age < 19 and data == 0 and dev == 3 and sex == 0:
            meta_label = 26
        elif age < 19 and data == 0 and dev == 3 and sex == 1:
            meta_label = 27
        elif age < 19 and data == 0 and dev == 4 and sex == 0:
            meta_label = 28
        elif age < 19 and data == 0 and dev == 4 and sex == 1:
            meta_label = 29
        elif age < 19 and data == 1 and dev == 0 and sex == 0:
            meta_label = 30
        elif age < 19 and data == 1 and dev == 0 and sex == 1:
            meta_label = 31
        elif age < 19 and data == 1 and dev == 1 and sex == 0:
            meta_label = 32
        elif age < 19 and data == 1 and dev == 1 and sex == 1:
            meta_label = 33
        elif age < 19 and data == 1 and dev == 2 and sex == 0:
            meta_label = 34
        elif age < 19 and data == 1 and dev == 2 and sex == 1:
            meta_label = 35
        elif age < 19 and data == 1 and dev == 3 and sex == 0:
            meta_label = 36
        elif age < 19 and data == 1 and dev == 3 and sex == 1:
            meta_label = 37
        elif age < 19 and data == 1 and dev == 4 and sex == 0:
            meta_label = 38
        elif age < 19 and data == 1 and dev == 4 and sex == 1:
            meta_label = 39
    
    
    elif args.meta_mode == 'age_dev_loc_sex':
        if age >= 19 and sex == 0 and dev == 0 and loc == 0:
            meta_label = 0
        elif age >= 19 and sex == 0 and dev == 0 and loc == 1:
            meta_label = 1
        elif age >= 19 and sex == 0 and dev == 0 and loc == 2:
            meta_label = 2
        elif age >= 19 and sex == 0 and dev == 0 and loc == 3:
            meta_label = 3
        elif age >= 19 and sex == 0 and dev == 0 and loc == 4:
            meta_label = 4
        elif age >= 19 and sex == 0 and dev == 0 and loc == 5:
            meta_label = 5
        elif age >= 19 and sex == 0 and dev == 0 and loc == 6:
            meta_label = 6
        elif age >= 19 and sex == 0 and dev == 1 and loc == 0:
            meta_label = 7
        elif age >= 19 and sex == 0 and dev == 1 and loc == 1:
            meta_label = 8
        elif age >= 19 and sex == 0 and dev == 1 and loc == 2:
            meta_label = 9
        elif age >= 19 and sex == 0 and dev == 1 and loc == 3:
            meta_label = 10
        elif age >= 19 and sex == 0 and dev == 1 and loc == 4:
            meta_label = 11
        elif age >= 19 and sex == 0 and dev == 1 and loc == 5:
            meta_label = 12
        elif age >= 19 and sex == 0 and dev == 1 and loc == 6:
            meta_label = 13
        elif age >= 19 and sex == 0 and dev == 2 and loc == 0:
            meta_label = 14
        elif age >= 19 and sex == 0 and dev == 2 and loc == 1:
            meta_label = 15
        elif age >= 19 and sex == 0 and dev == 2 and loc == 2:
            meta_label = 16
        elif age >= 19 and sex == 0 and dev == 2 and loc == 3:
            meta_label = 17
        elif age >= 19 and sex == 0 and dev == 2 and loc == 4:
            meta_label = 18
        elif age >= 19 and sex == 0 and dev == 2 and loc == 5:
            meta_label = 19
        elif age >= 19 and sex == 0 and dev == 2 and loc == 6:
            meta_label = 20
        elif age >= 19 and sex == 0 and dev == 3 and loc == 0:
            meta_label = 21
        elif age >= 19 and sex == 0 and dev == 3 and loc == 1:
            meta_label = 22
        elif age >= 19 and sex == 0 and dev == 3 and loc == 2:
            meta_label = 23
        elif age >= 19 and sex == 0 and dev == 3 and loc == 3:
            meta_label = 24
        elif age >= 19 and sex == 0 and dev == 3 and loc == 4:
            meta_label = 25
        elif age >= 19 and sex == 0 and dev == 3 and loc == 5:
            meta_label = 26
        elif age >= 19 and sex == 0 and dev == 3 and loc == 6:
            meta_label = 27
        elif age >= 19 and sex == 0 and dev == 4 and loc == 0:
            meta_label = 28
        elif age >= 19 and sex == 0 and dev == 4 and loc == 1:
            meta_label = 28
        elif age >= 19 and sex == 0 and dev == 4 and loc == 2:
            meta_label = 30
        elif age >= 19 and sex == 0 and dev == 4 and loc == 3:
            meta_label = 31
        elif age >= 19 and sex == 0 and dev == 4 and loc == 4:
            meta_label = 32
        elif age >= 19 and sex == 0 and dev == 4 and loc == 5:
            meta_label = 33
        elif age >= 19 and sex == 0 and dev == 4 and loc == 6:
            meta_label = 34
        elif age >= 19 and sex == 1 and dev == 0 and loc == 0:
            meta_label = 35
        elif age >= 19 and sex == 1 and dev == 0 and loc == 1:
            meta_label = 36
        elif age >= 19 and sex == 1 and dev == 0 and loc == 2:
            meta_label = 37
        elif age >= 19 and sex == 1 and dev == 0 and loc == 3:
            meta_label = 38
        elif age >= 19 and sex == 1 and dev == 0 and loc == 4:
            meta_label = 39
        elif age >= 19 and sex == 1 and dev == 0 and loc == 5:
            meta_label = 40
        elif age >= 19 and sex == 1 and dev == 0 and loc == 6:
            meta_label = 41
        elif age >= 19 and sex == 1 and dev == 1 and loc == 0:
            meta_label = 42
        elif age >= 19 and sex == 1 and dev == 1 and loc == 1:
            meta_label = 43
        elif age >= 19 and sex == 1 and dev == 1 and loc == 2:
            meta_label = 44
        elif age >= 19 and sex == 1 and dev == 1 and loc == 3:
            meta_label = 45
        elif age >= 19 and sex == 1 and dev == 1 and loc == 4:
            meta_label = 46
        elif age >= 19 and sex == 1 and dev == 1 and loc == 5:
            meta_label = 47
        elif age >= 19 and sex == 1 and dev == 1 and loc == 6:
            meta_label = 48
        elif age >= 19 and sex == 1 and dev == 2 and loc == 0:
            meta_label = 49
        elif age >= 19 and sex == 1 and dev == 2 and loc == 1:
            meta_label = 50
        elif age >= 19 and sex == 1 and dev == 2 and loc == 2:
            meta_label = 51
        elif age >= 19 and sex == 1 and dev == 2 and loc == 3:
            meta_label = 52
        elif age >= 19 and sex == 1 and dev == 2 and loc == 4:
            meta_label = 53
        elif age >= 19 and sex == 1 and dev == 2 and loc == 5:
            meta_label = 54
        elif age >= 19 and sex == 1 and dev == 2 and loc == 6:
            meta_label = 55
        elif age >= 19 and sex == 1 and dev == 3 and loc == 0:
            meta_label = 56
        elif age >= 19 and sex == 1 and dev == 3 and loc == 1:
            meta_label = 57
        elif age >= 19 and sex == 1 and dev == 3 and loc == 2:
            meta_label = 58
        elif age >= 19 and sex == 1 and dev == 3 and loc == 3:
            meta_label = 59
        elif age >= 19 and sex == 1 and dev == 3 and loc == 4:
            meta_label = 60
        elif age >= 19 and sex == 1 and dev == 3 and loc == 5:
            meta_label = 61
        elif age >= 19 and sex == 1 and dev == 3 and loc == 6:
            meta_label = 62
        elif age >= 19 and sex == 1 and dev == 4 and loc == 0:
            meta_label = 63
        elif age >= 19 and sex == 1 and dev == 4 and loc == 1:
            meta_label = 64
        elif age >= 19 and sex == 1 and dev == 4 and loc == 2:
            meta_label = 65
        elif age >= 19 and sex == 1 and dev == 4 and loc == 3:
            meta_label = 66
        elif age >= 19 and sex == 1 and dev == 4 and loc == 4:
            meta_label = 67
        elif age >= 19 and sex == 1 and dev == 4 and loc == 5:
            meta_label = 68
        elif age >= 19 and sex == 1 and dev == 4 and loc == 6:
            meta_label = 69
        
        elif age < 19 and sex == 0 and dev == 0 and loc == 0:
            meta_label = 70
        elif age < 19 and sex == 0 and dev == 0 and loc == 1:
            meta_label = 71
        elif age < 19 and sex == 0 and dev == 0 and loc == 2:
            meta_label = 72
        elif age < 19 and sex == 0 and dev == 0 and loc == 3:
            meta_label = 73
        elif age < 19 and sex == 0 and dev == 0 and loc == 4:
            meta_label = 74
        elif age < 19 and sex == 0 and dev == 0 and loc == 5:
            meta_label = 75
        elif age < 19 and sex == 0 and dev == 0 and loc == 6:
            meta_label = 76
        elif age < 19 and sex == 0 and dev == 1 and loc == 0:
            meta_label = 77
        elif age < 19 and sex == 0 and dev == 1 and loc == 1:
            meta_label = 78
        elif age < 19 and sex == 0 and dev == 1 and loc == 2:
            meta_label = 79
        elif age < 19 and sex == 0 and dev == 1 and loc == 3:
            meta_label = 80
        elif age < 19 and sex == 0 and dev == 1 and loc == 4:
            meta_label = 81
        elif age < 19 and sex == 0 and dev == 1 and loc == 5:
            meta_label = 82
        elif age < 19 and sex == 0 and dev == 1 and loc == 6:
            meta_label = 83
        elif age < 19 and sex == 0 and dev == 2 and loc == 0:
            meta_label = 84
        elif age < 19 and sex == 0 and dev == 2 and loc == 1:
            meta_label = 85
        elif age < 19 and sex == 0 and dev == 2 and loc == 2:
            meta_label = 86
        elif age < 19 and sex == 0 and dev == 2 and loc == 3:
            meta_label = 87
        elif age < 19 and sex == 0 and dev == 2 and loc == 4:
            meta_label = 88
        elif age < 19 and sex == 0 and dev == 2 and loc == 5:
            meta_label = 89
        elif age < 19 and sex == 0 and dev == 2 and loc == 6:
            meta_label = 90
        elif age < 19 and sex == 0 and dev == 3 and loc == 0:
            meta_label = 91
        elif age < 19 and sex == 0 and dev == 3 and loc == 1:
            meta_label = 92
        elif age < 19 and sex == 0 and dev == 3 and loc == 2:
            meta_label = 93
        elif age < 19 and sex == 0 and dev == 3 and loc == 3:
            meta_label = 94
        elif age < 19 and sex == 0 and dev == 3 and loc == 4:
            meta_label = 95
        elif age < 19 and sex == 0 and dev == 3 and loc == 5:
            meta_label = 96
        elif age < 19 and sex == 0 and dev == 3 and loc == 6:
            meta_label = 97
        elif age < 19 and sex == 0 and dev == 4 and loc == 0:
            meta_label = 98
        elif age < 19 and sex == 0 and dev == 4 and loc == 1:
            meta_label = 99
        elif age < 19 and sex == 0 and dev == 4 and loc == 2:
            meta_label = 100
        elif age < 19 and sex == 0 and dev == 4 and loc == 3:
            meta_label = 101
        elif age < 19 and sex == 0 and dev == 4 and loc == 4:
            meta_label = 102
        elif age < 19 and sex == 0 and dev == 4 and loc == 5:
            meta_label = 103
        elif age < 19 and sex == 0 and dev == 4 and loc == 6:
            meta_label = 104
        elif age < 19 and sex == 1 and dev == 0 and loc == 0:
            meta_label = 105
        elif age < 19 and sex == 1 and dev == 0 and loc == 1:
            meta_label = 106
        elif age < 19 and sex == 1 and dev == 0 and loc == 2:
            meta_label = 107
        elif age < 19 and sex == 1 and dev == 0 and loc == 3:
            meta_label = 108
        elif age < 19 and sex == 1 and dev == 0 and loc == 4:
            meta_label = 109
        elif age < 19 and sex == 1 and dev == 0 and loc == 5:
            meta_label = 110
        elif age < 19 and sex == 1 and dev == 0 and loc == 6:
            meta_label = 111
        elif age < 19 and sex == 1 and dev == 1 and loc == 0:
            meta_label = 112
        elif age < 19 and sex == 1 and dev == 1 and loc == 1:
            meta_label = 113
        elif age < 19 and sex == 1 and dev == 1 and loc == 2:
            meta_label = 114
        elif age < 19 and sex == 1 and dev == 1 and loc == 3:
            meta_label = 115
        elif age < 19 and sex == 1 and dev == 1 and loc == 4:
            meta_label = 116
        elif age < 19 and sex == 1 and dev == 1 and loc == 5:
            meta_label = 117
        elif age < 19 and sex == 1 and dev == 1 and loc == 6:
            meta_label = 118
        elif age < 19 and sex == 1 and dev == 2 and loc == 0:
            meta_label = 119
        elif age < 19 and sex == 1 and dev == 2 and loc == 1:
            meta_label = 120
        elif age < 19 and sex == 1 and dev == 2 and loc == 2:
            meta_label = 121
        elif age < 19 and sex == 1 and dev == 2 and loc == 3:
            meta_label = 122
        elif age < 19 and sex == 1 and dev == 2 and loc == 4:
            meta_label = 123
        elif age < 19 and sex == 1 and dev == 2 and loc == 5:
            meta_label = 124
        elif age < 19 and sex == 1 and dev == 2 and loc == 6:
            meta_label = 125
        elif age < 19 and sex == 1 and dev == 3 and loc == 0:
            meta_label = 126
        elif age < 19 and sex == 1 and dev == 3 and loc == 1:
            meta_label = 127
        elif age < 19 and sex == 1 and dev == 3 and loc == 2:
            meta_label = 128
        elif age < 19 and sex == 1 and dev == 3 and loc == 3:
            meta_label = 129
        elif age < 19 and sex == 1 and dev == 3 and loc == 4:
            meta_label = 130
        elif age < 19 and sex == 1 and dev == 3 and loc == 5:
            meta_label = 131
        elif age < 19 and sex == 1 and dev == 3 and loc == 6:
            meta_label = 132
        elif age < 19 and sex == 1 and dev == 4 and loc == 0:
            meta_label = 133
        elif age < 19 and sex == 1 and dev == 4 and loc == 1:
            meta_label = 134
        elif age < 19 and sex == 1 and dev == 4 and loc == 2:
            meta_label = 135
        elif age < 19 and sex == 1 and dev == 4 and loc == 3:
            meta_label = 136
        elif age < 19 and sex == 1 and dev == 4 and loc == 4:
            meta_label = 137
        elif age < 19 and sex == 1 and dev == 4 and loc == 5:
            meta_label = 138
        elif age < 19 and sex == 1 and dev == 4 and loc == 6:
            meta_label = 139
    
    elif args.meta_mode == 'data_dev_loc_sex':
        if data == 0 and sex == 0 and dev == 0 and loc == 0:
            meta_label = 0
        elif data == 0 and sex == 0 and dev == 0 and loc == 1:
            meta_label = 1
        elif data == 0 and sex == 0 and dev == 0 and loc == 2:
            meta_label = 2
        elif data == 0 and sex == 0 and dev == 0 and loc == 3:
            meta_label = 3
        elif data == 0 and sex == 0 and dev == 0 and loc == 4:
            meta_label = 4
        elif data == 0 and sex == 0 and dev == 0 and loc == 5:
            meta_label = 5
        elif data == 0 and sex == 0 and dev == 0 and loc == 6:
            meta_label = 6
        elif data == 0 and sex == 0 and dev == 1 and loc == 0:
            meta_label = 7
        elif data == 0 and sex == 0 and dev == 1 and loc == 1:
            meta_label = 8
        elif data == 0 and sex == 0 and dev == 1 and loc == 2:
            meta_label = 9
        elif data == 0 and sex == 0 and dev == 1 and loc == 3:
            meta_label = 10
        elif data == 0 and sex == 0 and dev == 1 and loc == 4:
            meta_label = 11
        elif data == 0 and sex == 0 and dev == 1 and loc == 5:
            meta_label = 12
        elif data == 0 and sex == 0 and dev == 1 and loc == 6:
            meta_label = 13
        elif data == 0 and sex == 0 and dev == 2 and loc == 0:
            meta_label = 14
        elif data == 0 and sex == 0 and dev == 2 and loc == 1:
            meta_label = 15
        elif data == 0 and sex == 0 and dev == 2 and loc == 2:
            meta_label = 16
        elif data == 0 and sex == 0 and dev == 2 and loc == 3:
            meta_label = 17
        elif data == 0 and sex == 0 and dev == 2 and loc == 4:
            meta_label = 18
        elif data == 0 and sex == 0 and dev == 2 and loc == 5:
            meta_label = 19
        elif data == 0 and sex == 0 and dev == 2 and loc == 6:
            meta_label = 20
        elif data == 0 and sex == 0 and dev == 3 and loc == 0:
            meta_label = 21
        elif data == 0 and sex == 0 and dev == 3 and loc == 1:
            meta_label = 22
        elif data == 0 and sex == 0 and dev == 3 and loc == 2:
            meta_label = 23
        elif data == 0 and sex == 0 and dev == 3 and loc == 3:
            meta_label = 24
        elif data == 0 and sex == 0 and dev == 3 and loc == 4:
            meta_label = 25
        elif data == 0 and sex == 0 and dev == 3 and loc == 5:
            meta_label = 26
        elif data == 0 and sex == 0 and dev == 3 and loc == 6:
            meta_label = 27
        elif data == 0 and sex == 0 and dev == 4 and loc == 0:
            meta_label = 28
        elif data == 0 and sex == 0 and dev == 4 and loc == 1:
            meta_label = 28
        elif data == 0 and sex == 0 and dev == 4 and loc == 2:
            meta_label = 30
        elif data == 0 and sex == 0 and dev == 4 and loc == 3:
            meta_label = 31
        elif data == 0 and sex == 0 and dev == 4 and loc == 4:
            meta_label = 32
        elif data == 0 and sex == 0 and dev == 4 and loc == 5:
            meta_label = 33
        elif data == 0 and sex == 0 and dev == 4 and loc == 6:
            meta_label = 34
        elif data == 0 and sex == 1 and dev == 0 and loc == 0:
            meta_label = 35
        elif data == 0 and sex == 1 and dev == 0 and loc == 1:
            meta_label = 36
        elif data == 0 and sex == 1 and dev == 0 and loc == 2:
            meta_label = 37
        elif data == 0 and sex == 1 and dev == 0 and loc == 3:
            meta_label = 38
        elif data == 0 and sex == 1 and dev == 0 and loc == 4:
            meta_label = 39
        elif data == 0 and sex == 1 and dev == 0 and loc == 5:
            meta_label = 40
        elif data == 0 and sex == 1 and dev == 0 and loc == 6:
            meta_label = 41
        elif data == 0 and sex == 1 and dev == 1 and loc == 0:
            meta_label = 42
        elif data == 0 and sex == 1 and dev == 1 and loc == 1:
            meta_label = 43
        elif data == 0 and sex == 1 and dev == 1 and loc == 2:
            meta_label = 44
        elif data == 0 and sex == 1 and dev == 1 and loc == 3:
            meta_label = 45
        elif data == 0 and sex == 1 and dev == 1 and loc == 4:
            meta_label = 46
        elif data == 0 and sex == 1 and dev == 1 and loc == 5:
            meta_label = 47
        elif data == 0 and sex == 1 and dev == 1 and loc == 6:
            meta_label = 48
        elif data == 0 and sex == 1 and dev == 2 and loc == 0:
            meta_label = 49
        elif data == 0 and sex == 1 and dev == 2 and loc == 1:
            meta_label = 50
        elif data == 0 and sex == 1 and dev == 2 and loc == 2:
            meta_label = 51
        elif data == 0 and sex == 1 and dev == 2 and loc == 3:
            meta_label = 52
        elif data == 0 and sex == 1 and dev == 2 and loc == 4:
            meta_label = 53
        elif data == 0 and sex == 1 and dev == 2 and loc == 5:
            meta_label = 54
        elif data == 0 and sex == 1 and dev == 2 and loc == 6:
            meta_label = 55
        elif data == 0 and sex == 1 and dev == 3 and loc == 0:
            meta_label = 56
        elif data == 0 and sex == 1 and dev == 3 and loc == 1:
            meta_label = 57
        elif data == 0 and sex == 1 and dev == 3 and loc == 2:
            meta_label = 58
        elif data == 0 and sex == 1 and dev == 3 and loc == 3:
            meta_label = 59
        elif data == 0 and sex == 1 and dev == 3 and loc == 4:
            meta_label = 60
        elif data == 0 and sex == 1 and dev == 3 and loc == 5:
            meta_label = 61
        elif data == 0 and sex == 1 and dev == 3 and loc == 6:
            meta_label = 62
        elif data == 0 and sex == 1 and dev == 4 and loc == 0:
            meta_label = 63
        elif data == 0 and sex == 1 and dev == 4 and loc == 1:
            meta_label = 64
        elif data == 0 and sex == 1 and dev == 4 and loc == 2:
            meta_label = 65
        elif data == 0 and sex == 1 and dev == 4 and loc == 3:
            meta_label = 66
        elif data == 0 and sex == 1 and dev == 4 and loc == 4:
            meta_label = 67
        elif data == 0 and sex == 1 and dev == 4 and loc == 5:
            meta_label = 68
        elif data == 0 and sex == 1 and dev == 4 and loc == 6:
            meta_label = 69
        
        elif data == 1 and sex == 0 and dev == 0 and loc == 0:
            meta_label = 70
        elif data == 1 and sex == 0 and dev == 0 and loc == 1:
            meta_label = 71
        elif data == 1 and sex == 0 and dev == 0 and loc == 2:
            meta_label = 72
        elif data == 1 and sex == 0 and dev == 0 and loc == 3:
            meta_label = 73
        elif data == 1 and sex == 0 and dev == 0 and loc == 4:
            meta_label = 74
        elif data == 1 and sex == 0 and dev == 0 and loc == 5:
            meta_label = 75
        elif data == 1 and sex == 0 and dev == 0 and loc == 6:
            meta_label = 76
        elif data == 1 and sex == 0 and dev == 1 and loc == 0:
            meta_label = 77
        elif data == 1 and sex == 0 and dev == 1 and loc == 1:
            meta_label = 78
        elif data == 1 and sex == 0 and dev == 1 and loc == 2:
            meta_label = 79
        elif data == 1 and sex == 0 and dev == 1 and loc == 3:
            meta_label = 80
        elif data == 1 and sex == 0 and dev == 1 and loc == 4:
            meta_label = 81
        elif data == 1 and sex == 0 and dev == 1 and loc == 5:
            meta_label = 82
        elif data == 19 and sex == 0 and dev == 1 and loc == 6:
            meta_label = 83
        elif data == 1 and sex == 0 and dev == 2 and loc == 0:
            meta_label = 84
        elif data == 1 and sex == 0 and dev == 2 and loc == 1:
            meta_label = 85
        elif data == 1 and sex == 0 and dev == 2 and loc == 2:
            meta_label = 86
        elif data == 1 and sex == 0 and dev == 2 and loc == 3:
            meta_label = 87
        elif data == 1 and sex == 0 and dev == 2 and loc == 4:
            meta_label = 88
        elif data == 1 and sex == 0 and dev == 2 and loc == 5:
            meta_label = 89
        elif data == 1 and sex == 0 and dev == 2 and loc == 6:
            meta_label = 90
        elif data == 1 and sex == 0 and dev == 3 and loc == 0:
            meta_label = 91
        elif data == 1 and sex == 0 and dev == 3 and loc == 1:
            meta_label = 92
        elif data == 1 and sex == 0 and dev == 3 and loc == 2:
            meta_label = 93
        elif data == 1 and sex == 0 and dev == 3 and loc == 3:
            meta_label = 94
        elif data == 1 and sex == 0 and dev == 3 and loc == 4:
            meta_label = 95
        elif data == 1 and sex == 0 and dev == 3 and loc == 5:
            meta_label = 96
        elif data == 1 and sex == 0 and dev == 3 and loc == 6:
            meta_label = 97
        elif data == 1 and sex == 0 and dev == 4 and loc == 0:
            meta_label = 98
        elif data == 1 and sex == 0 and dev == 4 and loc == 1:
            meta_label = 99
        elif data == 1 and sex == 0 and dev == 4 and loc == 2:
            meta_label = 100
        elif data == 1 and sex == 0 and dev == 4 and loc == 3:
            meta_label = 101
        elif data == 1 and sex == 0 and dev == 4 and loc == 4:
            meta_label = 102
        elif data == 1 and sex == 0 and dev == 4 and loc == 5:
            meta_label = 103
        elif data == 1 and sex == 0 and dev == 4 and loc == 6:
            meta_label = 104
        elif data == 1 and sex == 1 and dev == 0 and loc == 0:
            meta_label = 105
        elif data == 1 and sex == 1 and dev == 0 and loc == 1:
            meta_label = 106
        elif data == 1 and sex == 1 and dev == 0 and loc == 2:
            meta_label = 107
        elif data == 1 and sex == 1 and dev == 0 and loc == 3:
            meta_label = 108
        elif data == 1 and sex == 1 and dev == 0 and loc == 4:
            meta_label = 109
        elif data == 1 and sex == 1 and dev == 0 and loc == 5:
            meta_label = 110
        elif data == 1 and sex == 1 and dev == 0 and loc == 6:
            meta_label = 111
        elif data == 1 and sex == 1 and dev == 1 and loc == 0:
            meta_label = 112
        elif data == 1 and sex == 1 and dev == 1 and loc == 1:
            meta_label = 113
        elif data == 1 and sex == 1 and dev == 1 and loc == 2:
            meta_label = 114
        elif data == 1 and sex == 1 and dev == 1 and loc == 3:
            meta_label = 115
        elif data == 1 and sex == 1 and dev == 1 and loc == 4:
            meta_label = 116
        elif data == 1 and sex == 1 and dev == 1 and loc == 5:
            meta_label = 117
        elif data == 1 and sex == 1 and dev == 1 and loc == 6:
            meta_label = 118
        elif data == 1 and sex == 1 and dev == 2 and loc == 0:
            meta_label = 119
        elif data == 1 and sex == 1 and dev == 2 and loc == 1:
            meta_label = 120
        elif data == 1 and sex == 1 and dev == 2 and loc == 2:
            meta_label = 121
        elif data == 1 and sex == 1 and dev == 2 and loc == 3:
            meta_label = 122
        elif data == 1 and sex == 1 and dev == 2 and loc == 4:
            meta_label = 123
        elif data == 1 and sex == 1 and dev == 2 and loc == 5:
            meta_label = 124
        elif data == 1 and sex == 1 and dev == 2 and loc == 6:
            meta_label = 125
        elif data == 1 and sex == 1 and dev == 3 and loc == 0:
            meta_label = 126
        elif data == 1 and sex == 1 and dev == 3 and loc == 1:
            meta_label = 127
        elif data == 1 and sex == 1 and dev == 3 and loc == 2:
            meta_label = 128
        elif data == 1 and sex == 1 and dev == 3 and loc == 3:
            meta_label = 129
        elif data == 1 and sex == 1 and dev == 3 and loc == 4:
            meta_label = 130
        elif data == 1 and sex == 1 and dev == 3 and loc == 5:
            meta_label = 131
        elif data == 1 and sex == 1 and dev == 3 and loc == 6:
            meta_label = 132
        elif data == 1 and sex == 1 and dev == 4 and loc == 0:
            meta_label = 133
        elif data == 1 and sex == 1 and dev == 4 and loc == 1:
            meta_label = 134
        elif data == 1 and sex == 1 and dev == 4 and loc == 2:
            meta_label = 135
        elif data == 1 and sex == 1 and dev == 4 and loc == 3:
            meta_label = 136
        elif data == 1 and sex == 1 and dev == 4 and loc == 4:
            meta_label = 137
        elif data == 1 and sex == 1 and dev == 4 and loc == 5:
            meta_label = 138
        elif data == 1 and sex == 1 and dev == 4 and loc == 6:
            meta_label = 139
    
    elif args.meta_mode == 'age_data_dev_loc_sex':
        if sex == 0:
            if age >= 19 and data == 0 and dev == 0 and loc == 0:
                meta_label = 0
            elif age >= 19 and data == 0 and dev == 0 and loc == 1:
                meta_label = 1
            elif age >= 19 and data == 0 and dev == 0 and loc == 2:
                meta_label = 2
            elif age >= 19 and data == 0 and dev == 0 and loc == 3:
                meta_label = 3
            elif age >= 19 and data == 0 and dev == 0 and loc == 4:
                meta_label = 4
            elif age >= 19 and data == 0 and dev == 0 and loc == 5:
                meta_label = 5
            elif age >= 19 and data == 0 and dev == 0 and loc == 6:
                meta_label = 6
            elif age >= 19 and data == 0 and dev == 1 and loc == 0:
                meta_label = 7
            elif age >= 19 and data == 0 and dev == 1 and loc == 1:
                meta_label = 8
            elif age >= 19 and data == 0 and dev == 1 and loc == 2:
                meta_label = 9
            elif age >= 19 and data == 0 and dev == 1 and loc == 3:
                meta_label = 10
            elif age >= 19 and data == 0 and dev == 1 and loc == 4:
                meta_label = 11
            elif age >= 19 and data == 0 and dev == 1 and loc == 5:
                meta_label = 12
            elif age >= 19 and data == 0 and dev == 1 and loc == 6:
                meta_label = 13
            elif age >= 19 and data == 0 and dev == 2 and loc == 0:
                meta_label = 14
            elif age >= 19 and data == 0 and dev == 2 and loc == 1:
                meta_label = 15
            elif age >= 19 and data == 0 and dev == 2 and loc == 2:
                meta_label = 16
            elif age >= 19 and data == 0 and dev == 2 and loc == 3:
                meta_label = 17
            elif age >= 19 and data == 0 and dev == 2 and loc == 4:
                meta_label = 18
            elif age >= 19 and data == 0 and dev == 2 and loc == 5:
                meta_label = 19
            elif age >= 19 and data == 0 and dev == 2 and loc == 6:
                meta_label = 20
            elif age >= 19 and data == 0 and dev == 3 and loc == 0:
                meta_label = 21
            elif age >= 19 and data == 0 and dev == 3 and loc == 1:
                meta_label = 22
            elif age >= 19 and data == 0 and dev == 3 and loc == 2:
                meta_label = 23
            elif age >= 19 and data == 0 and dev == 3 and loc == 3:
                meta_label = 24
            elif age >= 19 and data == 0 and dev == 3 and loc == 4:
                meta_label = 25
            elif age >= 19 and data == 0 and dev == 3 and loc == 5:
                meta_label = 26
            elif age >= 19 and data == 0 and dev == 3 and loc == 6:
                meta_label = 27
            elif age >= 19 and data == 0 and dev == 4 and loc == 0:
                meta_label = 28
            elif age >= 19 and data == 0 and dev == 4 and loc == 1:
                meta_label = 28
            elif age >= 19 and data == 0 and dev == 4 and loc == 2:
                meta_label = 30
            elif age >= 19 and data == 0 and dev == 4 and loc == 3:
                meta_label = 31
            elif age >= 19 and data == 0 and dev == 4 and loc == 4:
                meta_label = 32
            elif age >= 19 and data == 0 and dev == 4 and loc == 5:
                meta_label = 33
            elif age >= 19 and data == 0 and dev == 4 and loc == 6:
                meta_label = 34
            elif age >= 19 and data == 1 and dev == 0 and loc == 0:
                meta_label = 35
            elif age >= 19 and data == 1 and dev == 0 and loc == 1:
                meta_label = 36
            elif age >= 19 and data == 1 and dev == 0 and loc == 2:
                meta_label = 37
            elif age >= 19 and data == 1 and dev == 0 and loc == 3:
                meta_label = 38
            elif age >= 19 and data == 1 and dev == 0 and loc == 4:
                meta_label = 39
            elif age >= 19 and data == 1 and dev == 0 and loc == 5:
                meta_label = 40
            elif age >= 19 and data == 1 and dev == 0 and loc == 6:
                meta_label = 41
            elif age >= 19 and data == 1 and dev == 1 and loc == 0:
                meta_label = 42
            elif age >= 19 and data == 1 and dev == 1 and loc == 1:
                meta_label = 43
            elif age >= 19 and data == 1 and dev == 1 and loc == 2:
                meta_label = 44
            elif age >= 19 and data == 1 and dev == 1 and loc == 3:
                meta_label = 45
            elif age >= 19 and data == 1 and dev == 1 and loc == 4:
                meta_label = 46
            elif age >= 19 and data == 1 and dev == 1 and loc == 5:
                meta_label = 47
            elif age >= 19 and data == 1 and dev == 1 and loc == 6:
                meta_label = 48
            elif age >= 19 and data == 1 and dev == 2 and loc == 0:
                meta_label = 49
            elif age >= 19 and data == 1 and dev == 2 and loc == 1:
                meta_label = 50
            elif age >= 19 and data == 1 and dev == 2 and loc == 2:
                meta_label = 51
            elif age >= 19 and data == 1 and dev == 2 and loc == 3:
                meta_label = 52
            elif age >= 19 and data == 1 and dev == 2 and loc == 4:
                meta_label = 53
            elif age >= 19 and data == 1 and dev == 2 and loc == 5:
                meta_label = 54
            elif age >= 19 and data == 1 and dev == 2 and loc == 6:
                meta_label = 55
            elif age >= 19 and data == 1 and dev == 3 and loc == 0:
                meta_label = 56
            elif age >= 19 and data == 1 and dev == 3 and loc == 1:
                meta_label = 57
            elif age >= 19 and data == 1 and dev == 3 and loc == 2:
                meta_label = 58
            elif age >= 19 and data == 1 and dev == 3 and loc == 3:
                meta_label = 59
            elif age >= 19 and data == 1 and dev == 3 and loc == 4:
                meta_label = 60
            elif age >= 19 and data == 1 and dev == 3 and loc == 5:
                meta_label = 61
            elif age >= 19 and data == 1 and dev == 3 and loc == 6:
                meta_label = 62
            elif age >= 19 and data == 1 and dev == 4 and loc == 0:
                meta_label = 63
            elif age >= 19 and data == 1 and dev == 4 and loc == 1:
                meta_label = 64
            elif age >= 19 and data == 1 and dev == 4 and loc == 2:
                meta_label = 65
            elif age >= 19 and data == 1 and dev == 4 and loc == 3:
                meta_label = 66
            elif age >= 19 and data == 1 and dev == 4 and loc == 4:
                meta_label = 67
            elif age >= 19 and data == 1 and dev == 4 and loc == 5:
                meta_label = 68
            elif age >= 19 and data == 1 and dev == 4 and loc == 6:
                meta_label = 69
            
            elif age < 19 and data == 0 and dev == 0 and loc == 0:
                meta_label = 70
            elif age < 19 and data == 0 and dev == 0 and loc == 1:
                meta_label = 71
            elif age < 19 and data == 0 and dev == 0 and loc == 2:
                meta_label = 72
            elif age < 19 and data == 0 and dev == 0 and loc == 3:
                meta_label = 73
            elif age < 19 and data == 0 and dev == 0 and loc == 4:
                meta_label = 74
            elif age < 19 and data == 0 and dev == 0 and loc == 5:
                meta_label = 75
            elif age < 19 and data == 0 and dev == 0 and loc == 6:
                meta_label = 76
            elif age < 19 and data == 0 and dev == 1 and loc == 0:
                meta_label = 77
            elif age < 19 and data == 0 and dev == 1 and loc == 1:
                meta_label = 78
            elif age < 19 and data == 0 and dev == 1 and loc == 2:
                meta_label = 79
            elif age < 19 and data == 0 and dev == 1 and loc == 3:
                meta_label = 80
            elif age < 19 and data == 0 and dev == 1 and loc == 4:
                meta_label = 81
            elif age < 19 and data == 0 and dev == 1 and loc == 5:
                meta_label = 82
            elif age < 19 and data == 0 and dev == 1 and loc == 6:
                meta_label = 83
            elif age < 19 and data == 0 and dev == 2 and loc == 0:
                meta_label = 84
            elif age < 19 and data == 0 and dev == 2 and loc == 1:
                meta_label = 85
            elif age < 19 and data == 0 and dev == 2 and loc == 2:
                meta_label = 86
            elif age < 19 and data == 0 and dev == 2 and loc == 3:
                meta_label = 87
            elif age < 19 and data == 0 and dev == 2 and loc == 4:
                meta_label = 88
            elif age < 19 and data == 0 and dev == 2 and loc == 5:
                meta_label = 89
            elif age < 19 and data == 0 and dev == 2 and loc == 6:
                meta_label = 90
            elif age < 19 and data == 0 and dev == 3 and loc == 0:
                meta_label = 91
            elif age < 19 and data == 0 and dev == 3 and loc == 1:
                meta_label = 92
            elif age < 19 and data == 0 and dev == 3 and loc == 2:
                meta_label = 93
            elif age < 19 and data == 0 and dev == 3 and loc == 3:
                meta_label = 94
            elif age < 19 and data == 0 and dev == 3 and loc == 4:
                meta_label = 95
            elif age < 19 and data == 0 and dev == 3 and loc == 5:
                meta_label = 96
            elif age < 19 and data == 0 and dev == 3 and loc == 6:
                meta_label = 97
            elif age < 19 and data == 0 and dev == 4 and loc == 0:
                meta_label = 98
            elif age < 19 and data == 0 and dev == 4 and loc == 1:
                meta_label = 99
            elif age < 19 and data == 0 and dev == 4 and loc == 2:
                meta_label = 100
            elif age < 19 and data == 0 and dev == 4 and loc == 3:
                meta_label = 101
            elif age < 19 and data == 0 and dev == 4 and loc == 4:
                meta_label = 102
            elif age < 19 and data == 0 and dev == 4 and loc == 5:
                meta_label = 103
            elif age < 19 and data == 0 and dev == 4 and loc == 6:
                meta_label = 104
            elif age < 19 and data == 1 and dev == 0 and loc == 0:
                meta_label = 105
            elif age < 19 and data == 1 and dev == 0 and loc == 1:
                meta_label = 106
            elif age < 19 and data == 1 and dev == 0 and loc == 2:
                meta_label = 107
            elif age < 19 and data == 1 and dev == 0 and loc == 3:
                meta_label = 108
            elif age < 19 and data == 1 and dev == 0 and loc == 4:
                meta_label = 109
            elif age < 19 and data == 1 and dev == 0 and loc == 5:
                meta_label = 110
            elif age < 19 and data == 1 and dev == 0 and loc == 6:
                meta_label = 111
            elif age < 19 and data == 1 and dev == 1 and loc == 0:
                meta_label = 112
            elif age < 19 and data == 1 and dev == 1 and loc == 1:
                meta_label = 113
            elif age < 19 and data == 1 and dev == 1 and loc == 2:
                meta_label = 114
            elif age < 19 and data == 1 and dev == 1 and loc == 3:
                meta_label = 115
            elif age < 19 and data == 1 and dev == 1 and loc == 4:
                meta_label = 116
            elif age < 19 and data == 1 and dev == 1 and loc == 5:
                meta_label = 117
            elif age < 19 and data == 1 and dev == 1 and loc == 6:
                meta_label = 118
            elif age < 19 and data == 1 and dev == 2 and loc == 0:
                meta_label = 119
            elif age < 19 and data == 1 and dev == 2 and loc == 1:
                meta_label = 120
            elif age < 19 and data == 1 and dev == 2 and loc == 2:
                meta_label = 121
            elif age < 19 and data == 1 and dev == 2 and loc == 3:
                meta_label = 122
            elif age < 19 and data == 1 and dev == 2 and loc == 4:
                meta_label = 123
            elif age < 19 and data == 1 and dev == 2 and loc == 5:
                meta_label = 124
            elif age < 19 and data == 1 and dev == 2 and loc == 6:
                meta_label = 125
            elif age < 19 and data == 1 and dev == 3 and loc == 0:
                meta_label = 126
            elif age < 19 and data == 1 and dev == 3 and loc == 1:
                meta_label = 127
            elif age < 19 and data == 1 and dev == 3 and loc == 2:
                meta_label = 128
            elif age < 19 and data == 1 and dev == 3 and loc == 3:
                meta_label = 129
            elif age < 19 and data == 1 and dev == 3 and loc == 4:
                meta_label = 130
            elif age < 19 and data == 1 and dev == 3 and loc == 5:
                meta_label = 131
            elif age < 19 and data == 1 and dev == 3 and loc == 6:
                meta_label = 132
            elif age < 19 and data == 1 and dev == 4 and loc == 0:
                meta_label = 133
            elif age < 19 and data == 1 and dev == 4 and loc == 1:
                meta_label = 134
            elif age < 19 and data == 1 and dev == 4 and loc == 2:
                meta_label = 135
            elif age < 19 and data == 1 and dev == 4 and loc == 3:
                meta_label = 136
            elif age < 19 and data == 1 and dev == 4 and loc == 4:
                meta_label = 137
            elif age < 19 and data == 1 and dev == 4 and loc == 5:
                meta_label = 138
            elif age < 19 and data == 1 and dev == 4 and loc == 6:
                meta_label = 139
        
        else:
            if age >= 19 and data == 0 and dev == 0 and loc == 0:
                meta_label = 0
            elif age >= 19 and data == 0 and dev == 0 and loc == 1:
                meta_label = 1
            elif age >= 19 and data == 0 and dev == 0 and loc == 2:
                meta_label = 2
            elif age >= 19 and data == 0 and dev == 0 and loc == 3:
                meta_label = 3
            elif age >= 19 and data == 0 and dev == 0 and loc == 4:
                meta_label = 4
            elif age >= 19 and data == 0 and dev == 0 and loc == 5:
                meta_label = 5
            elif age >= 19 and data == 0 and dev == 0 and loc == 6:
                meta_label = 6
            elif age >= 19 and data == 0 and dev == 1 and loc == 0:
                meta_label = 7
            elif age >= 19 and data == 0 and dev == 1 and loc == 1:
                meta_label = 8
            elif age >= 19 and data == 0 and dev == 1 and loc == 2:
                meta_label = 9
            elif age >= 19 and data == 0 and dev == 1 and loc == 3:
                meta_label = 10
            elif age >= 19 and data == 0 and dev == 1 and loc == 4:
                meta_label = 11
            elif age >= 19 and data == 0 and dev == 1 and loc == 5:
                meta_label = 12
            elif age >= 19 and data == 0 and dev == 1 and loc == 6:
                meta_label = 13
            elif age >= 19 and data == 0 and dev == 2 and loc == 0:
                meta_label = 14
            elif age >= 19 and data == 0 and dev == 2 and loc == 1:
                meta_label = 15
            elif age >= 19 and data == 0 and dev == 2 and loc == 2:
                meta_label = 16
            elif age >= 19 and data == 0 and dev == 2 and loc == 3:
                meta_label = 17
            elif age >= 19 and data == 0 and dev == 2 and loc == 4:
                meta_label = 18
            elif age >= 19 and data == 0 and dev == 2 and loc == 5:
                meta_label = 19
            elif age >= 19 and data == 0 and dev == 2 and loc == 6:
                meta_label = 20
            elif age >= 19 and data == 0 and dev == 3 and loc == 0:
                meta_label = 21
            elif age >= 19 and data == 0 and dev == 3 and loc == 1:
                meta_label = 22
            elif age >= 19 and data == 0 and dev == 3 and loc == 2:
                meta_label = 23
            elif age >= 19 and data == 0 and dev == 3 and loc == 3:
                meta_label = 24
            elif age >= 19 and data == 0 and dev == 3 and loc == 4:
                meta_label = 25
            elif age >= 19 and data == 0 and dev == 3 and loc == 5:
                meta_label = 26
            elif age >= 19 and data == 0 and dev == 3 and loc == 6:
                meta_label = 27
            elif age >= 19 and data == 0 and dev == 4 and loc == 0:
                meta_label = 28
            elif age >= 19 and data == 0 and dev == 4 and loc == 1:
                meta_label = 28
            elif age >= 19 and data == 0 and dev == 4 and loc == 2:
                meta_label = 30
            elif age >= 19 and data == 0 and dev == 4 and loc == 3:
                meta_label = 31
            elif age >= 19 and data == 0 and dev == 4 and loc == 4:
                meta_label = 32
            elif age >= 19 and data == 0 and dev == 4 and loc == 5:
                meta_label = 33
            elif age >= 19 and data == 0 and dev == 4 and loc == 6:
                meta_label = 34
            elif age >= 19 and data == 1 and dev == 0 and loc == 0:
                meta_label = 35
            elif age >= 19 and data == 1 and dev == 0 and loc == 1:
                meta_label = 36
            elif age >= 19 and data == 1 and dev == 0 and loc == 2:
                meta_label = 37
            elif age >= 19 and data == 1 and dev == 0 and loc == 3:
                meta_label = 38
            elif age >= 19 and data == 1 and dev == 0 and loc == 4:
                meta_label = 39
            elif age >= 19 and data == 1 and dev == 0 and loc == 5:
                meta_label = 40
            elif age >= 19 and data == 1 and dev == 0 and loc == 6:
                meta_label = 41
            elif age >= 19 and data == 1 and dev == 1 and loc == 0:
                meta_label = 42
            elif age >= 19 and data == 1 and dev == 1 and loc == 1:
                meta_label = 43
            elif age >= 19 and data == 1 and dev == 1 and loc == 2:
                meta_label = 44
            elif age >= 19 and data == 1 and dev == 1 and loc == 3:
                meta_label = 45
            elif age >= 19 and data == 1 and dev == 1 and loc == 4:
                meta_label = 46
            elif age >= 19 and data == 1 and dev == 1 and loc == 5:
                meta_label = 47
            elif age >= 19 and data == 1 and dev == 1 and loc == 6:
                meta_label = 48
            elif age >= 19 and data == 1 and dev == 2 and loc == 0:
                meta_label = 49
            elif age >= 19 and data == 1 and dev == 2 and loc == 1:
                meta_label = 50
            elif age >= 19 and data == 1 and dev == 2 and loc == 2:
                meta_label = 51
            elif age >= 19 and data == 1 and dev == 2 and loc == 3:
                meta_label = 52
            elif age >= 19 and data == 1 and dev == 2 and loc == 4:
                meta_label = 53
            elif age >= 19 and data == 1 and dev == 2 and loc == 5:
                meta_label = 54
            elif age >= 19 and data == 1 and dev == 2 and loc == 6:
                meta_label = 55
            elif age >= 19 and data == 1 and dev == 3 and loc == 0:
                meta_label = 56
            elif age >= 19 and data == 1 and dev == 3 and loc == 1:
                meta_label = 57
            elif age >= 19 and data == 1 and dev == 3 and loc == 2:
                meta_label = 58
            elif age >= 19 and data == 1 and dev == 3 and loc == 3:
                meta_label = 59
            elif age >= 19 and data == 1 and dev == 3 and loc == 4:
                meta_label = 60
            elif age >= 19 and data == 1 and dev == 3 and loc == 5:
                meta_label = 61
            elif age >= 19 and data == 1 and dev == 3 and loc == 6:
                meta_label = 62
            elif age >= 19 and data == 1 and dev == 4 and loc == 0:
                meta_label = 63
            elif age >= 19 and data == 1 and dev == 4 and loc == 1:
                meta_label = 64
            elif age >= 19 and data == 1 and dev == 4 and loc == 2:
                meta_label = 65
            elif age >= 19 and data == 1 and dev == 4 and loc == 3:
                meta_label = 66
            elif age >= 19 and data == 1 and dev == 4 and loc == 4:
                meta_label = 67
            elif age >= 19 and data == 1 and dev == 4 and loc == 5:
                meta_label = 68
            elif age >= 19 and data == 1 and dev == 4 and loc == 6:
                meta_label = 69
            
            elif age < 19 and data == 0 and dev == 0 and loc == 0:
                meta_label = 70
            elif age < 19 and data == 0 and dev == 0 and loc == 1:
                meta_label = 71
            elif age < 19 and data == 0 and dev == 0 and loc == 2:
                meta_label = 72
            elif age < 19 and data == 0 and dev == 0 and loc == 3:
                meta_label = 73
            elif age < 19 and data == 0 and dev == 0 and loc == 4:
                meta_label = 74
            elif age < 19 and data == 0 and dev == 0 and loc == 5:
                meta_label = 75
            elif age < 19 and data == 0 and dev == 0 and loc == 6:
                meta_label = 76
            elif age < 19 and data == 0 and dev == 1 and loc == 0:
                meta_label = 77
            elif age < 19 and data == 0 and dev == 1 and loc == 1:
                meta_label = 78
            elif age < 19 and data == 0 and dev == 1 and loc == 2:
                meta_label = 79
            elif age < 19 and data == 0 and dev == 1 and loc == 3:
                meta_label = 80
            elif age < 19 and data == 0 and dev == 1 and loc == 4:
                meta_label = 81
            elif age < 19 and data == 0 and dev == 1 and loc == 5:
                meta_label = 82
            elif age < 19 and data == 0 and dev == 1 and loc == 6:
                meta_label = 83
            elif age < 19 and data == 0 and dev == 2 and loc == 0:
                meta_label = 84
            elif age < 19 and data == 0 and dev == 2 and loc == 1:
                meta_label = 85
            elif age < 19 and data == 0 and dev == 2 and loc == 2:
                meta_label = 86
            elif age < 19 and data == 0 and dev == 2 and loc == 3:
                meta_label = 87
            elif age < 19 and data == 0 and dev == 2 and loc == 4:
                meta_label = 88
            elif age < 19 and data == 0 and dev == 2 and loc == 5:
                meta_label = 89
            elif age < 19 and data == 0 and dev == 2 and loc == 6:
                meta_label = 90
            elif age < 19 and data == 0 and dev == 3 and loc == 0:
                meta_label = 91
            elif age < 19 and data == 0 and dev == 3 and loc == 1:
                meta_label = 92
            elif age < 19 and data == 0 and dev == 3 and loc == 2:
                meta_label = 93
            elif age < 19 and data == 0 and dev == 3 and loc == 3:
                meta_label = 94
            elif age < 19 and data == 0 and dev == 3 and loc == 4:
                meta_label = 95
            elif age < 19 and data == 0 and dev == 3 and loc == 5:
                meta_label = 96
            elif age < 19 and data == 0 and dev == 3 and loc == 6:
                meta_label = 97
            elif age < 19 and data == 0 and dev == 4 and loc == 0:
                meta_label = 98
            elif age < 19 and data == 0 and dev == 4 and loc == 1:
                meta_label = 99
            elif age < 19 and data == 0 and dev == 4 and loc == 2:
                meta_label = 100
            elif age < 19 and data == 0 and dev == 4 and loc == 3:
                meta_label = 101
            elif age < 19 and data == 0 and dev == 4 and loc == 4:
                meta_label = 102
            elif age < 19 and data == 0 and dev == 4 and loc == 5:
                meta_label = 103
            elif age < 19 and data == 0 and dev == 4 and loc == 6:
                meta_label = 104
            elif age < 19 and data == 1 and dev == 0 and loc == 0:
                meta_label = 105
            elif age < 19 and data == 1 and dev == 0 and loc == 1:
                meta_label = 106
            elif age < 19 and data == 1 and dev == 0 and loc == 2:
                meta_label = 107
            elif age < 19 and data == 1 and dev == 0 and loc == 3:
                meta_label = 108
            elif age < 19 and data == 1 and dev == 0 and loc == 4:
                meta_label = 109
            elif age < 19 and data == 1 and dev == 0 and loc == 5:
                meta_label = 110
            elif age < 19 and data == 1 and dev == 0 and loc == 6:
                meta_label = 111
            elif age < 19 and data == 1 and dev == 1 and loc == 0:
                meta_label = 112
            elif age < 19 and data == 1 and dev == 1 and loc == 1:
                meta_label = 113
            elif age < 19 and data == 1 and dev == 1 and loc == 2:
                meta_label = 114
            elif age < 19 and data == 1 and dev == 1 and loc == 3:
                meta_label = 115
            elif age < 19 and data == 1 and dev == 1 and loc == 4:
                meta_label = 116
            elif age < 19 and data == 1 and dev == 1 and loc == 5:
                meta_label = 117
            elif age < 19 and data == 1 and dev == 1 and loc == 6:
                meta_label = 118
            elif age < 19 and data == 1 and dev == 2 and loc == 0:
                meta_label = 119
            elif age < 19 and data == 1 and dev == 2 and loc == 1:
                meta_label = 120
            elif age < 19 and data == 1 and dev == 2 and loc == 2:
                meta_label = 121
            elif age < 19 and data == 1 and dev == 2 and loc == 3:
                meta_label = 122
            elif age < 19 and data == 1 and dev == 2 and loc == 4:
                meta_label = 123
            elif age < 19 and data == 1 and dev == 2 and loc == 5:
                meta_label = 124
            elif age < 19 and data == 1 and dev == 2 and loc == 6:
                meta_label = 125
            elif age < 19 and data == 1 and dev == 3 and loc == 0:
                meta_label = 126
            elif age < 19 and data == 1 and dev == 3 and loc == 1:
                meta_label = 127
            elif age < 19 and data == 1 and dev == 3 and loc == 2:
                meta_label = 128
            elif age < 19 and data == 1 and dev == 3 and loc == 3:
                meta_label = 129
            elif age < 19 and data == 1 and dev == 3 and loc == 4:
                meta_label = 130
            elif age < 19 and data == 1 and dev == 3 and loc == 5:
                meta_label = 131
            elif age < 19 and data == 1 and dev == 3 and loc == 6:
                meta_label = 132
            elif age < 19 and data == 1 and dev == 4 and loc == 0:
                meta_label = 133
            elif age < 19 and data == 1 and dev == 4 and loc == 1:
                meta_label = 134
            elif age < 19 and data == 1 and dev == 4 and loc == 2:
                meta_label = 135
            elif age < 19 and data == 1 and dev == 4 and loc == 3:
                meta_label = 136
            elif age < 19 and data == 1 and dev == 4 and loc == 4:
                meta_label = 137
            elif age < 19 and data == 1 and dev == 4 and loc == 5:
                meta_label = 138
            elif age < 19 and data == 1 and dev == 4 and loc == 6:
                meta_label = 139
            meta_label += 140
    
                
    return meta_label


def get_individual_cycles_torchaudio(args, recording_annotations, metadata, data_folder, filename, sample_rate, n_cls):
    sample_data = []
    fpath = os.path.join(data_folder, filename+'.wav')
    data, sr = torchaudio.load(fpath)
    
    if sr != sample_rate:
        resample = T.Resample(sr, sample_rate)
        data = resample(data)
    
    fade_samples_ratio = 16
    fade_samples = int(sample_rate / fade_samples_ratio)
    fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
    data = fade(data)
    
    for idx in recording_annotations.index:
        row = recording_annotations.loc[idx]

        start = row['Start'] # start time (second)
        end = row['End'] # end time (second)
        audio_chunk = _slice_data_torchaudio(start, end, data, sample_rate)

        if args.class_split == 'lungsound':
            crackles = row['Crackles']
            wheezes = row['Wheezes']
            label = _get_lungsound_label(crackles, wheezes, n_cls, args)
            meta_label = get_meta_infor(metadata, args, label)
            #print('meta_label', meta_label)
            sample_data.append((audio_chunk, _get_lungsound_label(crackles, wheezes, n_cls, args), meta_label))
        elif args.class_split == 'wheeze':
            crackles = row['Crackles']
            wheezes = row['Wheezes']
            label = _get_lungsound_label(crackles, wheezes, n_cls, args)
            #meta_label = get_meta_infor(metadata, label, args)
            
            sample_data.append((audio_chunk, _get_lungsound_label(crackles, wheezes, n_cls, args), metadata))
        elif args.class_split == 'diagnosis':
            disease = row['Disease']            
            sample_data.append((audio_chunk, _get_diagnosis_label(disease, n_cls), meta_label))
        
    padded_sample_data = []
    for data, label, m_label in sample_data:
        data = cut_pad_sample_torchaudio(data, args) # --> resample to [1, 128000] --> 8 seconds
        padded_sample_data.append((data, label, m_label))

    return padded_sample_data


def generate_fbank(args, audio, sample_rate, n_mels=128): 
    """
    use torchaudio library to convert mel fbank for AST model
    """    
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    
    if args.model in ['ast']:
        mean, std =  -4.2677393, 4.5689974
    else:
        mean, std = fbank.mean(), fbank.std()
    fbank = (fbank - mean) / (std * 2) # mean / std
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank 


# ==========================================================================


# ==========================================================================
""" evaluation metric """
def get_score(hits, counts, pflag=False):
    # normal accuracy
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        # print("************* Metrics ******************")
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))

    return sp, se, sc
# ==========================================================================
