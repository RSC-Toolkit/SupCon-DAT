from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from copy import deepcopy

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio_snubh_ver2, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio


class SNUBHDataset2(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False):
        data_folder = os.path.join(args.data_folder, 'snubh_dataset')
        test_fold = args.test_fold
        
        self.data_folder = data_folder
        self.train_flag = train_flag
        self.split = 'train' if train_flag else 'test'
        self.transform = transform
        self.args = args
        self.mean_std = mean_std

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.n_mels = args.n_mels
        
        
        cache_path = './data/test_snubh2.pt'
                
        if not os.path.isfile(cache_path):                        
            meta_files = os.path.join(self.data_folder, 'training.csv' if self.train_flag else 'test.csv')
            
            #meta_file = pd.read_csv(metafiles, names=['path', 'gender', 'age', 'label', 'device'], delimiter= '\t')
            annotation = pd.read_csv(meta_files)
            data_path = annotation['path'].values.tolist()
            data_label = annotation['label'].values.tolist()
            meta_gender = annotation['gender'].values.tolist()
            meta_age = annotation['age'].values.tolist()
            meta_device = annotation['device'].values.tolist()
            
            self.file_to_metadata = {}
            annotation_dict = {}
            
            
            for fpath, label, gender, age, device in zip(data_path, data_label, meta_gender, meta_age, meta_device):
                self.file_to_metadata[fpath] = torch.tensor([0 if label == 'Nonwheezing' else 1, 0 if gender == 'M' else 1, int(age), 4 if device == 'Jabes' else None]) #label, sex, age, device
            
            self.audio_data = []  # each sample is a tuple with (audio_data, label, metadata)
    
            if print_flag:
                print('*' * 20)  
                print("Extracting individual breathing cycles..")
    
            self.cycle_list = []
            
            for idx, filename in enumerate(data_path):
                sample_data = get_individual_cycles_torchaudio_snubh_ver2(args, filename, self.file_to_metadata[filename], self.data_folder, filename, self.sample_rate, args.n_cls)
                cycles_with_labels = [(data[0], data[1], data[2]) for data in sample_data]
                self.cycle_list.extend(cycles_with_labels)
            
            for sample in self.cycle_list:
                self.audio_data.append(sample)
            
            self.class_nums = np.zeros(args.n_cls)
            if args.m_cls:
                self.domain_nums = np.zeros(args.m_cls)
                
            for sample in self.audio_data:
                if args.domain_adaptation or args.domain_adaptation2:
                    self.class_nums[sample[1]] += 1
                    self.domain_nums[sample[2]] += 1
                else:
                    self.class_nums[sample[1]] += 1
                
            self.class_ratio = self.class_nums / sum(self.class_nums) * 100
            if args.m_cls:
                self.domain_ratio = self.domain_nums / sum(self.domain_nums) * 100
                        
            # ==========================================================================
            """ convert fbank """
            self.audio_images = []
            for index in range(len(self.audio_data)): #for the training set, 4142
                audio, label, meta_label = self.audio_data[index][0], self.audio_data[index][1], self.audio_data[index][2] # wav, label, metadata --> [1, 128000], 0~3, [1, 7]
                
                audio_image = []
                for aug_idx in range(self.args.raw_augment+1):
                    if aug_idx > 0:
                        if self.train_flag:
                            audio = augment_raw_audio(np.asarray(audio.squeeze(0)), self.sample_rate, self.args)
                            audio = cut_pad_sample_torchaudio(torch.tensor(audio), args)                
                        
                        image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                        audio_image.append(image)
                    else:
                        image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                        audio_image.append(image)
                self.audio_images.append((audio_image, label, meta_label))
            
            
            torch.save(self.audio_images, './data/test_snubh2.pt')
        else:
            self.audio_images = torch.load('./data/test_snubh2.pt')
            
            # ==========================================================================

    def __getitem__(self, index):
        audio_images, label, meta_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2]
        #print('index {} audio_images {} label {} label_type {} meta_label {} meta_label_type {}'.format(index, len(audio_images), label, type(label), meta_label, type(meta_label)))

        if self.args.raw_augment and self.train_flag and not self.mean_std:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        if meta_label is None or not self.train_flag:
            return audio_image, label
        else:
            return audio_image, (label, meta_label)

    def __len__(self):
        return len(self.audio_images)