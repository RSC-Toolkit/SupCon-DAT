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

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio_snubh, get_individual_cycles_torchaudio_smart, cut_pad_sample_torchaudio, get_meta_infor

from .augmentation import augment_raw_audio


class SMARTDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False):
        #data_folder = os.path.join(args.data_folder, 'snubh_final')
        data_folder = os.path.join(args.data_folder, 'smartsound/data')
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
        
        
        if args.multitask:
            if self.train_flag and os.path.isfile('./data/training_smart_mtl_lung_class_nums.pt') and os.path.isfile('./data/training_smart_mtl_disease_class_nums.pt'):
                self.lung_class_nums = torch.load('./data/training_smart_mtl_lung_class_nums.pt')
                self.disease_class_nums = torch.load('./data/training_smart_mtl_disease_class_nums.pt')
            cache_path = './data/training_smart_mtl.pt' if self.train_flag else './data/test_smart_mtl.pt'        
        elif args.multitask_domain:
            if self.train_flag and os.path.isfile('./data/training_smart_mtld_lung_class_nums.pt') and os.path.isfile('./data/training_smart_mtld_disease_class_nums.pt'):
                self.lung_class_nums = torch.load('./data/training_smart_mtld_lung_class_nums.pt')
                self.disease_class_nums = torch.load('./data/training_smart_mtld_disease_class_nums.pt')
            cache_path = './data/training_smart_mtld.pt' if self.train_flag else './data/test_smart_mtld.pt'        
        else:        
            if self.train_flag and os.path.isfile('./data/training_smart_class_nums.pt'):
                self.class_nums = torch.load('./data/training_smart_class_nums.pt')
            cache_path = './data/training_smart.pt' if self.train_flag else './data/test_smart.pt'
        
        if not os.path.isfile(cache_path):
            annotation_files = os.path.join(self.data_folder, 'training.csv' if self.train_flag else 'test.csv')
            
            annotation = pd.read_csv(annotation_files, encoding='cp949') if args.dataset =='smart' else pd.read_csv(annotation_files) 
            paths = annotation['filename'].values.tolist()
            labels = annotation['label'].values.tolist()
            genders = annotation['sex'].values.tolist() 
            ages = annotation['age'].values.tolist() #
            devices = annotation['device'].values.tolist() #
            locations = annotation['location'].values.tolist() #
            diseases = annotation['disease'].values.tolist()
            
            self.file_to_metadata = {}
            annotation_dict = {}
            
            
            for fpath, label, gender, age, device, loc, disease in zip(paths, labels, genders, ages, devices, locations, diseases):
                self.file_to_metadata[fpath] = torch.tensor([label, 0 if gender == 'M' else 1, int(age), 5 if device == 'SM-300' else None, int(loc), 2]) #label, sex, age, device, loc, data
            
            self.audio_data = []  # each sample is a tuple with (audio_data, label, metadata)
    
            if print_flag:
                print('*' * 20)  
                print("Extracting individual breathing cycles..")
    
            self.cycle_list = []
            
            for idx, (filename, disease) in enumerate(zip(paths, diseases)):
                sample_data = get_individual_cycles_torchaudio_smart(args, filename, self.file_to_metadata[filename], self.data_folder, filename, self.sample_rate, args.n_cls, disease)
                #cycles_with_labels = [(data[0], data[1], data[2]) for data in sample_data]
                cycles_with_labels = [(data[0], data[1], data[2], data[3]) for data in sample_data] if args.multitask or args.multitask_domain else [(data[0], data[1], data[2]) for data in sample_data]
                self.cycle_list.extend(cycles_with_labels)
            
            for sample in self.cycle_list:
                self.audio_data.append(sample)
            
            
            ##
            if args.multitask:
                self.lung_class_nums = np.zeros(args.lung_cls)
                self.disease_class_nums = np.zeros(args.disease_cls)
                
                if args.m_cls:
                    self.domain_nums = np.zeros(args.m_cls)
                    
                for sample in self.audio_data:
                    if args.domain_adaptation or args.domain_adaptation2:
                        self.lung_class_nums[sample[1]] += 1
                        self.disease_class_nums[sample[2]] += 1
                        self.domain_nums[sample[-1]] += 1
                    else:
                        self.lung_class_nums[sample[1]] += 1
                        self.disease_class_nums[sample[2]] += 1
                
                if self.train_flag:
                    lung_class_nums_cache_path = './data/training_smart_mtl_lung_class_nums.pt'
                    disease_class_nums_cache_path = './data/training_smart_mtl_disease_class_nums.pt'
                    if not os.path.isfile(lung_class_nums_cache_path):
                        torch.save(self.lung_class_nums, lung_class_nums_cache_path)
                    if not os.path.isfile(disease_class_nums_cache_path):
                        torch.save(self.disease_class_nums, disease_class_nums_cache_path)
                
                
                self.lung_class_ratio = self.lung_class_nums / sum(self.lung_class_nums) * 100
                self.disease_class_ratio = self.disease_class_nums / sum(self.disease_class_nums) * 100
                if args.m_cls:
                    self.domain_ratio = self.domain_nums / sum(self.domain_nums) * 100
                
                if print_flag:
                    print('[Preprocessed {} dataset information]'.format(self.split))
                    print('total number of audio data: {}'.format(len(self.audio_data)))
                    print('*' * 25)
                    print('For the Lung Label Distribution')
                    for i, (n, p) in enumerate(zip(self.lung_class_nums, self.lung_class_ratio)):
                        print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.lung_list[i]+')', int(n), p))
                    
                    print('For the Disease Label Distribution')
                    for i, (n, p) in enumerate(zip(self.disease_class_nums, self.disease_class_ratio)):
                        print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.disease_list[i]+')', int(n), p))
                    
                    if args.domain_adaptation or args.domain_adaptation2:
                        print('*' * 25)
                        print('For the Meta Label Distribution')            
                        for i, (n, p) in enumerate(zip(self.domain_nums, self.domain_ratio)):
                            print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.meta_cls_list[i]+')', int(n), p))
                
                # ==========================================================================
                """ convert fbank """
                self.audio_images = []
                for index in range(len(self.audio_data)): #for the training set, 4142
                    audio, lung_label, disease_label, meta_label = self.audio_data[index][0], self.audio_data[index][1], self.audio_data[index][2], self.audio_data[index][3] # wav, lung_label, disease_label, metadata
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
                    self.audio_images.append((audio_image, lung_label, disease_label, meta_label))
                
                
                # ==========================================================================
                if self.train_flag:
                    torch.save(self.audio_images, './data/training_smart_mtl.pt')
                else:
                    torch.save(self.audio_images, './data/test_smart_mtl.pt')
            
            elif args.multitask_domain:
                self.lung_class_nums = np.zeros(args.lung_cls)
                self.disease_class_nums = np.zeros(args.disease_cls)
                self.domain_class_nums = np.zeros(args.domain_cls)
                
                    
                for sample in self.audio_data:
                    self.lung_class_nums[sample[1]] += 1
                    self.disease_class_nums[sample[2]] += 1
                    self.domain_class_nums[sample[-1]] += 1
                
                if self.train_flag:
                    lung_class_nums_cache_path = './data/training_smart_mtld_lung_class_nums.pt'
                    disease_class_nums_cache_path = './data/training_smart_mtld_disease_class_nums.pt'
                    if not os.path.isfile(lung_class_nums_cache_path):
                        torch.save(self.lung_class_nums, lung_class_nums_cache_path)
                    if not os.path.isfile(disease_class_nums_cache_path):
                        torch.save(self.disease_class_nums, disease_class_nums_cache_path)
                
                
                            
                self.lung_class_ratio = self.lung_class_nums / sum(self.lung_class_nums) * 100
                self.disease_class_ratio = self.disease_class_nums / sum(self.disease_class_nums) * 100
                self.domain_class_ratio = self.domain_class_nums / sum(self.domain_class_nums) * 100
                
                            
                if print_flag:
                    print('[Preprocessed {} dataset information]'.format(self.split))
                    print('total number of audio data: {}'.format(len(self.audio_data)))
                    print('*' * 25)
                    print('For the Lung Label Distribution')
                    for i, (n, p) in enumerate(zip(self.lung_class_nums, self.lung_class_ratio)):
                        print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.lung_list[i]+')', int(n), p))
                    
                    print('For the Disease Label Distribution')
                    for i, (n, p) in enumerate(zip(self.disease_class_nums, self.disease_class_ratio)):
                        print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.disease_list[i]+')', int(n), p))
                    
                    print('For the Domain Label Distribution')
                    for i, (n, p) in enumerate(zip(self.domain_class_nums, self.domain_class_ratio)):
                        print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.domain_list[i]+')', int(n), p))
                
                # ==========================================================================
                """ convert fbank """
                self.audio_images = []
                for index in range(len(self.audio_data)): #for the training set, 4142
                    audio, lung_label, disease_label, domain_label = self.audio_data[index][0], self.audio_data[index][1], self.audio_data[index][2], self.audio_data[index][3] # wav, lung_label, disease_label, metadata
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
                    self.audio_images.append((audio_image, lung_label, disease_label, domain_label))
                
                
                # ==========================================================================
                if self.train_flag:
                    torch.save(self.audio_images, './data/training_smart_mtld.pt')
                else:
                    torch.save(self.audio_images, './data/test_smart_mtld.pt')
            
            ##
            
            else:          
            
                self.class_nums = np.zeros(args.n_cls)
                if args.m_cls:
                    self.domain_nums = np.zeros(args.m_cls)
                    
                for sample in self.audio_data:
                    if args.domain_adaptation or args.domain_adaptation2:
                        self.class_nums[sample[1]] += 1
                        self.domain_nums[sample[2]] += 1
                    else:
                        self.class_nums[sample[1]] += 1
                
                if self.train_flag:
                    class_nums_cache_path = './data/training_smart_class_nums.pt'
                    if not os.path.isfile(class_nums_cache_path):
                        torch.save(self.class_nums, class_nums_cache_path)
                                                                                            
                    
                self.class_ratio = self.class_nums / sum(self.class_nums) * 100
                if args.m_cls:
                    self.domain_ratio = self.domain_nums / sum(self.domain_nums) * 100
                
                if print_flag:
                    print('[Preprocessed {} dataset information]'.format(self.split))
                    print('total number of audio data: {}'.format(len(self.audio_data)))
                    print('*' * 25)
                    print('For the Label Distribution')
                    for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                        print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))
                    
                    if args.domain_adaptation or args.domain_adaptation2:
                        print('*' * 25)
                        print('For the Meta Label Distribution')            
                        for i, (n, p) in enumerate(zip(self.domain_nums, self.domain_ratio)):
                            print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.meta_cls_list[i]+')', int(n), p))                        
            
                                                            
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
                
                # ==========================================================================
                if self.train_flag:
                    torch.save(self.audio_images, './data/training_smart.pt')
                else:
                    torch.save(self.audio_images, './data/test_smart.pt')
        else:
            
            if args.multitask:
                if self.train_flag:
                    self.audio_images = torch.load('./data/training_smart_mtl.pt')
                    self.lung_class_nums = torch.load('./data/training_smart_mtl_lung_class_nums.pt')
                    self.disease_class_nums = torch.load('./data/training_smart_mtl_disease_class_nums.pt')
                else:
                    self.audio_images = torch.load('./data/test_smart_mtl.pt')
            
            elif args.multitask_domain:
                if self.train_flag:
                    self.audio_images = torch.load('./data/training_smart_mtld.pt')
                    self.lung_class_nums = torch.load('./data/training_smart_mtld_lung_class_nums.pt')
                    self.disease_class_nums = torch.load('./data/training_smart_mtld_disease_class_nums.pt')
                else:
                    self.audio_images = torch.load('./data/test_smart_mtld.pt')
            
            else:
                if self.train_flag:
                    self.audio_images = torch.load('./data/training_smart.pt')
                    self.class_nums = torch.load('./data/training_smart_class_nums.pt')
                else:
                    self.audio_images = torch.load('./data/test_smart.pt')
                
            # ==========================================================================

    def __getitem__(self, index):
        meta_label = None
        if self.args.multitask or self.args.multitask_domain:
            audio_images, lung_label, disease_label, domain_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2], self.audio_images[index][3]
        else:
            audio_images, label, meta_str = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2]
        
        if not self.args.multitask and not self.args.multitask_domain:
            if self.train_flag:
                meta_label = get_meta_infor(meta_str, self.args, len(meta_str))
            
        if self.args.raw_augment and self.train_flag and not self.mean_std:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        
        if self.args.multitask:
            return audio_image, (lung_label, disease_label)
        elif self.args.multitask_domain:
            return audio_image, (lung_label, disease_label, domain_label)
        else:
            if meta_label is None or not self.train_flag:
                return audio_image, label
            else:
                return audio_image, (label, meta_label)

    def __len__(self):
        return len(self.audio_images)