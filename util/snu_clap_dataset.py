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

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio_snubh, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio

from transformers import ClapProcessor
Processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", sampling_rate=48000)

class CLAPSNUBHDataset(Dataset):
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
        
        
        if self.train_flag and os.path.isfile('./data/training_snubh_class_nums.pt'):
            self.class_nums = torch.load('./data/training_snubh_class_nums.pt')
        
        cache_path = './data/training_snubh_clap.pt' if self.train_flag else './data/test_snubh_clap.pt'
        
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
            audio_image_array = []
            label_array = []
            meta_array = []
            
            for idx, filename in enumerate(data_path):
                sample_data = get_individual_cycles_torchaudio_snubh(args, filename, self.file_to_metadata[filename], self.data_folder, filename, self.sample_rate, args.n_cls)
                for samples in sample_data:
                    data1, data2, data3 = samples[0], samples[1], samples[2] # audio_image, label
                    audio_image_array.append(data1.squeeze(0).numpy())
                    label_array.append(data2)
                    meta_array.append(data3)
                
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
                
            if self.train_flag:
                class_nums_cache_path = './data/training_snubh_class_nums.pt'
                if not os.path.isfile(class_nums_cache_path):
                    torch.save(self.class_nums, class_nums_cache_path)
            
            # ==========================================================================
            self.audio_images = []
            inputs = Processor(audios=audio_image_array, return_tensors="pt")
            audio_inputs = inputs["input_features"]
            for audio, label, meta in zip(audio_inputs, label_array, meta_array):
                self.audio_images.append((audio, label, meta))
            if self.train_flag:
                torch.save(self.audio_images, './data/training_snubh_clap.pt')
            else:
                torch.save(self.audio_images, './data/test_snubh_clap.pt')
            # ==========================================================================
        else:
            if self.train_flag:
                self.audio_images = torch.load('./data/training_snubh_clap.pt')
            else:
                self.audio_images = torch.load('./data/test_snubh_clap.pt')
            

    def __getitem__(self, index):
        audio_image, label, meta_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2]
        if meta_label is None or not self.train_flag:
            return audio_image, label
        else:
            return audio_image, (label, meta_label)

    def __len__(self):
        return len(self.audio_images)