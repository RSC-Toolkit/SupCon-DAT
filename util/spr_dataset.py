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
from PIL import Image

from .icbhi_util import get_individual_cycles_torchaudio, cut_pad_sample_torchaudio
from .icbhi_util import get_annotations, get_individual_cycles_torchaudio, generate_fbank
from sklearn.model_selection import train_test_split


import os
import torch
from torch.utils.data import Dataset

import torchaudio
from torchaudio import transforms as T

class SPRSoundDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False):
        
        self.train_flag = train_flag
        self.split = 'train' if train_flag else 'test'
        self.transform = transform
        self.args = args
        self.mean_std = mean_std
        self.sample_rate = args.sample_rate
        self.n_mels = args.n_mels
        
        if os.path.isfile('./data/spr_test.pt'):
            self.audio_images = torch.load('./data/spr_test.pt')
        else:
            train_data_folder = './data/SPRSound/Classification/train_classification_json/'
            train_filenames = sorted(glob(train_data_folder+'/*'))
            train_filenames = set(f for f in train_filenames if '.json' in f)
            train_filenames = sorted(list(set(train_filenames)))
            
            print(len(train_filenames))
    
            valid_data_folder = './data/SPRSound/Classification/valid_classification_json/2022/inter_test_json/'
            valid_filenames = sorted(glob(valid_data_folder+'/*'))
            valid_filenames = set(f for f in valid_filenames if '.json' in f)
            valid_filenames = sorted(list(set(valid_filenames)))
            
            print(len(valid_filenames))
    
            annotation_list = list()
    
            import json
    
            for filename in train_filenames:
                with open (filename, "r") as f:
                    data = json.load(f)
    
                filename_tokens = os.path.basename(filename).split('_')
    
                data_dict = {
                    'Patient Number': filename_tokens[0], 
                    'Age': filename_tokens[1], 
                    'Gender': filename_tokens[2], 
                    'Recording location': filename_tokens[3],
                    'Recording number': filename_tokens[4].split('.')[0]
                }
    
                data_dict.update(data)
                annotation_list.append(data_dict)
    
            flattened_rows = []
            for record in annotation_list:
                base_info = {k: v for k, v in record.items() if k != 'event_annotation'}
                for event in record['event_annotation']:
                    combined = {**base_info, **event}
                    flattened_rows.append(combined)
    
            final_train_meta = pd.DataFrame(flattened_rows)
            final_train_meta['sort'] = 'train'
    
            annotation_list = list()
    
            for filename in valid_filenames:
                with open (filename, "r") as f:
                    data = json.load(f)
    
                # filename_tokens = filename.split('\\')[1].split('_')
                filename_tokens = os.path.basename(filename).split('_')
    
                data_dict = {
                    'Patient Number': filename_tokens[0], 
                    'Age': filename_tokens[1], 
                    'Gender': filename_tokens[2], 
                    'Recording location': filename_tokens[3],
                    'Recording number': filename_tokens[4].split('.')[0]
                }
    
                data_dict.update(data)
                annotation_list.append(data_dict)
    
            flattened_rows = []
            for record in annotation_list:
                base_info = {k: v for k, v in record.items() if k != 'event_annotation'}
                for event in record['event_annotation']:
                    combined = {**base_info, **event}
                    flattened_rows.append(combined)
    
            final_valid_meta = pd.DataFrame(flattened_rows)
            final_valid_meta['sort'] = 'valid'
    
            def map_annotation_type(annotation):
                ann = annotation.strip().lower()
                
                if ann == 'normal':
                    return 'Normal'
                elif ann == 'wheeze':
                    return 'Wheeze'
                elif ann in ['coarse crackle', 'fine crackle']:
                    return 'Crackle'
                elif ann == 'wheeze + crackle':
                    return 'Both'
                elif ann in ['rhonchi', 'stridor']:
                    return None
                else:
                    return annotation
    
            final_meta_data = pd.concat([final_train_meta, final_valid_meta], axis=0)
    
            final_meta_data['start'] = pd.to_numeric(final_meta_data['start']) * 0.001
            final_meta_data['end'] = pd.to_numeric(final_meta_data['end']) * 0.001
    
            final_meta_data['final_type'] = final_meta_data['type'].apply(map_annotation_type)
            final_meta_data = final_meta_data[final_meta_data['final_type'].notnull()]
    
            final_meta_data = final_meta_data[[
                'Patient Number', 
                'Recording number', 
                'Age', 
                'Gender', 
                'Recording location',
                'start', 
                'end', 
                'type', 
                'record_annotation', 
                'final_type',
                'sort'
            ]]
    
            audio_data = []  # each sample is a tuple with (audio_data, label, filename)
            labels = []
    
            cycle_list = []
            filename_to_label = {}
            classwise_cycle_list = [[] for _ in range(4)]
    
            def cut_pad_sample_torchaudio(data):
    
                sample_rate = 16000
                fade_samples_ratio = 16
                desired_length = 8
                fade_samples = int(sample_rate / fade_samples_ratio)
                fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
                target_duration = desired_length * sample_rate
    
                if data.shape[-1] > target_duration:
                    data = data[..., :target_duration]
                    if data.dim() == 1:
                        data = data.unsqueeze(0)
                else:
                    import math
                    ratio = math.ceil(target_duration / data.shape[-1])
                    data = data.repeat(1, ratio)
                    data = data[..., :target_duration]
                    data = fade_out(data)
                
                return data
    
            def _slice_data_torchaudio(start, end, data, sample_rate):
                """
                SCL paper..
                sample_rate denotes how many sample points for one second
                """
                max_ind = data.shape[1]
                start_ind = min(int(start * sample_rate), max_ind)
                end_ind = min(int(end * sample_rate), max_ind)
    
                return data[:, start_ind: end_ind]
    
            def _get_lungsound_label(ann):
                
                if ann == 'Normal':
                    return 0
                elif ann == 'Crackle':
                    return 0
                elif ann == 'Wheeze': 
                    return 1
                elif ann == 'Wheeze+Crackle':
                    return 1
    
            def get_individual_cycles_torchaudio(meta_df, filename, sample_rate, n_cls):
    
                sample_data = []
    
                data, sr = torchaudio.load(filename)
                
                if sr != sample_rate:
                    resample = T.Resample(sr, sample_rate)
                    data = resample(data)
                
                fade_samples_ratio = 16 # 50
                fade_samples = int(sample_rate / fade_samples_ratio)
                fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
                data = fade(data)
    
                # filename = filename.split('\\')[1]
                filename = os.path.basename(filename)
    
                patient_id, age, gender, location, rec_number = filename.replace('.wav', '').split('_')
    
                matching_rows = meta_df[
                    (meta_df['Patient Number'] == patient_id) &
                    (meta_df['Age'] == age) &
                    (meta_df['Gender'] == gender) &
                    (meta_df['Recording location'] == location) &
                    (meta_df['Recording number'] == rec_number)
                ].reset_index(drop=True)
                
                for idx in matching_rows.index:
                    row = matching_rows.loc[idx]
    
                    start = row['start']
                    end = row['end']
                    audio_chunk = _slice_data_torchaudio(start, end, data, sample_rate)
    
                    label = _get_lungsound_label(matching_rows['final_type'][idx])
    
                    sample_data.append((audio_chunk, label))
                    
                padded_sample_data = []
                for data, label in sample_data:
                    data = cut_pad_sample_torchaudio(data)
                    padded_sample_data.append((data, label))
                    
                return padded_sample_data
            
            valid_data_folder = './data/SPRSound/Classification/valid_classification_wav/2022/'
            valid_filenames = sorted(glob(valid_data_folder+'/*'))
            valid_filenames = set(f for f in valid_filenames if '.wav' in f)
            valid_filenames = sorted(list(set(valid_filenames)))
    
            filenames = valid_filenames
    
            # ==========================================================================
            """ extract individual cycles by librosa or torchaudio """
            print(len(filenames))
            for idx, filename in enumerate(filenames):
                # you can use self.filename_to_label to get statistics of original sample labels (will not be used on other function)
                filename_to_label[filename] = []
    
                # "RespireNet" version: get original cycles 6,898 by librosa
                # sample_data = get_individual_cycles_librosa(args, annotation_dict[filename], data_folder, filename, args.sample_rate, args.n_cls, args.butterworth_filter)
    
                # "SCL" version: get original cycles 6,898 by torchaudio and cut_pad samples
                sample_data = get_individual_cycles_torchaudio(final_meta_data, filename, 16000, 4)
                # cycles_with_labels: [(audio_chunk, label, metadata), (...)]
                cycles_with_labels = [(data[0], data[1]) for data in sample_data]
    
                cycle_list.extend(cycles_with_labels)
                for d in cycles_with_labels:
                    # {filename: [label for cycle 1, ...]}
                    filename_to_label[filename].append(d[1])
                    classwise_cycle_list[d[1]].append(d)
    
            for sample in cycle_list:
                audio_data.append(sample)
    
            class_nums = np.zeros(4)
            for sample in audio_data:
                class_nums[sample[1]] += 1
                labels.append(sample[1])
            class_ratio = class_nums / sum(class_nums) * 100
    
            print('total number of audio data: {}'.format(len(audio_data))) 
    
            # ==========================================================================
            """ convert mel-spectrogram """
            self.audio_images = []
            for index in range(len(audio_data)):
                audio, label = audio_data[index][0], audio_data[index][1] # Metalabel
    
                audio_image = []
                # self.aug_times = 1 + 5 * self.args.augment_times  # original + five naa augmentations * augment_times (optional)
                for aug_idx in range(args.raw_augment+1): 
                    if aug_idx > 0:
                        if self.train_flag:
                            audio = augment_raw_audio(np.asarray(audio.squeeze(0)), self.sample_rate, self.args)
                            audio = cut_pad_sample_torchaudio(torch.tensor(audio), args)                
                        
                        image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                        audio_image.append(image)                                        
                    else:
                        image = generate_fbank(args, audio, self.sample_rate, n_mels=self.n_mels)
                        audio_image.append(image)
                self.audio_images.append((audio_image, label))
            print(len(self.audio_images))
            torch.save(self.audio_images, './data/spr_test.pt')

    def __getitem__(self, index):
        audio_images, label= self.audio_images[index][0], self.audio_images[index][1]
        
        audio_image = audio_images[0]
        audio_image = self.transform(audio_image)
            
        return audio_image, label
        
    def __len__(self):
        return len(self.audio_images)
