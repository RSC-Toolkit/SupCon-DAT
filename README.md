# SupCon-DAT
Official Implementation of "Evaluating Generalization Strategies for Respiratory Sound Classification Across Heterogeneous Clinical Datasets: Algorithm Development and Multi-Cohort Validation"


## Requirements
Install the necessary packages with: 
```
$ pip install torch torchvision torchaudio
$ pip install -r requirements.txt
```
For the reproducibility, we used `torch=2.0.7` and `torchaudio=2.0.`

## Data Preparation
Download the ICBHI dataset files from [official_page](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge).     
```bash
$ wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
```
All `*.wav` and `*.txt` should be saved in `data/icbhi_dataset/audio_test_data`.     

Note that ICBHI dataset consists of a total of 6,898 respiratory cycles, 
of which 1,864 contain crackles, 886 contain wheezes, and 506 contain both crackles and wheezes, in 920 annotated audio samples from 126 subjects.

Note that SNUBH and SMART datasets are our in-house dataset. So please do experiment with ICBHI first.

## Training 
To simply train the model, run the shell files in `scripts/`.    
1. **`scripts/icbhi_ce.sh`**: Cross-Entropy loss with AST model.
2. **`scripts/icbhi_dat_device.sh`**: Cross-Entropy loss with Domain Adaptation (DANN) in terms of Device (stethoscope) with AST Model. 
3. **`scripts/icbhi_dat2_accum16_device.sh`**: Cross-Entropy loss with SupCon-DAT (w/ recording device attribute) on AST model.

Important arguments for different data settings.
- `--dataset`: other lungsound datasets or heart sound can be implemented.
- `--n_cls`: set number of classes as 4 or 2 (normal / abnormal) for lungsound classification.
- `--test_fold`: "official" denotes 60/40% train/test split, and "0"~"4" denote 80/20% split.
- **`--domain_adaptation`**: Using the proposed `DAT` in this paper.
- **`--domain_adaptation2`**: Using the proposed `SCL` in this paper.
- **`--meta_mode`**: meta information for cross-domain; choices=`['none', 'age', 'sex', 'loc', 'dev', 'label']`. The default is `dev`.

Important arguments for models.
- `--model`: network architecture, see [models](models/).
- `--from_sl_official`: load ImageNet pretrained checkpoint.
- `--audioset_pretrained`: load AudioSet pretrained checkpoint and only support AST and SSAST.

Important argument for evaluation.
- `--eval`: switch mode to evaluation without any training.
- `--pretrained`: load pretrained checkpoint and require `pretrained_ckpt` argument.
- `--pretrained_ckpt`: path for the pretrained checkpoint.

The pretrained model checkpoints will be saved at `save/[EXP_NAME]/best.pth`.     

Our submitted paper is now under review. This repository will be updated after paper is accepted.