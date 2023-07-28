# DexCLR

Dwelling Extraction using Contrastive Representation Learning based on SimCLR

(Source: https://github.com/Spijkervet/SimCLR)

### Requirements 

Using GPU
```
pip install -r requirements.txt
```

Using CPU
```
pip install -r requirements_CPU.txt
```

### Define dataset path

Define the path to the dataset in `config/config.yaml`

### 1-Find best learning rate for pretraining and finetuning
The accuracy matrix computed by this function is necessary to run the next functions:
 - Pre-Training DexCLR ResNet encoder
 - Finetuning DexCLR ResNet
 - Training full supervised ResNet.

(Accuracy matrix saved to `results/accuracyMatrix_lr_IOU.csv`)
```
python find_Best_LR_Pretraining_Finetuning
```

### 2-Pre-Training DexCLR ResNet encoder:
Simply run the following to pre-train the dexCLR ResNet encoder using nt_xent loss function:
```
python dexCLR_Pretraining.py
```

### 3-Finetuning DexCLR ResNet:
Simply run the following to finetuning the dexCLR ResNet:
```
python dexCLR_finetuning_Nb_Labels.py
```

### 4-Training full supervised ResNet:
Simply run the following to train the dexCLR full supervised ResNet:
```
python full_Supervised_No_Pretraining.py
```

## Configuration
The configuration of training can be found in: `config/config.yaml`. I personally prefer to use files instead of long strings of arguments when configuring a run. An example `config.yaml` file:
```
# datasets
dataset_dir: "./datasets/dataset_Name/"
valid_ratio: 0.2
ext_imgs: [".tif", ".jpg"]

# train options
seed: 42 # sacred handles automatic seeding when passed in the config
batch_size: 8 #32 #128
image_size: 256
start_epoch: 0
epochs: 100
dataset: "CUSTOM" #"CIFAR10" # STL10 

# model options
projection_dim: 64 # "[...] to project the representation to a 64-dimensional latent space"
nb_class: 1  # number of classes
nb_channels: 3 # number of input chanells to the model
phase: "pretrain" # phase of Pretraining or Finetuning
lr: None # Learning Rate initialisation see optimizer

# loss options
optimizer: "Adam"
temperature: 0.5 # see appendix B.7.: Optimal temperature under different batch sizes

# reload options
model_path: "save" # set to the directory containing `checkpoint_##.tar
epoch_num: 100 # set to checkpoint number
load_freeze: False

# logistic regression options
logistic_batch_size: 256
logistic_epochs: 500
```
