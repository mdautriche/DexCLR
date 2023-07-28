import pandas as pd
from dexclr.utils.utils_Results import best_lr, read_Matrix
import pandas as pd
import os
from dexclr.DexCLR import DexCLR
from dexclr.utils import yaml_config_hook

import torch
import argparse

##########################################################################################

# Experimentation: Use weights from best learning rate pretraining and tune nb labels in finetuning

def dexCLR_finetuning_Nb_Labels(args):
    valid_ratio_list = [x / 100.0 for x in range(10, 100, 5)]

    args.accuracy_Matrix = {'nbLabels':[], 'IOU':[]}

    for ratio in valid_ratio_list:
        read_accuracy_Matrix = read_Matrix("results/accuracyMatrix_lr_IOU.csv")
        args.model_path = "save"
        # Load retraining with custom Image
        args.nameModel = best_lr(read_accuracy_Matrix)
        args.model_path = args.model_path+"/Exp_"+str(args.nameModel)+"/pretrain/"
        # Finetuning
        # Change the phase for the fintuning step
        args.phase = "finetuning_"+str(ratio)
        args.root_path = args.model_path[:-9]
        # Freeze the backbone
        args.load_freeze = True
        # Set learning rate finetuning
        args.lr = float(args.nameModel.split('_')[1])
        # Set valid ratio
        args.valid_ratio = ratio
        # Finetuning phase
        DexCLR(args)

        pd.DataFrame.from_dict(args.accuracy_Matrix).to_csv("save/accuracyMatrix_label_finetuning.csv")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DexCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dexCLR_finetuning_Nb_Labels(args)