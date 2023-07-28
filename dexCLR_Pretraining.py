import pandas as pd
import os
from dexclr.DexCLR import DexCLR
from dexclr.utils import yaml_config_hook

import torch
import argparse
from dexclr.utils.utils_Results import best_lr, read_Matrix

def dexCLR_Pretraining(args):
    # Reset param for new experiment
    args.model_path = "save"
    args.phase = "pretrain"
    args.load_freeze = False

    read_accuracy_Matrix = read_Matrix("results/accuracyMatrix_lr_IOU.csv")
    args.nameModel = best_lr(read_accuracy_Matrix)
    lrPre = float(args.nameModel.split('_')[0])
    lrFine = float(args.nameModel.split('_')[1])

    # Pretraining with custom Image
    args.nameModel = str("%.5f" % lrPre).rstrip(
                    '0').rstrip('.')+"_"+str("%.5f" % lrFine).rstrip(
                    '0').rstrip('.')
    
    args.model_path = args.model_path+"/Exp_"+str(args.nameModel)+"/"
    args.root_path = args.model_path
    
    # Set learning rate pretraining
    args.lr = lrPre
    DexCLR(args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DexCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dexCLR_Pretraining(args)