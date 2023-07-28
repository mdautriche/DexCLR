import pandas as pd
import os
from dexclr.DexCLR import DexCLR
from dexclr.utils import yaml_config_hook

import torch
import argparse

##########################################################################################

# Experimentation: DexCLR pretraining
# Expeiments on learning rate of pretraining and finetuning
# Using 20% labels during finutuning

def find_Best_LR_Pretraining_Finetuning(args):
    pretrainLr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    finetuneLr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    args.accuracy_Matrix = {'LR':[], 'IOU':[]}

    for i, lrPre in enumerate(pretrainLr):
        for k, lrFine in enumerate(finetuneLr):
            # Reset param for new experiment
            args.model_path = "save"
            args.phase = "pretrain"
            args.load_freeze = False

            # Pretraining with custom Image
            args.nameModel = str("%.5f" % lrPre).rstrip(
                '0').rstrip('.')+"_"+str("%.5f" % lrFine).rstrip(
                '0').rstrip('.')
            args.model_path = args.model_path+"/Exp_"+str(args.nameModel)+"/"
            args.root_path = args.model_path
            # Set learning rate pretraining
            args.lr = lrPre
            DexCLR(args)

            # Finetuning
            # Change the phase for the fintuning step
            args.phase = "finetuning"
            # Freeze the backbone
            args.load_freeze = True
            # Set learning rate finetuning
            args.lr = lrFine
            # Finetuning phase
            DexCLR(args)

            pd.DataFrame.from_dict(args.accuracy_Matrix).to_csv("save/accuracyMatrix_lr_IOU.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DexCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    find_Best_LR_Pretraining_Finetuning(args)