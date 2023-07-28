import os
import numpy as np
from PIL import Image
import torch
# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# DexCLR / DeepLab
# Added model
from dexclr.deepLab import DeepLab

from dexclr.modules import NT_Xent
from dexclr.modules.contrastiveLoss import ContrastiveLoss
from dexclr.modules.diceLoss import BinaryDiceLoss
from dexclr.modules.transformations import TransformsDexCLR

# Dataset
from dexclr.modules.camp_dataset import CampDataset

from dexclr.model import load_optimizer, save_model

from torchmetrics.classification import BinaryJaccardIndex
from torch.nn import BCELoss

from dexclr.modules.early_stopper import EarlyStopper

def train(args, train_loader, model, criterion, optimizer, writer):
    """
    Function to perform the training of the model.

    Args:
        args (argparse.Namespace): Parsed arguments.
        train_loader (torch.utils.data.DataLoader): DataLoader for training dataset.
        model: The model to train.
        criterion: The loss function.
        optimizer: The optimizer.
        writer: SummaryWriter for TensorBoard logging.

    Returns:
        float: Total loss for the epoch.
    """
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.device.type == "cpu":
            x_i = x_i.cpu()
            x_j = x_j.cpu()
        else:
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()
        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


### Function finetuning/final training ###########
def finetune(args, train_loader, model, optimizer, writer, ls_fn, epoch=None):
    """
    Function to perform fine-tuning or final training of the model.

    Args:
        args (argparse.Namespace): Parsed arguments.
        train_loader (torch.utils.data.DataLoader): DataLoader for training dataset.
        model: The model to train.
        optimizer: The optimizer.
        writer: SummaryWriter for TensorBoard logging.
        ls_fn: The loss function.
        epoch (int, optional): Current epoch. Defaults to None.

    Returns:
        float: Total loss for the epoch.
    """
    if epoch == args.epochs//2:  # To relax the requirement for updating entire network weights after 50 epochs
        for param in model.net.backbone.parameters():
            param.requires_grad = True

    loss_epoch = 0

    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = ls_fn(output.sigmoid(), y.float())
        loss.backward()

        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step+1}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        args.global_step += 1

        loss_epoch += loss.item()

    return loss_epoch

######################################################################################################################
def test(args, test_loader, model):
    """
    Function to perform testing of the model.

    Args:
        args (argparse.Namespace): Parsed arguments.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing dataset.
        model: The model to test.

    Returns:
        float: Average accuracy for the testing dataset.
    """
    accuracy_epoch = 0
    jaccard = BinaryJaccardIndex(treshold=0.5).to(args.device)
    model.eval()
    total_steps = len(test_loader)
    
    for step, (x, y) in enumerate(test_loader):
        x = x.to(args.device)
        y = y.to(args.device)

        with torch.no_grad():
            output = model(x)
            predicted = torch.sigmoid(output)
            accu = jaccard(predicted, y)
            accuracy_epoch += accu.item()
        
        print(f'step: {step}, accuracy: {accu.item()}')
        
        # Function to save prediction results as png files
        save_prediction(args, step, predicted, y)
    
    accuracy_epoch /= total_steps  # Mean accuracy
    return accuracy_epoch

######################################################################################################################
def save_prediction(args, step, predicted, label):
    """
    Function to save prediction results as PNG files.

    Args:
        step (int): Step number.
        predicted: Predicted output.
        label: Ground truth labels.
    """
    try:
        os.makedirs(args.root_path + '/predict', exist_ok=True)
        os.makedirs(args.root_path + '/label', exist_ok=True)
    except:
        pass

    for i in range(predicted.shape[0]):
        img = predicted[i].cpu().detach().numpy().reshape(256, 256)
        imgL = label[i].cpu().detach().numpy().reshape(256, 256)
        
        # Convert probability map to hard binary classes
        img = np.where(img >= 0.5, 1, 0)
        
        
        # Convert label and Images to uint8 to be sure
        im_predict = Image.fromarray(img.astype(np.uint8))
        imgLabel = Image.fromarray(imgL.astype(np.uint8))

        im_predict.save(args.root_path + '/predict/' + str(step) + "_" + str(i + 1) + ".tif")
        imgLabel.save(args.root_path + '/label/' + str(step) + "_" + str(i + 1) + ".tif")
 
######################################################################################################################
def DexCLR(args):
    """
    Main function to train and test the model.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "CUSTOM":
        if args.phase == 'pretrain':
            train_dataset = CampDataset(root=args.dataset_dir+"images", transform=TransformsDexCLR(size=args.image_size), extImgs=args.ext_imgs)
        else:
            train_dataset = CampDataset(root=args.dataset_dir+"images", labels=args.dataset_dir+"labels", valid_ratio=args.valid_ratio, transform=TransformsDexCLR(size=args.image_size).test_transform, extImgs=args.ext_imgs)
            test_dataset = CampDataset(root=args.dataset_dir+"images", labels=args.dataset_dir+"labels", valid_ratio=args.valid_ratio, testPhase=True, transform=TransformsDexCLR(size=args.image_size).test_transform, extImgs=args.ext_imgs)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.phase != 'pretrain':
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Initialize model
    model = DeepLab(nb_class=args.nb_class, nb_channels=args.nb_channels, n_features=2048, projection_dim=args.projection_dim, phase=args.phase)
    if args.load_freeze:
        print('Pretrained model weights will be loaded...')
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.epoch_num))
        model.net.backbone.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        for param in model.net.backbone.parameters():
            param.requires_grad = False

    model = model.to(args.device)

    # optimizer / loss
    optimizer = load_optimizer(args, model)
    if args.phase == 'pretrain':
        #criterion = NT_Xent(args.batch_size, args.temperature)
        criterion = ContrastiveLoss(args.batch_size, args.temperature).to(args.device)

    writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    ls_fn = BCELoss().to(args.device)
    #ls_fn = BinaryDiceLoss().to(args.device)

    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]

        if args.phase == 'pretrain':
            loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)
        else:
            loss_epoch = finetune(args, train_loader, model, optimizer, writer, ls_fn=ls_fn, epoch=epoch)

        # Save model in different folder depending on the phase
        args.model_path = os.path.join(args.root_path, str(args.phase))
        print(args.model_path)

        if epoch % 10 == 0:
            save_model(args, model, optimizer)

        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}")
        args.current_epoch += 1

    if args.phase != 'pretrain':
        if len(args.phase.split("_")) == 1:
            accuracy_epoch = test(args, test_loader, model)
            args.accuracy_Matrix['LR'].append(args.nameModel)
            args.accuracy_Matrix['IOU'].append(accuracy_epoch)
        elif len(args.phase.split("_")) > 1:
            args.root_path = os.path.join(args.root_path, args.phase)
            accuracy_epoch = test(args, test_loader, model)
            args.accuracy_Matrix['nbLabels'].append(args.phase.split("_")[-1])
            args.accuracy_Matrix['IOU'].append(accuracy_epoch)
        else:
            print("Error format exp on nbLabels finetuning !!!")

        print(f"[FINAL]\t Accuracy: {accuracy_epoch}")

    save_model(args, model, optimizer)