import wandb

import argparse
import gc
import os
import random
import traceback
from datetime import datetime
from time import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import *
from torchvision.models._utils import IntermediateLayerGetter

from dataloader import FmriDataset

from evaluate import evaluate
# from losses import Loss
from models.model import Baseline, ROIBaseline
from train import train
from utilities.utils import print_

plt.rcParams["figure.figsize"] = (15, 15)

torch.manual_seed(12)
torch.cuda.manual_seed(12)
random.seed(12)
np.random.seed(12)

torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()

def get_args_parser():
    parser = argparse.ArgumentParser("Cogsci Project", add_help=False)

    # HYPER Params
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--gamma", default=0.7, type=float)
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--grad_check", default=False, action="store_true")

    ## DCRF
    parser.add_argument("--dcrf", default=False, action="store_true")

    # MODEL Params
    parser.add_argument("--model_dir", type=str, default="./saved_model")
    parser.add_argument("--save", default=False, action="store_true")

    ## Evalute??
    parser.add_argument("--model_filename", default="model_unc.pth", type=str)

    # LOSS Params
    parser.add_argument("--run_name", default="", type=str)

    parser.add_argument(
        "--dataroot", type=str, default="<data_path>"
    )

    return parser


def main(args):

    experiment = wandb.init(project="cogsci_project", config=args)
    if args.run_name == "":
        print_("No Name Provided, Using Default Run Name")
        args.run_name = f"{experiment.id}"
    print_(f"METHOD USED FOR CURRENT RUN {args.run_name}")
    experiment.name = args.run_name
    wandb.run.save()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print_(f"{device} being used with {n_gpu} GPUs!!")

    ####################### Model Initialization #######################

    # brain_model = Baseline()
    brain_model = ROIBaseline()

    wandb.watch(brain_model, log="all")

    total_parameters = 0
    for name, child in brain_model.named_children():
        num_params = sum([p.numel() for p in child.parameters() if p.requires_grad])
        if num_params > 0:
            print_(f"No. of params in {name}: {num_params}")
            total_parameters += num_params

    print_(f"Total number of params: {total_parameters}")

    if n_gpu > 1:
        brain_model = nn.DataParallel(brain_model)

    brain_model.to(device)

    params = list([p for p in brain_model.parameters()])

    optimizer = AdamW(params, lr=args.lr)

    save_path = args.model_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_filename = os.path.join(
        save_path,
        f'baseline_{datetime.now().strftime("%d_%b_%H-%M")}.pth',
    )

    ######################## Dataset Loading ########################
    print_("Initializing dataset")
    start = time()

    train_dataset = FmriDataset(split="train") # Brain2word Dataset
    val_dataset = FmriDataset(split="test") # Brain2word Dataset

    end = time()
    elapsed = end - start
    print_(f"Elapsed time for loading dataset is {elapsed}sec")

    start = time()

    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    end = time()
    elapsed = end - start
    print_(f"Elapsed time for loading dataloader is {elapsed}sec")

    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.gamma,
        patience=2,
        threshold=1e-3,
        min_lr=1e-8,
        verbose=True,
    )

    num_iter = len(train_loader)
    print_(f"training iterations {num_iter}")

    print_(
        f"===================== SAVING MODEL TO FILE {model_filename}! ====================="
    )

    best_acc = 0
    epochs_without_improvement = 0

    for epochId in range(args.epochs):

        train(train_loader, brain_model, 
                optimizer, experiment, epochId,
                args)

        val_loss, val_acc = evaluate(val_loader, brain_model,
                                        epochId, args)

        wandb.log({"val_loss": val_loss, "val_IOU": val_acc})

        lr_scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            
            print_(f"Saving Checkpoint at epoch {epochId}, best validation accuracy is {best_acc}!")
            if args.save:
                torch.save(
                    {
                        "epoch": epochId,
                        "state_dict": brain_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    model_filename,
                )
            epochs_without_improvement = 0
        elif val_acc <= best_acc and epochId != args.epochs - 1:
            epochs_without_improvement += 1
            print_(f"Epochs without Improvement: {epochs_without_improvement}")

            if epochs_without_improvement == 100:
                print_(
                    f"{epochs_without_improvement} epochs without improvement, Stopping Training!"
                )
                break         
    
    if args.save:
        print_(f"Current Run Name {args.run_name}")
        best_acc_filename = os.path.join(
            save_path,
            f"baseline_{best_acc:.5f}.pth",
        )
        os.rename(model_filename, best_acc_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Referring Image Segmentation", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    print_(args)

    try:
        main(args)
    except Exception as e:
        traceback.print_exc()
