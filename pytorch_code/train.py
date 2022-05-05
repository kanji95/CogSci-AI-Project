import os
import psutil
import gc
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.utils import print_, grad_check

from pytorch_metric_learning import distances, losses, miners, reducers
# from utilities.metrics import compute_mask_IOU


def train(
    train_loader,
    brain_model,
    optimizer,
    experiment,
    epochId,
    args,
):

    pid = os.getpid()
    py = psutil.Process(pid)

    brain_model.train()

    optimizer.zero_grad()

    total_loss = 0
    total_acc = 0
    
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    distance = distances.CosineSimilarity()
    contrastive_loss = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    
    data_len = len(train_loader)

    epoch_start = time()

    print_("\n=========================================================== Training Network ===================================================")
    
    num_examples = 0
    for step, batch in enumerate(train_loader):
        iterId = step + (epochId * data_len) - 1
        with torch.no_grad():
            batch = (x.cuda(non_blocking=True) for x in batch)
            fmri_scan, glove_emb, word_label = batch
            
            batch_size = fmri_scan.shape[0]
            
        start_time = time()
    
        reg_out, y_pred = brain_model(fmri_scan)
        
        indices_tuple = miner_func(reg_out, word_label)
        loss = contrastive_loss(reg_out, glove_emb, indices_tuple) + cross_entropy_loss(y_pred, word_label)
        
        loss.backward()
        if iterId % 500 == 0 and args.grad_check:
            grad_check(brain_model.named_parameters(), experiment)
            
        optimizer.step()
        brain_model.zero_grad()
        end_time = time()
        elapsed_time = end_time - start_time
        
        accuracy = (torch.argmax(y_pred) == word_label).sum()
        
        total_acc = accuracy.item()
        total_loss += float(loss.item())
        
        num_examples += batch_size
        
        if iterId % 200 == 0 and step != 0:
            gc.collect()
            memoryUse = py.memory_info()[0] / 2.0 ** 20
            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
            curr_loss = total_loss / (step + 1)
            curr_acc = total_acc / num_examples
            lr = optimizer.param_groups[0]["lr"]
            print_(
                f"{timestamp} Epoch:[{epochId:2d}/{args.epochs:2d}] iter {iterId:6d} loss {curr_loss:.4f} acc {curr_acc:.4f} memory_use {memoryUse:.3f}MB lr {lr:.7f} elapsed {elapsed_time:.2f}"
            )
    epoch_end = time()
    epoch_time = epoch_end - epoch_start

    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

    train_loss = total_loss / data_len
    train_acc = total_acc / num_examples

    experiment.log({"loss": train_loss, "acc": train_acc})

    print_(
        f"{timestamp} FINISHED Epoch:{epochId:2d} loss {train_loss:.4f} acc {train_acc:.4f} elapsed {epoch_time:.2f}"
    )
    print_("============================================================================================================================================\n")
