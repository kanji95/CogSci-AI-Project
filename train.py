import os
import psutil
import gc
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.utils import print_, grad_check
from utilities.metrics import *

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
    total_pacc = 0
    
    cross_entropy_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss(reduction='sum')
    smoothl1_loss = nn.SmoothL1Loss(reduction='sum')
    cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=0.1)
    
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

            miner_label = word_label
            batch_size = fmri_scan.shape[0]
            one_hot = torch.zeros((batch_size, 180)).cuda(non_blocking=True)
            one_hot[torch.arange(batch_size), word_label] = 1
            word_label = one_hot

            target = torch.ones(batch_size).cuda(non_blocking=True)
            
        start_time = time()
        recon_out, reg_out, y_pred = brain_model(fmri_scan)

        # indices_tuple = miner_func(reg_out, miner_label)
        # loss = contrastive_loss(reg_out, miner_label, indices_tuple) + cosine_embedding_loss(reg_out, glove_emb, target) + smoothl1_loss(recon_out, fmri_scan)
        loss = bce_loss(y_pred, word_label) + cosine_embedding_loss(reg_out, glove_emb, target) + smoothl1_loss(recon_out, fmri_scan)

        loss.backward()
        if iterId % 500 == 0 and args.grad_check:
            grad_check(brain_model.named_parameters(), experiment)
            
        optimizer.step()
        brain_model.zero_grad()
        end_time = time()
        elapsed_time = end_time - start_time
        
        accuracy, accuracy_five, accuracy_ten = top_5(y_pred.detach().cpu().numpy(), word_label.detach().cpu().numpy())
        # pair_accuracy = evaluation(reg_out.detach().cpu().numpy(), glove_emb.detach().cpu().numpy())

        # print('Accuracy_top1: ' + str(accuracy) + ' Accuracy_top5: ' + str(accuracy_five) + 'Accuracy_top10: ' + str(accuracy_ten))
        
        total_acc += accuracy
        # total_pacc += pair_accuracy
        total_loss += float(loss.item())
        
        num_examples += batch_size
        
        if iterId % 20 == 0 and step != 0:
            gc.collect()
            # print(pred_label, word_label)
            memoryUse = py.memory_info()[0] / 2.0 ** 20
            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
            curr_loss = total_loss / (step + 1)
            curr_acc = total_acc / num_examples
            lr = optimizer.param_groups[0]["lr"]
            print(f'Accuracy_top1: {accuracy:.4f}, Accuracy_top5: {accuracy_five:.4f}, Accuracy_top10: {accuracy_ten:.4f}')
            print_(
                f"{timestamp} Epoch:[{epochId:2d}/{args.epochs:2d}] iter {iterId:6d} loss {curr_loss:.4f} acc {curr_acc:.4f} memory_use {memoryUse:.3f}MB lr {lr:.7f} elapsed {elapsed_time:.2f}"
            )
    epoch_end = time()
    epoch_time = epoch_end - epoch_start

    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

    train_loss = total_loss / data_len
    train_acc = total_acc / data_len
    train_pacc = total_pacc / data_len
    
    experiment.log({"loss": train_loss, "acc": train_acc})

    print_(
        f"{timestamp} FINISHED Epoch:{epochId:2d} loss {train_loss:.4f} acc {train_acc:.4f} pacc {train_pacc:.4f} elapsed {epoch_time:.2f}"
    )
    print_("============================================================================================================================================\n")
