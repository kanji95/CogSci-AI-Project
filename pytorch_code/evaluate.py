import os
import psutil
import gc
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.position_encoding import *

from utilities.utils import print_
from utilities.metrics import *

# from utilities.metrics import compute_mask_IOU

from pytorch_metric_learning import distances, losses, miners, reducers


@torch.no_grad()
def evaluate(
    val_loader,
    brain_model,
    epochId,
    args,
):

    brain_model.eval()

    pid = os.getpid()
    py = psutil.Process(pid)

    total_loss = 0
    total_acc = 0
    
    cross_entropy_loss = nn.CrossEntropyLoss()
    cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=0.1)
    
    distance = distances.CosineSimilarity()
    contrastive_loss = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    miner_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )

    data_len = len(val_loader)

    print_(
        "\n================================================= Evaluating only Grounding Network ======================================================="
    )

    num_examples = 0
    for step, batch in enumerate(val_loader):

        batch = (x.cuda(non_blocking=True) for x in batch)
        fmri_scan, glove_emb, word_label = batch

        batch_size = fmri_scan.shape[0]
        target = torch.ones(batch_size).cuda(non_blocking=True)

        start_time = time()

        y_pred = brain_model(fmri_scan)
        end_time = time()
        elapsed_time = end_time - start_time
        
        # indices_tuple = miner_func(reg_out, word_label)
        # loss = contrastive_loss(reg_out, word_label, indices_tuple) + cross_entropy_loss(y_pred, word_label) + cosine_embedding_loss(reg_out, glove_emb, target)
        loss = cross_entropy_loss(y_pred, word_label)

        total_loss += float(loss.item())

        # accuracy = (torch.argmax(y_pred) == word_label).sum()

        accuracy, accuracy_five, accuracy_ten = top_5(y_pred.detach().cpu().numpy(), word_label.detach().cpu().numpy())
        total_acc += accuracy


        
        num_examples += batch_size

        if step % 200 == 0:
            gc.collect()
            memoryUse = py.memory_info()[0] / 2.0 ** 20

            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

            curr_acc = total_acc / num_examples

            print(f'Accuracy_top1: {accuracy:.4f}, Accuracy_top5: {accuracy_five:.4f}, Accuracy_top10: {accuracy_ten:.4f}')
            print_(
                f"{timestamp} Validation: iter [{step:3d}/{data_len}] curr_acc {curr_acc:.4f} memory_use {memoryUse:.3f}MB elapsed {elapsed_time:.2f}"
            )
            
    val_acc = total_acc / num_examples
    val_loss = total_loss / data_len

    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")
    print_(
        f"{timestamp} Validation: EpochId: {epochId:2d} val_acc {val_acc:.4f}"
    )
    print_("============================================================================================================================================\n")
    
    return val_loss, val_acc
