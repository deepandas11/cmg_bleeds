import time
import torch
# import numpy as np
# import math
# from statistics import mean
# import torch.utils.data as data
from tensorboardX import SummaryWriter
from torch.autograd import Variable

writer = SummaryWriter('logs/')

from utils.utils import AverageMeter


_mse_loss = torch.nn.MSELoss()
_criterion = torch.nn.BCELoss()  

def train(data_loader_train, encoder, decoder, optimizer, epoch,
          use_gpu=False, start_loss=0.0):

    losses = AverageMeter()
    total_loss = start_loss

    # Randomly Shuffled Dataset Indices
    indices = data_loader_train.get_indices()
    total_steps = len(indices)
    # total_steps = 5

    # Training with Batch Size 1
    for index in range(total_steps):
        # Fetch actual data index
        d_index = indices[index]

        encoder.train()
        decoder.train()

        img_sequence, label = data_loader_train[d_index]

        if torch.cuda.is_available() and use_gpu:
            img_sequence = img_sequence.cuda()
            label = label.cuda()

        img_sequence, label = map(Variable, (img_sequence, label))

        seq_op = encoder(img_sequence)
        pred = decoder(seq_op)
        # loss = _mse_loss(pred, label)
        loss = _criterion(pred, label)

        optimizer.zero_grad()
        total_loss += loss.data
        loss.backward()
        optimizer.step()
        # print("000000", loss, loss.shape)
        losses.update(loss.item())
        niter = epoch * total_steps + index
        writer.add_scalar('data/training_loss', losses.val, niter)

        print("Step: %d, Current Loss: %0.4f, Average Loss: %0.4f" %
              (index, loss, total_loss))

    return total_loss/total_steps



def validate(encoder, decoder, data_loader_val, epoch, use_gpu):

    val_losses = AverageMeter()
    total_val_loss = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()

    indices = data_loader_val.get_indices()
    total_steps = len(indices)
    # total_steps = 5

    # Loss Scores Record
    loss_scores = list()
    for index in range(total_steps):
        d_index = indices[index]

        img_sequence, label = data_loader_val[d_index]

        img_sequence = img_sequence.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            seq_op = encoder(img_sequence)
            pred = decoder(seq_op)

            loss = _mse_loss(pred, label)
            loss_scores.append(loss)
            total_val_loss += loss.data

        val_losses.update(loss.item())
        niter =  epoch * total_steps + index
        writer.add_scalar('data/validation_loss', val_losses.val, niter)

    return total_val_loss / total_steps





