import torch
import os
import shutil
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum = sum(self.vals)
        self.count = len(self.vals)
        self.avg = self.sum/self.count


def adjust_learning_rate(base_lr, optimizer, epoch, lr_decay=0.2):
    """Adjusts learning rate based on epoch of training

    At 60th, 120th and 150th epoch, divide learning rate by 0.2

    Args:
        base_lr: starting learning rate
        optimizer: optimizer used in training, SGD usually
        epoch: current epoch
        lr_decay: decay ratio (default: {0.2})
    """
    if epoch < 30:
        lr = base_lr
    elif 30 <= epoch < 70:
        lr = base_lr * lr_decay
    elif 70 <= epoch < 150:
        lr = base_lr * (lr_decay)**2
    else:
        lr = base_lr * (lr_decay)**3

    # lr = base_lr * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, is_best, dir_name, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (dir_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + filename
    torch.save(state, filename)
    print("Saved Checkpoint!")

    if is_best:
        print("Best Model found ! ")
        shutil.copyfile(filename, directory + '/model_best.pth.tar')


def load_checkpoint(encoder, decoder, resume_filename):
    start_epoch = 1
    best_loss = 10000

    if resume_filename:
        if os.path.isfile(resume_filename):
            print("=> Loading Checkpoint '{}'".format(resume_filename))
            checkpoint = torch.load(resume_filename)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])

            print("========================================================")

            print("Loaded checkpoint '{}' (epoch {})".format(
                resume_filename, checkpoint['epoch']))
            print("Current Loss : ", checkpoint['best_loss'])

            print("========================================================")

        else:
            print(" => No checkpoint found at '{}'".format(resume_filename))

    return start_epoch, best_loss

def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(outputs, labels)



def check_type(outputs, labels, use_gpu):
    if type(outputs) is not np.ndarray:
        if use_gpu:
            outputs = outputs.cpu()
        outputs = outputs.detach().numpy()

    if type(labels) is not np.ndarray:
        if use_gpu:
            labels = labels.cpu()
        labels = labels.detach().numpy()

    outputs = np.argmax(outputs, axis=1)
    return outputs, labels



def find_metrics(outputs, labels, use_gpu=False):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Returns: (float) accuracy in [0,1]
    """

    outputs, labels = check_type(outputs, labels, use_gpu)

    accuracy = np.sum(outputs == labels) / float(outputs.size)

    prec, rec, _, _ = precision_recall_fscore_support(
        labels, outputs, average='weighted')

    return accuracy, prec, rec
