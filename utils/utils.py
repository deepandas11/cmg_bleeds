import torch
import os
import shutil
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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


def find_metrics(outputs, labels, thresh=0.5, pos_label=1, use_gpu=False):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Returns: (float) accuracy in [0,1]
    """

    if use_gpu:
        outputs = outputs.cpu()
        labels = labels.cpu()
        
    outputs = outputs.detach().numpy()
    labels = labels.detach().numpy()

    outputs[outputs >= thresh] = 1.0
    outputs[outputs < thresh] = 0.0

    acc_score = accuracy_score(labels, outputs)
    prec, rec, _, _ = precision_recall_fscore_support(
        labels, outputs, average='binary', pos_label=pos_label)
    return acc_score, prec, rec
