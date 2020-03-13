import torch
import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    # Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs
    lr = base_lr * (0.1 ** (epoch // lr_decay))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, is_best, dir_name, filename='checkpoint.pth.tar'):
    directory = "../runs/%s/" % (dir_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + filename
    torch.save(state, filename)
    print("Saved Checkpoint!")

    if is_best:
        print("Best Model found ! ")
        shutil.copyfile(filename, 'runs/%s/' %
                        (args.name) + 'model_best.pth.tar')


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
