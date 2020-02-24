import os
import time
import torch
from torchvision import transforms
import math
import argparse
import shutil

from dataloader import trainloader
from models import EncoderCNN, DecoderRNN

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--n_epochs', default=10)
parser.add_argument('-b', default=1)
parser.add_argument('--gpu', deafult=False)


def main(args):

    print("Process %s, running on %s: starting (%s)" % (
        os.getpid(), os.name, time.asctime()))

    encoder = EncoderCNN()
    decoder = DecoderRNN()
    if torch.cuda.is_available() and args.gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder_trainables = [p for p in encoder.parameters() if p.requires_grad]
    decoder_trainables = [p for p in decoder.parameters() if p.requires_grad]

    params = encoder_trainables + decoder_trainables

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    data_loader = trainloader(transform=transform)
    optimizer = torch.optim.SGD(params=params, lr=args.lr, momentum=0.9)
    
