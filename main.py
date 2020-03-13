import os
import time
import torch
from torchvision import transforms
import math
import argparse
import shutil

from data.dataloader import DataLoader
from models.models import EncoderCNN, DecoderRNN
from utils import utils
import train

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--n_epochs', default=10)
parser.add_argument('-b', default=1)
parser.add_argument('--gpu', default=False)
parser.add_argument('--name', default="DefaultRun")
parser.add_argument('--resume', default='')

_DATASET_PATH = '../data/Data/DataSet13_20200221/raw_patient_based'
def main(args):

    print("Process %s, running on %s: starting (%s)" % (
        os.getpid(), os.name, time.asctime()))

    use_gpu = args.gpu
    if not torch.cuda.is_available():
        use_gpu = False

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

    train_loader = DataLoader(transform=transform, mode="train", dataset_path=_DATASET_PATH)
    val_loader = DataLoader(transform=transform, mode="val", dataset_path=_DATASET_PATH)

    start_epoch, best_loss = utils.load_checkpoint(encoder, decoder, args.resume)

    optimizer = torch.optim.SGD(params=params, lr=args.lr, momentum=0.9)

    # Batch Size = 1 in this case
    train_steps = math.ceil(len(train_loader))
    epoch = start_epoch
    best_epoch = start_epoch

    while epoch <= int(args.n_epochs):
        print("========================================================")

        print("Epoch %d Training Starting" % epoch)
        print("Learning Rate : ", utils.get_lr(optimizer))

        train_loss = train.train(train_loader, encoder, decoder,
                           optimizer, epoch, use_gpu)
        val_loss = train.validate(encoder, decoder, val_loader, epoch, use_gpu)

        print("Training Loss: ", float(train_loss.data))
        print("Validation Loss: ", float(val_loss.data))
        print("========================================================")

        utils.save_checkpoint(
            {
                "epoch": epoch,
                "best_loss": min(best_loss, val_loss),
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict()
            },
            val_loss < best_loss,
            best_epoch=epoch,
            best_loss=val_loss,
            dir_name=args.name
        )
        if val_loss < best_loss:
          best_epoch = epoch
          best_loss = val_loss

        epoch += 1


if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
