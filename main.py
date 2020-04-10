import os
import time
import torch
import argparse
import socket
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from data.dataloader import BleedsDataset, get_loader
from models.models import EncoderCNN, DecoderLSTM
from utils import utils
import train


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--n_epochs', default=100)
parser.add_argument('--batch_size', default=8)
parser.add_argument('--gpu', default=True)
parser.add_argument('--resume', default='')
parser.add_argument('--loss_fn', default='bce')
parser.add_argument('--upsample', default=False)
parser.add_argument('--pretrained', default=False)
parser.add_argument('--name', default="baseline")


def main(args):
    print("Process %s, running on %s: starting (%s)" % (
        os.getpid(), os.name, time.asctime()))
    process_num = round(time.time())
    dir_name = args.name + '_' + str(process_num)
    tb_path = "bleeds_experiments/logs/%s/" % (dir_name)

    writer = SummaryWriter(tb_path)

    use_gpu = args.gpu
    if not torch.cuda.is_available():
        use_gpu = False

    encoder = EncoderCNN(pretrained=args.pretrained)
    decoder = DecoderLSTM()

    if use_gpu:
        cudnn.benchmark = True
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        # transforms.RandomPerspective(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    train_dataset = BleedsDataset(
        transform=transform, mode="train", dataset_path=_DATASET_PATH, batch_size=8, upsample=args.upsample)
    val_dataset = BleedsDataset(transform=val_transform,
                                mode="val", dataset_path=_DATASET_PATH)

    train_loader = get_loader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_loader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    encoder_trainables = [p for p in encoder.parameters() if p.requires_grad]
    decoder_trainables = [p for p in decoder.parameters() if p.requires_grad]

    params = encoder_trainables + decoder_trainables
    optimizer = torch.optim.SGD(params=params, lr=args.lr, momentum=0.9)

    if args.loss_fn == 'mse':
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.BCELoss()
    metrics_fn = utils.find_metrics

    start_epoch, best_loss = utils.load_checkpoint(
        encoder, decoder, args.resume)
    epoch = start_epoch

    while epoch <= int(args.n_epochs):
        print("="*50)
        utils.adjust_learning_rate(args.lr, optimizer, epoch)
        print("Epoch %d Training Starting" % epoch)
        print("Learning Rate : ", utils.get_lr(optimizer))

        print("\n","-"*10, "Training","-"*10,"\n")
        train_loss = train.train(
            train_loader, encoder, decoder, optimizer, loss_fn, metrics_fn, epoch, writer, use_gpu)

        print("\n","-"*10, "Validation","-"*10,"\n")
        val_loss = train.validate(
            val_loader, encoder, decoder, loss_fn, metrics_fn, epoch, writer, use_gpu)

        print("-"*50)
        print("Training Loss: ", float(train_loss))
        print("Validation Loss: ", float(val_loss))
        print("="*50)

        curr_state = state = {
            "epoch": epoch,
            "best_loss": min(best_loss, val_loss),
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict()
        }

        # filename = 'epoch_' + str(epoch) + '_checkpoint.pth.tar'

        utils.save_checkpoint(
            state=curr_state,
            is_best=bool(val_loss < best_loss),
            dir_name=dir_name,
            # filename=filename
        )
        if val_loss < best_loss:
            best_loss = val_loss

        epoch += 1
        writer.add_scalar('data/learning_rate', utils.get_lr(optimizer), epoch)


if __name__ == "__main__":
    args = parser.parse_args()
    if socket.gethostname() == 'eru':
        _DATASET_PATH = '/home/deepandas11/computer/servers/euler/data/Data/DataSet13_20200221/raw_patient_based'
    else:
        _DATASET_PATH = '/srv/home/deepandas11/bleeds/data/Data/DataSet13_20200221/raw_patient_based'
    # else:
    #     _DATASET_PATH = '/home/data/'
    main(args)
