import time
import torch
from torch.autograd import Variable
from utils.utils import AverageMeter




def train(data_loader, encoder, decoder, optimizer, loss_fn, epoch, writer,
          use_gpu=False):

    encoder.train()
    decoder.train()

    losses = AverageMeter()
    epoch_steps = len(data_loader)

    for i, (train_batch, label_batch) in enumerate(data_loader):
        niter = (epoch - 1)*epoch_steps + i

        if use_gpu:
            train_batch, label_batch = train_batch.cuda(), label_batch.cuda()


        train_batch, label_batch = map(Variable, (train_batch, label_batch))
        output_batch = encoder(train_batch)
        output_batch = decoder(output_batch)

        loss = loss_fn(output_batch, label_batch)
        losses.update(loss.item())
        writer.add_scalar('data/stepwise_training_loss', losses.val, niter)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Step: %d, Current Loss: %0.4f, Average Loss: %0.4f" %
              (i, loss, losses.avg))

    writer.add_scalar('data/training_loss', losses.avg, epoch)
    return losses.avg



def validate(data_loader, encoder, decoder, loss_fn, epoch, writer, use_gpu=False):

    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    epoch_steps = len(data_loader)

    for i, (train_batch, label_batch) in enumerate(data_loader):
        niter = (epoch - 1)*epoch_steps + i

        if use_gpu:
            train_batch, label_batch = train_batch.cuda(), label_batch.cuda()

        with torch.no_grad():
            train_batch, label_batch = map(Variable, (train_batch, label_batch))
            output_batch = encoder(train_batch)
            output_batch = decoder(output_batch)    

            loss = loss_fn(output_batch, label_batch)
            losses.update(loss.item())
            
            writer.add_scalar('data/stepwise_val_loss', losses.val, niter)


        print("Step: %d, Current Loss: %0.4f, Average Loss: %0.4f" %
              (i, loss, losses.avg))

    writer.add_scalar('data/val_loss', losses.avg, epoch)
    return losses.avg






