import time
import torch
from torch.autograd import Variable
from utils.utils import AverageMeter




def train(data_loader, encoder, decoder, optimizer, loss_fn, metrics_fn, epoch, writer,
          use_gpu=False):

    encoder.train()
    decoder.train()

    losses = AverageMeter()
    accuracies = AverageMeter()
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_score, _, _ = metrics_fn(output_batch, label_batch, use_gpu=use_gpu)
        accuracies.update(acc_score)

        writer.add_scalar('data/stepwise_training_loss', losses.val, niter)
        writer.add_scalar('data/stepwise_training_accuracy', accuracies.val, niter)

        print("Step: %d, Current Loss: %0.4f, Average Loss: %0.4f" %
              (i, loss, losses.avg))

        break


    writer.add_scalar('data/training_loss', losses.avg, epoch)
    writer.add_scalar('data/training_accuracy', accuracies.avg, epoch)
    return losses.avg



def validate(data_loader, encoder, decoder, loss_fn, metrics_fn, epoch, writer, use_gpu=False):

    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    # bleeding_precs = AverageMeter()
    # bleeding_recs = AverageMeter()
    # healthy_precs = AverageMeter()
    # healthy_recs = AverageMeter()
    

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

            acc_score, prec, rec = metrics_fn(output_batch, label_batch, use_gpu=use_gpu)
            accuracies.update(acc_score)

            # acc_score, prec_h, rec_h = metrics_fn(output_batch, label_batch, use_gpu=use_gpu)

            # bleeding_precs.update(float(prec))
            # bleeding_recs.update(float(rec))
            # healthy_precs.update(float(prec_h))
            # healthy_recs.update(float(rec_h))

            writer.add_scalar('data/stepwise_val_loss', losses.val, niter)


        print("Step: %d, Current Loss: %0.4f, Average Loss: %0.4f" %
              (i, loss, losses.avg))

        break


    writer.add_scalar('data/val_loss', losses.avg, epoch)
    writer.add_scalar('data/val_accuracy', accuracies.avg, epoch)
    # writer.add_scalar('data/val_healthy_precision', healthy_precs.avg, epoch)
    # writer.add_scalar('data/val_healthy_recall', healthy_recs.avg, epoch)
    # writer.add_scalar('data/val_bleeding_precision',bleeding_precs.avg, epoch)
    # writer.add_scalar('data/val_bleeding_recall', bleeding_recs.avg, epoch)
    return losses.avg






