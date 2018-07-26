import argparse
import copy
import os
import torch

from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model import NN4SNLI
from data import SNLI
from test import test


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameters(model):
    print("parameters of each layer")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print('    ' + str(n))
            print('    ' + str(p.size()))


def train(args, data):
    model = NN4SNLI(args, data)
    if args.gpu > -1:
        model.cuda(args.gpu)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    print("number of all parameters: " + str(count_parameters(model)))
    print(print_parameters(model))

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    acc, loss, size, last_epoch = 0, 0, 0, -1
    max_dev_acc, max_test_acc = 0, 0

    iterator = data.train_iter
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
        last_epoch = present_epoch

        p, p_lens = batch.premise
        h, h_lens = batch.hypothesis

        if not args.no_char_emb:
            char_p = torch.LongTensor(data.characterize(p))
            char_h = torch.LongTensor(data.characterize(h))

            if args.gpu > -1:
                char_p = char_p.cuda(args.gpu)
                char_h = char_h.cuda(args.gpu)

            setattr(batch, 'char_p', char_p)
            setattr(batch, 'char_h', char_h)

        pred = model(batch)

        optimizer.zero_grad()
        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.item()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(parameters, max_norm=args.norm_limit)
        optimizer.step()

        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)

        if (i + 1) % args.print_freq == 0:
            acc /= size
            acc = acc.cpu().item()
            dev_loss, dev_acc = test(model, data, args, mode='dev')
            test_loss, test_acc = test(model, data, args)
            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('acc/train', acc, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('acc/dev', dev_acc, c)
            writer.add_scalar('loss/test', test_loss, c)
            writer.add_scalar('acc/test', test_acc, c)

            print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f} / test loss: {test_loss:.3f}'
                  f' / train acc: {acc:.3f} / dev acc: {dev_acc:.3f} / test acc: {test_acc:.3f}')

            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(model)

            acc, loss, size = 0, 0, 0
            model.train()

    writer.close()
    print(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')

    return best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-type', default='SNLI')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-dim', default=600, type=int)
    parser.add_argument('--learning-rate', default=4e-4, type=float)
    parser.add_argument('--print-freq', default=1000, type=int)
    parser.add_argument('--weight-decay', default=5e-5, type=float)
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--char-dim', default=15, type=int)
    parser.add_argument('--num-feature-maps', default=100, type=int)
    parser.add_argument('--num-layers', default=3, type=int)
    parser.add_argument('--num-heads', default=5, type=int)
    parser.add_argument('--no-char-emb', default=False, action='store_true')
    parser.add_argument('--norm-limit', default=10, type=float)

    args = parser.parse_args()

    print('loading SNLI data...')
    data = SNLI(args)

    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'max_word_len', data.max_word_len)
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    setattr(args, 'FILTER_SIZES', [1, 3, 5])

    print('training start!')
    best_model = train(args, data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(), f'saved_models/BiLSM_GP_{args.data_type}_{args.model_time}.pt')

    print('training finished!')


if __name__ == '__main__':
    main()
