import argparse

import torch
from torch import nn

from model import NN4SNLI
from data import SNLI


def test(model, data, args, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
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

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.item()

        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)

    acc /= size
    acc = acc.cpu().item()
    return loss, acc


def load_model(args, data):
    model = NN4SNLI(args, data)
    model.load_state_dict(torch.load(args.model_path))

    if args.gpu > -1:
        model.cuda(args.gpu)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--data-type', default='SNLI')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=300, type=int)
    parser.add_argument('--word-dim', default=300, type=int)

    parser.add_argument('--model-path', required=True)

    args = parser.parse_args()

    print('loading SNLI data...')
    data = SNLI(args)

    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    # if block size is lower than 0, a heuristic for block size is applied.
    if args.block_size < 0:
        args.block_size = data.block_size

    print('loading model...')
    model = load_model(args, data)

    _, acc = test(model, data)

    print(f'test acc: {acc:.3f}')