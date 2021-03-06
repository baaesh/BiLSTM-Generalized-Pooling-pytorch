# Enhancing Sentence Embedding with Generalized Pooling
Pytorch re-implementation of [Enhancing Sentence Embedding with Generalized Pooling](https://arxiv.org/abs/1806.09828) without penalization.

This is an unofficial implementation. There is [the implementation by the authors](https://github.com/lukecq1231/generalized-pooling), which is implemented on Theano.

## Results
Dataset: [SNLI](https://nlp.stanford.edu/projects/snli/)

| Model | Valid Acc(%) | Test Acc(%)
| ----- | ------------ | -----------
| Baseline from the paper (without penalization) | - | 86.4 |
| Re-implemenation | 86.4 | 85.7 |

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.6
- Pytorch: 0.4.0

## Requirements
Please install the following library requirements first.

    nltk==3.3
    tensorboardX==1.2
    torch==0.4.0
    torchtext==0.2.3
    
## Training
> python train.py --help

    usage: train.py [-h] [--batch-size BATCH_SIZE] [--data-type DATA_TYPE]
                [--dropout DROPOUT] [--epoch EPOCH] [--gpu GPU]
                [--hidden-dim HIDDEN_DIM] [--learning-rate LEARNING_RATE]
                [--print-freq PRINT_FREQ] [--weight-decay WEIGHT_DECAY]
                [--word-dim WORD_DIM] [--char-dim CHAR_DIM]
                [--num-feature-maps NUM_FEATURE_MAPS]
                [--num-layers NUM_LAYERS] [--num-heads NUM_HEADS]
                [--no-char-emb] [--norm-limit NORM_LIMIT]

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE
      --data-type DATA_TYPE
      --dropout DROPOUT
      --epoch EPOCH
      --gpu GPU
      --hidden-dim HIDDEN_DIM
      --learning-rate LEARNING_RATE
      --print-freq PRINT_FREQ
      --weight-decay WEIGHT_DECAY
      --word-dim WORD_DIM
      --char-dim CHAR_DIM
      --num-feature-maps NUM_FEATURE_MAPS
      --num-layers NUM_LAYERS
      --num-heads NUM_HEADS
      --no-char-emb
      --norm-limit NORM_LIMIT

**Note:** 
- Only codes to use SNLI as training data are implemented.

