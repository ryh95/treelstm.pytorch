from __future__ import print_function

import os, time, random, argparse
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var

# IMPORT CONSTANTS
import Constants
# NEURAL NETWORK MODULES/LAYERS
from model import *
# DATA HANDLING CLASSES
from tree import Tree
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import SICKDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
from utils import load_word_vectors, build_vocab
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import Trainer

# MAIN BLOCK
def main():
    global args
    args = parse_args()
    args.input_dim, args.mem_dim = 300, 150
    args.hidden_dim, args.num_classes = 50, 2
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.sparse and args.wd!=0:
        print('Sparsity and weight decay are incompatible, pick one!')
        exit()
    print(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = os.path.join(args.data,'train/')
    dev_dir = os.path.join(args.data,'dev/')
    # test_dir = os.path.join(args.data,'test/')

    # write unique words from all token files
    sick_vocab_file = os.path.join(args.data,'sick.vocab')
    if not os.path.isfile(sick_vocab_file):
        token_files_a = [os.path.join(split,'a.toks') for split in [train_dir,dev_dir]]
        token_files_b = [os.path.join(split,'b.toks') for split in [train_dir,dev_dir]]
        token_files = token_files_a+token_files_b
        sick_vocab_file = os.path.join(args.data,'sick.vocab')
        build_vocab(token_files, sick_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=sick_vocab_file, data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    print('==> SICK vocabulary size : %d ' % vocab.size())

    # load SICK dataset splits
    train_file = os.path.join(args.data,'sick_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SICKDataset(train_dir, vocab, args.num_classes)
        torch.save(train_dataset, train_file)
    print('==> Size of train data   : %d ' % len(train_dataset))
    dev_file = os.path.join(args.data,'sick_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SICKDataset(dev_dir, vocab, args.num_classes)
        torch.save(dev_dataset, dev_file)
    print('==> Size of dev data     : %d ' % len(dev_dataset))

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
                args.cuda, vocab.size(),
                args.input_dim, args.mem_dim,
                args.hidden_dim, args.num_classes,
                args.sparse)
    criterion = nn.KLDivLoss()
    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim=='adam':
        optimizer   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        optimizer   = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='sgd':
        optimizer   = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sick_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove,'glove.840B.300d'))
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.Tensor(vocab.size(),glove_emb.size(1)).normal_(-0.05,0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()
    model.childsumtreelstm.emb.state_dict()['weight'].copy_(emb)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer)

    check_path = os.path.join(args.save, args.expname + '.pth.pt')

    checkpoint = torch.load(check_path)
    args = checkpoint['args']
    optimizer = checkpoint['optim']
    model_para = checkpoint['model']

    model.load_state_dict(model_para)

    best = -float('inf')
    for epoch in range(args.epochs):
        # train_loss             = trainer.train(train_dataset)
        train_loss, train_pred = trainer.test(train_dataset)
        dev_loss, dev_pred     = trainer.test(dev_dataset)

        train_pearson = metrics.pearson(train_pred,train_dataset.labels)
        train_mse = metrics.mse(train_pred,train_dataset.labels)
        train_f1 = metrics.f1(train_pred,train_dataset.labels)
        print('==> Train    Loss: {}\tPearson: {}\tMSE: {}\tF1: {}'.format(train_loss,train_pearson,train_mse,train_f1))
        dev_pearson = metrics.pearson(dev_pred,dev_dataset.labels)
        dev_mse = metrics.mse(dev_pred,dev_dataset.labels)
        dev_f1 = metrics.f1(dev_pred,dev_dataset.labels)
        print('==> Dev      Loss: {}\tPearson: {}\tMSE: {}\tF1: {}'.format(dev_loss,dev_pearson,dev_mse,dev_f1))

        if best < dev_pearson:
            best = dev_pearson
            checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
                          'pearson': dev_pearson, 'mse': dev_mse,
                          'args': args, 'epoch': epoch }
            print('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname+'.pth'))

def inference(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    args = checkpoint['args']
    optimizer = checkpoint['optim']
    model_para = checkpoint['model']

    sick_vocab_file = os.path.join(args.data, 'sick.vocab')
    vocab = Vocab(filename=sick_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])

    model = SimilarityTreeLSTM(
        args.cuda, vocab.size(),
        args.input_dim, args.mem_dim,
        args.hidden_dim, args.num_classes,
        args.sparse)
    criterion = nn.KLDivLoss()
    metrics = Metrics(args.num_classes)

    model.load_state_dict(model_para)

    dev_file = os.path.join(args.data, 'sick_dev.pth')
    dev_dir = os.path.join(args.data, 'dev/')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SICKDataset(dev_dir, vocab, args.num_classes)
        torch.save(dev_dataset, dev_file)
    print('==> Size of dev data     : %d ' % len(dev_dataset))

    train_dir = os.path.join(args.data, 'train/')
    train_file = os.path.join(args.data, 'sick_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SICKDataset(train_dir, vocab, args.num_classes)
        torch.save(train_dataset, train_file)
    print('==> Size of train data   : %d ' % len(train_dataset))

    trainer = Trainer(args, model, criterion, optimizer)
    dev_loss, dev_pred = trainer.test(dev_dataset)
    train_loss, train_pred = trainer.test(train_dataset)

    train_pearson = metrics.pearson(train_pred, train_dataset.labels)
    train_mse = metrics.mse(train_pred, train_dataset.labels)
    train_f1 = metrics.f1(train_pred, train_dataset.labels)
    print('==> Train    Loss: {}\tPearson: {}\tMSE: {}\tF1: {}'.format(train_loss, train_pearson, train_mse, train_f1))

    dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
    dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
    dev_f1 = metrics.f1(dev_pred, dev_dataset.labels)
    print('==> Dev      Loss: {}\tPearson: {}\tMSE: {}\tF1: {}'.format(dev_loss, dev_pearson, dev_mse, dev_f1))


if __name__ == "__main__":
    # main()
    global args
    args = parse_args()
    check_path = os.path.join(args.save, args.expname+'.pth.pt')
    inference(check_path)
