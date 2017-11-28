from __future__ import print_function

import os, math
import torch
from copy import deepcopy
from torch.autograd import Variable as Var
from tree import Tree
from vocab import Vocab

# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path):
    if os.path.isfile(path+'.pth') and os.path.isfile(path+'.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path+'.pth')
        vocab = Vocab(filename=path+'.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path+'.txt'))
    with open(path+'.txt','r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None]*(count)
    vectors = torch.zeros(count,dim)
    with open(path+'.txt','r') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(map(float, contents[1:]))
            idx += 1
    with open(path+'.vocab','w') as f:
        for word in words:
            f.write(word+'\n')
    vocab = Vocab(filename=path+'.vocab')
    torch.save(vectors, path+'.pth')
    return vocab, vectors

# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile):
    vocab = set()
    for filename in filenames:
        with open(filename,'r') as f:
            for line in f:
                tokens = line.rstrip('\n').split(' ')
                vocab |= set(tokens)
    with open(vocabfile,'w') as f:
        for token in sorted(vocab):
            f.write(token+'\n')

# mapping from scalar to vector
def map_label_to_target(label,num_classes):
    target = torch.zeros(1,num_classes)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil==floor:
        target[0][floor-1] = 1
    else:
        target[0][floor-1] = ceil - label
        target[0][ceil-1] = label - floor
    return target

def collect_wrong_samples(preds,labels,dev_file_path,wrong_file_path):
    x = Var(deepcopy(preds), volatile=True)
    y = Var(deepcopy(labels), volatile=True)
    # TODO: make 1.5(1+2/2)
    x = (x >= 1.5).type(torch.IntTensor)
    y = (y - 1).type(torch.IntTensor)
    equal = (x == y).type(torch.IntTensor)

    with open(dev_file_path,'r') as f,\
         open(wrong_file_path,'w') as f_out:
        f.readline()
        f_out.write('pair_ID' + '\t' + 'sentence_A' + '\t' + 'sentence_B' + '\t' + 'relatedness_score' + '\t' + 'pred_score'+ '\n' )
        for idx,line in enumerate(f):
	    # TODO: fix [:5] in python3(*rest)
            i, a, b, sim, ent = line.strip().split('\t')[:5]
            if equal[idx].data[0] == 0:
                # means model makes mistakes
                f_out.write('\t'.join([i,a,b,sim,str(preds[idx])]) + '\n')

def dump_preds(preds,file_path,result_file_path):
    x = Var(deepcopy(preds), volatile=True)
    with open(file_path,'r') as f,\
         open(result_file_path,'w') as f_out:
        f.readline()
        f_out.write('\t'.join(['pair_ID','sentence_A','sentence_B','pred_score','relatedness_score','is_adversarial']))
        f_out.write('\n')
        for idx,line in enumerate(f):
            line = line.strip('\n')
            pair_ID,sen_A,sen_B,re_score,ans,is_adver = line.split('\t')
            pred_value = x[idx].data[0]
            f_out.write('\t'.join([pair_ID,sen_A,sen_B,str(pred_value),re_score,is_adver])+'\n')
