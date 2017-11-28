from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score,precision_recall_fscore_support
from torch.autograd import Variable as Var

class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x -= x.mean()
        x /= x.std()
        y -= y.mean()
        y /= y.std()
        return torch.mean(torch.mul(x,y))

    def mse(self, predictions, labels):
        x = Var(deepcopy(predictions), volatile=True)
        y = Var(deepcopy(labels), volatile=True)
        return nn.MSELoss()(x,y).data[0]

    def f1(self, predictions,labels):
        '''

        :param predictions: float tensor
        :param labels: int tensor
        :return:
        '''
        x = Var(deepcopy(predictions), volatile=True)
        y = Var(deepcopy(labels), volatile=True)
        # TODO: make 1.5(1+2/2)
        x = (x >= 1.5).int()
        y = (y - 1).int()
        # TODO: fix own implementation of f1
        # intersection = (x == y).type(torch.IntTensor)
        # x or y
        # intersection = (intersection + x).eq(2)
        # print intersection
        # print x
        # print y
        # print intersection.sum()
        # print x.sum()
        # print y.sum()
        # p = intersection.sum().float()/x.sum().float()
        # r = intersection.sum().float()/y.sum().float()
        # f1 = 2*p*r/(p+r)
        # return p.data[0],r.data[0],f1.data[0]
        answer = precision_recall_fscore_support(y.data.numpy(), x.data.numpy())
        return answer[0][1],answer[1][1],answer[2][1]

if __name__ == "__main__":
    # test f1
    pred = torch.FloatTensor([1.4,1.6,1.6,1.1,2,2])
    true = torch.FloatTensor([1,2,1,1,2,1])
    print (Metrics(2).f1(pred,true))

    # test random F1 and MSE
    true = [np.concatenate((np.ones(50),np.zeros(50))).tolist() for _ in range(100)]
    pred = [np.random.randint(2, size=100).tolist() for _ in range(100)]
    pred = torch.FloatTensor(pred)
    true = torch.FloatTensor(true)
    print (Metrics(2).mse(pred,true))

    pred = pred + 1
    true = true + 1
    print (sum(Metrics(2).f1(pred[i],true[i]) for i in range(100))/float(100))

