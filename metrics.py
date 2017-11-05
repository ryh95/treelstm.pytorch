from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
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
        x = (x >= 1.5).type(torch.IntTensor)
        y = y - 1
        return f1_score(y.data.numpy(),x.data.numpy())

if __name__ == "__main__":
    # test f1
    pred = torch.FloatTensor([1.4,1.6,1.6,1.1,2,2])
    true = torch.FloatTensor([1,2,1,1,2,1])
    print Metrics(2).f1(pred,true)

    # test random F1 and MSE
    true = [np.concatenate((np.ones(50),np.zeros(50))).tolist() for _ in range(100)]
    pred = [np.random.randint(2, size=100).tolist() for _ in range(100)]
    pred = torch.FloatTensor(pred)
    true = torch.FloatTensor(true)
    print Metrics(2).mse(pred,true)

    pred = pred + 1
    true = true + 1
    print sum(Metrics(2).f1(pred[i],true[i]) for i in range(100))/float(100)

