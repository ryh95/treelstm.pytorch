from copy import deepcopy
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
        x = Var(deepcopy(predictions), volatile=True)
        y = Var(deepcopy(labels), volatile=True)
        return f1_score(y.data.numpy(),x.data.numpy())

if __name__ == "__main__":
    # test f1
    pred = torch.FloatTensor([0,0,1,1,0,0])
    true = torch.FloatTensor([0,0,1,1,0,1])
    print Metrics(2).f1(pred,true)