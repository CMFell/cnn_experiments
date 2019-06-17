import torch
import numpy as np

def accuracy(y_pred, y_true, thresh):
    outputs = y_pred.unsqueeze(4)
    outputs = torch.chunk(outputs, 5, dim=3)
    outputs = torch.cat(outputs, dim=4)
    outputs = outputs.transpose(4, 3)
    predconf = torch.sigmoid(outputs[..., 4])
    ones = y_true[..., 4]
    poz = torch.ge(predconf, thresh)
    negz = torch.lt(predconf, thresh)
    truez = torch.ge(ones, thresh)
    falsez = torch.lt(ones, thresh)
    tp = torch.sum(np.logical_and(poz, truez))
    fp = torch.sum(np.logical_and(poz, falsez))
    fn = torch.sum(np.logical_and(negz, truez))

    return tp, fp, fn