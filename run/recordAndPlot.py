import pickle as pk
import torch
import matplotlib.pyplot as plt
import os

class Record(object):
    def __init__(self, acc:torch.Tensor, loss: torch.Tensor):
        self.root = "../results/"
        self.acc = acc.detach().clone().cpu().numpy()
        self.loss = loss.detach().clone().cpu().numpy()

    def save(self, epoch, folder:str):
        if os.path.exists(self.root + folder) is False:
            os.makedirs(self.root + folder)
        loss_file = open(self.root + folder + "ep" + str(epoch) + "_loss.pkl", "ab+")
        acc_file = open(self.root + folder + "ep" + str(epoch) + "_acc.pkl", "ab+")
        pk.dump(self.loss, loss_file)
        pk.dump(self.acc, acc_file)
# plot class remains implementation
