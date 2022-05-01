#!/bin/env/python3
'''
Some material taken from Inzamam Rahaman's COMP3608 Lecture Notes
URL: https://github.com/InzamamRahaman/COMP3608-2020
'''

import torch as th
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.BCELoss()

    def forward(self, X):
        return None

    def loss(self, X, y):
        probs = self.forward(X)
        return self.loss_func(probs, y)

    def predict_proba(self, X, as_numpy=False):
        res = self.forward(X)
        if as_numpy:
            res = res.detach().numpy()
        return res

    def predict(self, X, threshold=0.5, as_numpy=False):
        probs = self.predict_proba(X, as_numpy)
        return probs > threshold

    def fit(self, X, y, epochs=1000, lr=0.1, lam=0):
        optimizer = optim.RMSprop(self.parameters(), lr=lr)
        loss_curve = []
        for _ in range(epochs):
            optimizer.zero_grad()
            loss_val = self.loss(X, y) + self.regularize(lam)
            loss_curve.append(loss_val.data.item())
            loss_val.backward()
            optimizer.step()
        return loss_curve

    def regularize(self, lam):
        loss_val = 0
        for p in self.parameters():
            loss_val += lam * th.norm(p)
        return loss_val


class NNModel(Model):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Conv1d(128, 10, 5, stride=1)
        self.l2 = nn.Conv1d(64, 10, 5, stride=1)
        self.l3 = nn.MaxPool1d(3, stride =1)
        self.l4 = nn.Conv1d(2, 3, stride=1)
        self.act1 = nn.Sigmoid()

    def get_intermediary(self, X):
        res = self.l1(X)
        res = self.act1(res)
        res = self.l2(res)
        res = self.l3(res)
        res = self.l4(res)
        return res

    def forward(self, X):
        res = self.get_intermediary(X)
        res = self.l3(res)
        res = self.act1(res)
        return res
