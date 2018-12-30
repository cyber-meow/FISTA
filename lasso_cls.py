import numbers
from copy import deepcopy
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.nn import functional as F

import fista
import proximal


class ClassifierLasso(object):

    def __init__(self, model, data, target,
                 test_data=None, test_target=None, regularize_idxs=None):
        self.model = model
        self.data = data
        self.target = target
        self.test_data = test_data
        self.test_target = test_target
        self.regularize_idxs = regularize_idxs
        self.reset()

    def reset(self):
        self.counter = 0
        self.n_used_samples = [0]
        self.ce_losses = []
        self.l1_losses = []
        self.test_accuracies = []
        self.full_losses = []
        self.params_his = [deepcopy(list(self.model.parameters()))]

    def train(self, n_iters, lr, lamb, batch_size=None,
              alpha=None, test=False, print_interval=100,
              store_params=False, save_full_loss=False):
        """train the model for a number of iterations

        :n_iter: integer, number of iterations to run
        :param lr: learning rate, constant or function of iteration
        :param lamb: regularization parameter, constant
        :param batch_size:
            constant or function of iteration, use full dataset if not given
        :param alpha: function of iteration, ignored if not given
        """
        proximal_op = proximal.SoftThresholding(lamb)
        if alpha is None:
            optimizer = fista.ForwardBackward(
                self.model.parameters(), 1, proximal_op,
                regularize_idxs=self.regularize_idxs)
        else:
            optimizer = fista.FISTA(
                self.model.parameters(), 1, proximal_op,
                regularize_idxs=self.regularize_idxs)
        if isinstance(lr, numbers.Number):
            decay = lambda _: lr
        else:
            decay = lambda k: lr(k)
        # Should update learning rate for torch.optim.optimizer
        scheduler = LambdaLR(
            optimizer.optimizer, lr_lambda=decay)
        scheduler.last_epoch = self.counter
        for _ in range(n_iters):
            self._train_step(optimizer, scheduler, batch_size, alpha)
            self.l1_losses.append(self.l1_loss() * lamb)
            if test:
                self.test(update=True)
            if store_params:
                self.params_his.append(
                    deepcopy(list(self.model.parameters())))
            if self.counter % print_interval == 0:
                self.log(test)
            if save_full_loss:
                outputs = self.model(self.data)
                full_loss = F.cross_entropy(outputs, self.target).item()
                full_loss += self.l1_losses[-1]
                self.full_losses.append(full_loss)

    def _train_step(self, optimizer, scheduler, batch_size, alpha):
        scheduler.step()
        optimizer.zero_grad()
        if batch_size is None:
            X = self.data
            labels = self.target
            bs = self.data.size(0)
        else:
            if isinstance(batch_size, numbers.Number):
                bs = batch_size
            else:
                bs = batch_size(self.counter + 1)
            if bs > self.data.size(0):
                warn('Batch size larger than dataset size')
            idxs = np.random.choice(self.data.size(0), bs)
            X = self.data[idxs]
            labels = self.target[idxs]
        self.n_used_samples.append(self.n_used_samples[-1] + bs)
        outputs = self.model(X)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        if alpha is None:
            optimizer.step()
        else:
            optimizer.step(alpha(self.counter + 1))
        self.ce_losses.append(loss.item())
        self.counter += 1

    def log(self, test):
        print(f'Iteration {self.counter}')
        print(f'CE Loss: {self.ce_losses[-1]}')
        print(f'L1 Loss: {self.l1_losses[-1]}')
        if test:
            print(f'Test Accuracy: {self.test_accuracies[-1]*100:.2f}%')

    def l1_loss(self):
        res = 0
        for i, param in enumerate(self.model.parameters()):
            if i in self.regularize_idxs:
                res += torch.sum(torch.abs(param)).item()
        return res

    def test(self, test_data=None, test_target=None, update=True):
        if test_data is None:
            if self.test_data is None:
                raise Exception('No test data provided')
            test_data = self.test_data
        if test_target is None:
            if self.test_target is None:
                raise Exception('No test target provided')
            test_target = self.test_target
        predicted = torch.argmax(self.model(test_data), dim=1).cpu().numpy()
        acc = accuracy_score(test_target.cpu().numpy(), predicted)
        if update:
            self.test_accuracies.append(acc)
        return acc


class Linear(nn.Module):

    def __init__(self, input_dim, n_classes):
        super(Linear, self).__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


def compute_relative_err(params_his):
    params_his_arr = []
    for params in params_his:
        params_arr = []
        for param in params:
            w = param.cpu().detach().numpy().reshape(-1)
            params_arr.append(w)
        params_his_arr.append(np.concatenate(params_arr))
    params_his_arr = np.array(params_his_arr)
    rel_errs = (
        np.sum((params_his_arr - params_his_arr[-1])**2, axis=1)
        / np.linalg.norm(params_his_arr[-1]))
    return rel_errs


def sparsity_f1_score(params_his):
    f1_scores = []
    # Suppose that we are using the linear model
    w = params_his[-1][0].cpu().detach().numpy().reshape(-1)
    sparse_str_final = w != 0
    for params in params_his:
        # Suppose that we are using the linear model
        w = params[0].cpu().detach().numpy().reshape(-1)
        sparse_str = w != 0
        f1_scores.append(f1_score(sparse_str_final, sparse_str))
    return f1_scores


def plot_full_losses(clf, name, n_simulations, div, **kwargs):
    n_used_samples = np.array(clf.n_used_samples)
    n_used_samples[n_used_samples > n_simulations*div] = 0
    n_used_samples = np.trim_zeros(n_used_samples)/div
    full_losses = np.array(clf.full_losses)
    full_losses -= np.min(full_losses)
    plt.plot(
        n_used_samples, full_losses[:len(n_used_samples)],
        label=name, **kwargs)


def plot_test_accuracies(clf, name, n_simulations, div, **kwargs):
    n_used_samples = np.array(clf.n_used_samples)
    n_used_samples[n_used_samples > n_simulations*div] = 0
    n_used_samples = np.trim_zeros(n_used_samples)/div
    plt.plot(
        n_used_samples, clf.test_accuracies[:len(n_used_samples)],
        label=name, **kwargs)


def plot_relative_err(clf, name, n_simulations, div, **kwargs):
    n_used_samples = np.array(clf.n_used_samples)
    n_used_samples[n_used_samples > n_simulations*div] = 0
    n_used_samples = np.trim_zeros(n_used_samples)/div
    rel_errs = compute_relative_err(clf.params_his)
    plt.plot(
        n_used_samples, rel_errs[:len(n_used_samples)],
        label=name, **kwargs)


def plot_sparsity_f1_scores(clf, name, div, **kwargs):
    n_used_samples = np.array(clf.n_used_samples)[:-1]/div
    sp_f1 = sparsity_f1_score(clf.params_his)[:-1]
    plt.plot(n_used_samples, sp_f1, label=name, **kwargs)
