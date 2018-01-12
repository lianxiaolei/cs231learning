#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)
    num_train = X.shape[0]
    num_class = W.shape[1]
    loss = 0.0
    for i in range(num_train):
        score = X[i].dot(W)
        for j in range(num_class):
            if j == y[i]:
                continue
            margin = score[j] - score[y[i]] + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] -= X[i].T

    dW /= num_train
    loss /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[range(num_train), list(y)] = 0
    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

    mask = np.zeros((num_train, num_classes))
    mask[margins > 0] = 1
    mask[range(num_train), list(y)] = 0
    mask[range(num_train), list(y)] = -np.sum(mask, axis=1)
    dW = (X.T).dot(mask)
    dW = dW / num_train + reg * W

    return loss, dW
