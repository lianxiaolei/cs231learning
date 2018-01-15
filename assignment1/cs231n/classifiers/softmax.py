#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    loss = 0.0
    dW = np.zeros_like(W)

    num_train, num_feature = X.shape
    num_class = W.shape[1]

    for i in xrange(num_train):
        scores = X[i].dot(W)
        shift_scores = scores - max(scores)
        loss_i = -shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
        loss += loss_i
        for j in xrange(num_class):
            softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
            if j == y[i]:
                dW[:, j] += (-1 + softmax_output) * X[i]
            else:
                dW[:, j] += softmax_output * X[i]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    loss = 0.0
    dW = np.zeros_like(W)
    scores = X.dot(W)
    shift_scores = scores - max(scores)
    loss = np.sum()
