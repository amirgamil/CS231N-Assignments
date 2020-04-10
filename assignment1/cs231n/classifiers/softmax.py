import numpy as np
from random import shuffle
from past.builtins import xrange
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_iterations = X.shape[0]
    classes = W.shape[1]
    gradients = np.zeros((num_iterations, classes))
    #scores matrix will have shape N by C
    scores = np.dot(X, W)
    
    for sample in range(num_iterations):
        #regularize to avoid blow up
        scores_a = X[sample].dot(W)
        scores_a -= scores_a.max()

        scores[sample] -= np.max(scores[sample])
        p = np.exp(scores[sample][y[sample]]) / np.sum(np.exp(scores[sample]))
        loss += -1*np.log(p)
        scores_expsum = np.sum(np.exp(scores[sample]))
        cor_ex = np.exp(scores[sample][y[sample]])

        
        #gradient for correct class
        dW[:,y[sample]] += -1  * (scores_expsum - cor_ex) / scores_expsum *X[sample]
        #incorrect class gradients
        for j in range(classes):
            if j == y[sample]:
                continue
            dW[:,j] += np.exp(scores[sample][j]) / scores_expsum * X[sample]

    loss = loss/num_iterations
    dW /= num_iterations
    dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_iterations = X.shape[0]
    scores = np.dot(X, W)
    maxes = np.max(scores, axis = 1)
    maxes = maxes[:, np.newaxis]
    scores = scores - maxes  
    
    exp = np.exp(scores)
    sum_rows = np.sum(exp, axis=1).reshape(exp.shape[0],1)
    scores = exp/sum_rows
    y_one_hot_encoded = np.zeros((y.shape[0], 10))
    y_one_hot_encoded[np.arange(y.shape[0]), y] = 1
    loss = np.trace(np.dot(y_one_hot_encoded, np.log(scores.T)))
    loss /= -num_iterations
    loss += reg * np.sum(W * W)
    dW = np.dot(X.T, (scores - y_one_hot_encoded))/num_iterations
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def jacobian_softmax(s):
    """Return the Jacobian matrix of softmax vector s.

    :type s: ndarray
    :param s: vector input

    :returns: ndarray of shape (len(s), len(s))
    """
    return np.diag(s) - np.outer(s, s)

    
    




