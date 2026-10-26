"""
Implementation of regularized linear classification/regression
"""
import numpy as np
import util
from gd import gd

class LossFunction:
    def loss(self, Y, Yhat):
        util.raiseNotDefined()

    def lossGradient(self, X, Y, Yhat):
        util.raiseNotDefined()


class SquaredLoss(LossFunction):
    def loss(self, Y, Yhat):
        return 0.5 * np.dot(Y - Yhat, Y - Yhat)

    def lossGradient(self, X, Y, Yhat):
        return -np.dot(X.T, (Y - Yhat))


class LogisticLoss(LossFunction):
    def loss(self, Y, Yhat):
        util.raiseNotDefined()    ### TODO: YOUR CODE HERE

    def lossGradient(self, X, Y, Yhat):
        util.raiseNotDefined()    ### TODO: YOUR CODE HERE


class HingeLoss(LossFunction):
    def loss(self, Y, Yhat):
        util.raiseNotDefined()    ### TODO: YOUR CODE HERE

    def lossGradient(self, X, Y, Yhat):
        util.raiseNotDefined()    ### TODO: YOUR CODE HERE


class ExpLoss(LossFunction):
    def loss(self, Y, Yhat):
        # TODO: Implement the exponential loss function.
        # HINT: Use np.clip(..., -100, 100) on your exponents to prevent overflow.
        pass

    def lossGradient(self, X, Y, Yhat):
        # TODO: Implement the gradient of the exponential loss.
        # HINT: Use np.clip(..., -100, 100) on your exponents to prevent overflow.
        pass


class LinearClassifierHardcoded:
    """
    A simple classifier that uses hardcoded weights and bias.
    """
    def __init__(self, w, b):
        self.w = np.array(w)
        self.b = b

    def predict(self, X):
        util.raiseNotDefined()    ### TODO: YOUR CODE HERE


class LinearClassifier:
    def __init__(self, opts):
        self.opts = opts
        self.weights = None
        self.bias = 0.0 # Ignore bias for the GD training portion to keep it simple

    def train(self, X, Y):
        lossFn = self.opts['lossFunction']
        lambd = self.opts['lambda']
        numIter = self.opts['numIter']
        stepSize = self.opts['stepSize']

        self.weights = np.zeros(X.shape[1])

        def func(w):
            Yhat = util.raiseNotDefined()  ### TODO: YOUR CODE HERE
            obj = util.raiseNotDefined()   ### TODO: YOUR CODE HERE
            return obj

        def grad(w):
            Yhat = util.raiseNotDefined()  ### TODO: YOUR CODE HERE
            gr = util.raiseNotDefined()    ### TODO: YOUR CODE HERE
            return gr

        self.weights, trajectory = gd(func, grad, self.weights, numIter, stepSize)

    def predict(self, X):
        return np.sign(np.dot(X, self.weights))