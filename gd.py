"""
Generic implementation of gradient descent.
"""
import numpy as np
import util

def gd(func, grad, x0, numIter, stepSize):
    """
    Perform gradient descent on some function func, where grad(x)
    computes its gradient at position x. Begin at position x0 and run
    for exactly numIter iterations. Use stepSize/sqrt(t) as a
    step-size, where t is the iteration number (starting from 1).

    Returns:
    (x, trajectory): The final optimized value and the trajectory array.
    """
    x = x0
    trajectory = np.zeros(numIter + 1)
    trajectory[0] = func(x)

    for iter in range(numIter):
        t = iter + 1
        
        # compute the gradient at the current location
        g = util.raiseNotDefined()    ### TODO: YOUR CODE HERE

        # compute the step size
        eta = util.raiseNotDefined()  ### TODO: YOUR CODE HERE

        # step in the direction of the gradient
        x = util.raiseNotDefined()    ### TODO: YOUR CODE HERE

        # record the trajectory
        trajectory[t] = func(x)

    return (x, trajectory)