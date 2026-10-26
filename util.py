import numpy as np
import matplotlib.pyplot as plt

def raiseNotDefined():
    raise Exception("Method not implemented: TODO")

def plot_decision_boundary(X, Y, classifier, title="Decision Boundary"):
    """
    Plots the decision boundary of a trained classifier on a 2D dataset.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Values < 0 get light red, Values > 0 get light blue.
    plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=['#FFAAAA', '#AAAAFF'], alpha=0.5)
    
    # Draw a line exactly where the boundary transitions
    plt.contour(xx, yy, Z, colors='k', levels=[0], linewidths=1.5)
    
    # Plot Class -1 with red squares
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], 
                c='red', marker='s', s=70, edgecolor='k', label='Class -1')
    
    # Plot Class +1 with blue triangles
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], 
                c='blue', marker='^', s=70, edgecolor='k', label='Class +1')
    
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.legend(loc='best')
    plt.show()