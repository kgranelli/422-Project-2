import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import datasets
import multiclass
import gd
import linear

import matplotlib.pyplot as plt



def part1():
    print("PART 1: MULTICLASS CLASSIFICATION")
    
    # Q1: Data Familiarization
    X_train = datasets.WineData.X
    y_train = datasets.WineData.Y
    X_test = datasets.WineData.Xte
    y_test = datasets.WineData.Yte
    
    # TODO: write code for data inspection (Q1 A, B, C)
    print("\n--- Q1: Data Familiarization ---")

    # Q2: Multiclass
    # TODO: implement OVA and AVA in multiclass.py
    print("\n--- Q2: OVA and AVA Decision Trees ---")
    # Hint for timing:
    # start = time.time()
    # Train here...
    # end = time.time()
    # time_taken = end - start
    # print(f"Took {time_taken} sec")

    # Q4: Tree Reduction
    # TODO: implement MCTree in multiclass.py
    print("\n--- Q4: Balanced Tree Reduction ---")

    # Q5: Plotting complexity
    print("\n--- Q5: Tree Complexity Plot ---")
    # TODO: Generate plot


def part2():
    print("PART 2: GRADIENT DESCENT")
    
    # Q6: Implementation & Simple Convex
    # TODO: Run gd on x^2. Plot trajectory overlaid on function, and loss over time.
    
    # Q7: Step Size Analysis (Positive)
    # TODO: Find positive step sizes that cause convergence, and oscillation/divergence.
    # For BOTH cases, plot the trajectory overlaid on the function AND the loss over iterations.
    
    # Q9: Non-Convex Univariate Problem
    # TODO: Define your non-convex function f(x) and its gradient g(x).
    # Constraints: 
    #   1. Must have a local min that is NOT the global min.
    #   2. Global min cannot be at -infinity.
    # Run gd from two different start points (one leading to local min, one to global min).
    # Plot the trajectory overlaid on the function AND the loss over iterations for BOTH runs.
    
    # Q10: Multidimensional Extension
    # TODO: Define f(x) = ||x||^2 and its gradient g(x) = 2x.
    # Run gd from starting point np.array([10.0, 5.0]) for 100 iterations with step size 0.2.
    # Plot the loss over iterations.

def part3():
    print("PART 3: LINEAR CLASSIFIERS")

    # Q11: Geometric Understanding
    # A simple 2D toy dataset using clusters (blobs)
    np.random.seed(42)
    X_neg = np.random.randn(20, 2) * 0.45 + np.array([-1, -1])
    X_pos = np.random.randn(20, 2) * 0.45 + np.array([1, 1])
    X_toy = np.vstack([X_neg, X_pos])
    Y_toy = np.concatenate([-np.ones(20), np.ones(20)])
    
    # TODO: Q11 - Instantiate LinearClassifierHardcoded and plot the decision boundaries
    # 1. Base Case: w=[1, 1], b=0
    # 2. Shift towards +1 class (change bias, keep w=[1,1])
    # 3. Shift towards -1 class (change bias, keep w=[1,1])
    # 4. Rotation (change weights, keep b=0)
    # Use util.plot_decision_boundary(X_toy, Y_toy, classifier, title) for each.

    # Q12: Implementing Loss Functions
    # TODO: Train on datasets.TwoDDiagonal using Squared, Logistic, Hinge, and Exp loss.
    # Record and report train/test accuracies in your PDF (no plot required here).

    # Q13: Real-world Data & Word Analysis    
    # TODO: Q13 - Train and evaluate all four loss functions on WineDataBinary.
    # 1. Print the train and test accuracy for each loss function.
    # 2. Find the best performing model.
    # 3. Extract and print the top 5 positive words (highest weights) and 
    #    top 5 negative words (lowest/most negative weights).


if __name__ == "__main__":
    # You can comment/uncomment these out to run specific parts
    part1()
    # part2()
    # part3()
