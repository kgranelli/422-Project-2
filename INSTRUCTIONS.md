# Project 2: Multiclass and Linear Models

## Introduction and Setup

There are two parts to this project. The first is to play around with multiclass reductions. The second is to play around with linear models. The project files are available for download in the p2 folder on the course [Files](https://umd.instructure.com/courses/1398557/files/folder/p2 "Files Index") page. 

You will work on this project **individually**.

You can use the python environment you created for Project 1 since it already contains all the necessary packages. You may also reuse plotting code (figure, plot, x/ylabel, title, savefig) from Project 1.

Please **do not** change the names of any provided functions or classes within the code. The correctness of your implementation will be the final judge of your score. Please **do not** import (potentially unsafe) system-related modules such as sys, exec, eval.


### What and how to Submit

You will submit the assignment individually on Gradescope.
* **Upload the python files you will edit**, namely `multiclass.py`, `gd.py`, `linear.py`, as well as `main.py` (so all python files we gave you except `tests.py`, `util.py`, and `datasets.py`, which you will not need to change). Please note that even if the code passes the tests from `tests.py` this is *not* a gurantee for its corectness.
* **Include a PDF file named `REPORT.pdf`** that answers all the written questions in the assignment marked by "Q#" below.

Note that the file `main.py` is a scaffold to help you organize your code. It is used to generate your numbers and plots. 


## Multiclass Classification [50 pts total]

### Data Familiarization

We are evaluating a text classification task: wine classification. Your job is, given the description of the wine, predict the type (class) of wine. The main dataset `WineData` has 20 different wines, while `TinyWineData` is just the first five of those and is just used in the test script.

You can find the names of the wines both in `WineData.labels` as well as the file `wines.names`. Before jumping into classification, let's understand the dataset. 

**Q1 [3 pts]:** Look at the datasets in `datasets.py` and write code (in `main.py`) to answer the following:
(A) How many training examples are in `WineData`? How many test examples?
(B) How many features (words) does each example have?
(C) What is the most frequent wine class in the training set?

### Binary Classification under the hood

First, we are going to work with One-Versus-All (OVA) and All-Versus-All (AVA) approaches to mutliclass classification using binary classifiers under the hood (we will use `DecisionTreeClassifier` imported from sklearn). You should familiarize yourself with Section 6.2 of the textbook, and pay attention to Algorithms 13-16. Then **proceed to implement the two multiclass classification paradigms (classes `OVA` and `AVA`) in `multiclass.py` [10 pts]**.

After implementing them, here's how we can train decision "stumps" (aka `max_depth=1` decision trees) on the large data set using OVA:
```python
h = multiclass.OVA(20, lambda: DecisionTreeClassifier(max_depth=1))
h.train(datasets.WineData.X, datasets.WineData.Y)
p = h.predict_all(datasets.WineData.Xte)
accuracy = np.mean(p == datasets.WineData.Yte) # around 0.29
```

The default implementation of OVA uses decision tree confidence (probability of prediction) to weigh the votes. You can switch to zero/one predictions to see the effect:
```python
p = h.predict_all(datasets.WineData.Xte, use_zero_one=True)
accuracy = np.mean(p == datasets.WineData.Yte) # around 0.19
```
You can observe that this is **markedly worse**.


**Q2 [10 pts]:**
(A) For OVA, train depth 3 decision trees on the `WineData` task. Using the `feature_importances_` attribute of the trained scikit-learn `DecisionTreeClassifier` objects, extract the top 3 most important features (words) used for predicting Sauvignon-Blanc (label 0). Do the same for Pinot-Noir (label 2).
(B) For both OVA and AVA, train depth 3 decision trees on the `WineData` task (with 20 labels). What accuracy do you get? How long does this take (in seconds)?
(C) For both OVA and AVA, compare the accuracy using zero-one predictions versus using confidence on the `WineData` task. How much difference does it make?

**Q3 [2 pts]:** Which one of the OVA or AVA paradigms would you prefer for multiclass classification using binary classifiers? Think about discriminative ability and efficiency. What if you have a large dataset like ImageNet with 1000 classes?


### Classification Tree

Now, you must implement a tree-based reduction (classification tree) in the `MCTree` class, like the one seen in Figure 6.2 of your textbook. Some of the training logic in `train` is given to you, and **you need to fill out the splitting logic. You also need to implement `predict` and `predict_all` on your own [15 pts]**.

A tree class is provided to help you. This class implements the following basic functions:

\>>\> t \= multiclass.makeBalancedTree(range(6)) \
\>>\> t \
\[\[0 \[1 2\]\] \[3 \[4 5\]\]\] \
\>>\> t.is_leaf \
False \
\>>\> t.getLeft() \
\[0 \[1 2\]\] \
\>>\> t.getLeft().getLeft() \
0

Here is an example using `TinyWineData` (also included in test.py).
```python
t = multiclass.make_balanced_tree(range(5))  
h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3))  
h.train(datasets.TinyWineData.X, datasets.TinyWineData.Y)  
p = h.predict_all(datasets.TinyWineData.Xte)  
accuracy = np.mean(p == datasets.TinyWineData.Yte) # around 0.56
```

**Q4 [5 pts]:** Show the test accuracy you get with a balanced tree on `WineData` using a `DecisionTreeClassifier` with `max_depth=3`.

**Q5 [5 pts]:** In this case, the underlying classifier in each node is a decision tree. Plot the training and test accuracy as a function of the underlying classifier's complexity (vary the tree's `max_depth` from 1 to a larger number). What do you observe happening, especially in the edge cases (1 vs the largest depth you picked)?

## Gradient Descent [25 pts total]

To get started with linear models, we will implement a generic gradient descent method. 

Before you write any code, let's understand how this works. Suppose we want to minimize the simple function $f(x) = x^2$. The gradient (derivative) of this function is $f'(x) = 2x$. If we start at an initial position $x_0 = 10$, and choose an initial step size $\eta = 0.2$, our first step of gradient descent updates our position by moving in the opposite direction of the gradient:
$$x_1 = x_0 - \eta \cdot f'(x_0) = 10 - 0.2 \cdot (2 \cdot 10) = 10 - 4 = 6$$
In just one iteration, our objective function value drops from $10^2 = 100$ down to $6^2 = 36$. We repeat this process until convergence.

You will write your implementation in `gd.py`, which should contain a single function: `gd`. In each iteration of gradient descent, compute the gradient and take a step in the opposite direction. You must implement an adaptive step size, where the step size $\eta$ at iteration $t$ (counting from $1$) is computed as:

$$\eta = \frac{\text{"initial step size"}}{\sqrt{t}}$$

The function should return the final optimized value of $x$ and the trajectory (the history of $x$ values across all iterations as a NumPy array).

### Q6: Implementation & Function Evaluation [5 pts]
Implement the `gd` function in `gd.py`. Verify its correctness by running the provided test suite (`python tests.py`). Do not move on until your gradient descent implementation passes the basic tests. **Note that you will be using standard gradient descent** (i.e., using the exact gradient, not stochastic or mini-batch).

Once your tests pass, write code in `main.py` to minimize the simple convex function $f(x) = x^2$ with gradient $2x$ using your `gd` algorithm. Use a starting position of 100 iterations, and an initial step size of 0.2.

**Deliverable:** Generate two side-by-side plots: (1) The trajectory of $x$ overlaying the function curve, and (2) The loss (objective function value) over the iterations (with iterations on the x-axis). Include these plots in your report.

### Q7: Step Size Analysis (Positive) [5 pts]
What is the exact impact of the step size on convergence? Using the $x^2$ minimization from Q6, experiment with different strictly **positive** initial step sizes. 
* **Deliverable:** Find and report a specific value of the positive initial step size where the algorithm successfully converges, and a specific value where it **oscillates and diverges**. For *each* of these two cases, generate two plots: (1) the optimization trajectory overlaid on the function curve, and (2) the loss over iterations. Include all four of these plots in your report.

### Q8: Step Size Analysis (Negative) [5 pts]
What happens if you pick a negative step size? Explain what the algorithm does in this case and why this completely breaks the optimization process. 
* **Note:** You do not need to provide plots for this question; provide a conceptual explanation in your report.

### Q9: Non-Convex Univariate Problem [5 pts]
Not all functions are as well-behaved as $f(x) = x^2$. In this question, we will explore what happens when we use a non-convex function.

* **Deliverable:** Define a non-convex univariate function $f(x)$ of your choosing and its derivative $f'(x)$. Your function must satisfy the following constraints:
  1. It must have at least one local minimum that is *not* the global minimum.
  2. The global minimum cannot be at $-\infty$.
  *(Note: Be mindful of [saddle points](https://en.wikipedia.org/wiki/Saddle_point) where the gradient is zero but the point is neither a local nor a global minimum!)*
* Run your `gd` algorithm from two different starting points:
  1. A starting point where the algorithm gets trapped in the local minimum.
  2. A starting point where the algorithm successfully finds the global minimum.
* **Plots:** For *both* starting points, generate and include the following two plots (four plots total):
  1. The optimization trajectory overlaid on the function curve.
  2. The loss over iterations.
* **Analysis:** Include these plots and the mathematical definition of your $f(x)$ in your report. Explain why each starting point led to its respective outcome.

### Q10: Multi-Dimensional Extension [5 pts]
If you implemented your function cleanly using `numpy`, it should work natively in multiple dimensions. Test this by minimizing the squared L2 norm, $f(x) = \|x\|^2$. 
* **Setup:** Use the squared L2 norm as your objective function and $2x$ as its gradient.
* **Parameters:** Use a starting array of `[10, 5]`. Run the algorithm for 100 iterations with an initial step size of 0.2.
* **Deliverable:** Report your final output array and provide the plot of the loss over iterations showing convergence. Ensure your code passes `tests.py`.

## Linear Classifiers [25 pts total]

### Q11: Geometric Understanding of Linear Models [5 pts]
Before implementing the loss functions, let's understand how a linear classifier makes decisions geometrically. In a 2-dimensional feature space, a linear classifier predicts based on the sign of $w \cdot x + b$. The decision boundary is the line where $w \cdot x + b = 0$.
* **Deliverable (Code):** In `linear.py`, implement the class `LinearClassifierHardcoded` which takes hardcoded weights `w` and bias `b` upon initialization and uses them to predict, without any training.
* **Deliverable (Plots):** In `main.py`, we have provided a simple 2D toy dataset `X_toy` (a cluster of points centered around $(-1, -1)$ labeled as $-1$, and another cluster around $(1, 1)$ labeled as $+1$). Use the provided `util.plot_decision_boundary` function to generate the following four plots:
  1. **Base Case:** Use the hardcoded weights $w = [1, 1]$ and bias $b = 0$. This separates the two clusters with a boundary line exactly down the middle.
  2. **Changing Bias (Shift Towards +1):** Keep $w$ the same as your base case, but change $b$ to a value that shifts the boundary line significantly closer to the $+1$ cluster without misclassifying any of its points.
  3. **Changing Bias (Shift Towards -1):** Keep $w$ the same as your base case, but change $b$ to a value that shifts the boundary line significantly closer to the $-1$ cluster without misclassifying any of its points.
  4. **Changing Weights (Rotation):** Keep $b$ the same as your base case, but change $w$ to a new set of weights that changes the slope/angle (rotates) the boundary line, while still correctly separating the clusters.
* **Analysis:** Include all four plots in your report. Explain exactly how changing $b$ (while keeping $w$ fixed) affects the line geometrically, and how changing $w$ affects the line.

### Q12: Implementing Loss Functions [10 pts]

Our generic linear classifier implementation is in `linear.py`. We have an interface `LossFunction` that we want to minimize. This must be able to compute the loss for a pair `Y` and `Yhat` where the former is the truth and the latter are the predictions. It must also be able to compute a gradient when additionally given the data `X`. 

There are four loss function stubs: `SquaredLoss` (which is implemented for you!), `LogisticLoss`, `HingeLoss`, and `ExpLoss` (which you will have to implement). 

The `LinearClassifier` class is a stub implementation of a generic linear classifier with an L2 regularizer. Your implementation should go in the `train` method. The idea is to pass the appropriate objective and gradient functions to your `gd` implementation and let it do the heavy lifting. See the inline comments in the code for guidance.

Implement `LogisticLoss`, `HingeLoss`, and `ExpLoss` in `linear.py`. Once you've implemented the function evaluations and gradients, run the test suite (`python tests.py`). **Do not proceed until all linear classifier tests pass.** 

**Simplification Note (No Bias):** For this implementation, we are assuming the bias term $b = 0$. You **do not** need to calculate or return gradients for a bias term; only calculate the gradient with respect to the weights $w$.

**Numerical Stability Hint:** When implementing `ExpLoss`, `np.exp()` to overflow and return `NaNs`. To prevent this, clip your exponents using `np.clip(value, -100, 100)` before passing them into `np.exp()`.

Write code in `main.py` to train your classifier with each loss function on the `datasets.TwoDDiagonal` dataset using $\lambda = 1$, (initial) stepSize $= 0.5$, and numIter $= 100$. Include the train and test accuracies for each loss function in the pdf file.

### Q13: Real-world Data & Word Analysis [10 pts]
Now let's apply your linear classifiers to a real-world text classification dataset: `WineDataBinary`. The goal is to predict whether a wine review is positive (+1) or negative (-1) based on the words used in the review.

* **Deliverable (Code):** Train a `LinearClassifier` using each of the four loss functions on the `WineDataBinary` training set. Evaluate their accuracy on the test set.  You should use `lambda=1`, `numIter=100`, and `stepSize=0.5` in all cases. 
* **Feature Analysis:** Identify which loss function gave the best test accuracy. Using the trained weights from that specific model, extract the top 5 words with the greatest positive weights, and the top 5 words with the lowest (most negative) weights. (Hint: The `WineDataBinary` dataset includes a `words` list that maps feature indices to actual English words).
* **Analysis:** In your report, provide the following:
  1. The test accuracies for all four models.
  2. The lists of the top 5 positive and top 5 negative words from your best model.
  3. **Conceptual Analysis:** What is the significance of the features (words) with the greatest positive weights and the most negative weights? How do their roles in the classifier compare to features with weights that are around 0 or very small?
