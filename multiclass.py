import numpy as np


class OVA:
    def __init__(self, k_classes, classifier_function):
        self.k_classes = k_classes
        self.f = [classifier_function() for _ in range(k_classes)]

    def train(self, x, y):
        for k in range(self.k_classes):
            # print(f'Training classifier for {k} versus rest')
            # TODO: implement this
            pass

    def predict(self, x, use_zero_one=False):
        """
        Prediction is given to you. Notice that for zero-one voting,
        we just ensemble each classifier's vote. Otherwise, we
        do a weighted average according to probability.
        """
        # Initialize voting array (how many votes for each class)
        vote = np.zeros(self.k_classes)
        x_reshaped = x.reshape(1, -1)
        for k in range(self.k_classes):
            # Get the probability of each class accroding to all classifiers
            probs = self.f[k].predict_proba(x_reshaped)
            # Isolate the positive class (probs of +1 and -1 class sum up to 1)
            prob_positive = probs[0, 1]
            if use_zero_one:
                # If P[this_class] > 0.5, add 1 vote, otherwise don't add vote
                vote[k] += 1 if prob_positive > 0.5 else 0
            else:
                # If we don't do zero-one, just add P[this_class]
                vote[k] += prob_positive
        return np.argmax(vote)

    def predict_all(self, x, use_zero_one=False):
        return np.array([self.predict(row, use_zero_one) for row in x])


class AVA:
    def __init__(self, k_classes, classifier_function):
        self.k_classes = k_classes
        self.f = [[classifier_function() for _ in range(i)] for i in range(k_classes)]

    def train(self, x, y):
        for i in range(self.k_classes):
            for j in range(i):
                # print(f'Training classifier for {i} versus {j}')
                # TODO: implement this
                pass

    def predict(self, x, use_zero_one=False):
        """
        Should be the same as OVA.
        """
        pass

    def predict_all(self, x, use_zero_one=False):
        return np.array([self.predict(row, use_zero_one) for row in x])


class TreeNode:
    def __init__(self):
        self._is_leaf = True
        self.label = 0
        self.info = None
        self.left = None
        self.right = None

    def set_leaf_label(self, label):
        self._is_leaf = True
        self.label = label

    def set_children(self, left, right):
        self._is_leaf = False
        self.left = left
        self.right = right

    def is_leaf(self):
        return self._is_leaf
    
    def get_label(self):
        if self._is_leaf:
            return self.label
        raise Exception("called get_label on an internal node!")
        
    def get_left(self):
        if self._is_leaf:
            raise Exception("called get_left on a leaf!")
        return self.left
        
    def get_right(self):
        if self._is_leaf:
            raise Exception("called get_right on a leaf!")
        return self.right

    def set_node_info(self, info):
        self.info = info

    def get_node_info(self):
        return self.info

    def iter_all_labels(self):
        if self._is_leaf:
            yield self.label
        else:
            yield from self.left.iter_all_labels()
            yield from self.right.iter_all_labels()

    def iter_nodes(self):
        yield self
        if not self._is_leaf:
            yield from self.left.iter_nodes()
            yield from self.right.iter_nodes()

    def __repr__(self):
        return str(self.label) if self._is_leaf else f"[{repr(self.left)} {repr(self.right)}]"


def make_balanced_tree(all_k):
    if not all_k:
        raise Exception("make_balanced_tree: cannot make a tree of 0 classes")

    tree = TreeNode()
    if len(all_k) == 1:
        tree.set_leaf_label(all_k[0])
    else:
        split = len(all_k) // 2
        tree.set_children(
            make_balanced_tree(all_k[:split]),
            make_balanced_tree(all_k[split:])
        )
    return tree


class MCTree:
    def __init__(self, tree, classifier_function):
        self.tree = tree
        for n in self.tree.iter_nodes():
            n.set_node_info(classifier_function())

    def train(self, x, y):
        for n in self.tree.iter_nodes():
            if n.is_leaf():
                continue

            left_labels = set(n.get_left().iter_all_labels())
            right_labels = set(n.get_right().iter_all_labels())

            # print(f'Training classifier for {list(left_labels)} versus {list(right_labels)}')

            # TODO: implement this
            indices = ...
            this_x = ...
            this_y = ...

            classifier = n.get_node_info()
            classifier.fit(this_x, this_y)

    def predict(self, x):
        curr_node = self.tree
        # TODO: implement this
        pass

    def predict_all(self, x):
        # TODO: implement this
        pass
