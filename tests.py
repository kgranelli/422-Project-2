import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import multiclass
import gd
import linear
import datasets


class TestPart1Multiclass(unittest.TestCase):
    def test_ova_depth1_tiny(self):
        print("\n[Part 1] Testing OVA depth 1 on TinyWineData...")
        h = multiclass.OVA(5, lambda: DecisionTreeClassifier(max_depth=1, random_state=42))
        h.train(datasets.TinyWineData.X, datasets.TinyWineData.Y)
        p = h.predict_all(datasets.TinyWineData.Xte)
        acc = np.mean(p == datasets.TinyWineData.Yte)
        # Check if accuracy is roughly 54%
        self.assertAlmostEqual(acc, 0.54, delta=0.02)

    def test_ova_depth3_tiny(self):
        print("\n[Part 1] Testing OVA depth 3 on TinyWineData...")
        h = multiclass.OVA(5, lambda: DecisionTreeClassifier(max_depth=3, random_state=42))
        h.train(datasets.TinyWineData.X, datasets.TinyWineData.Y)
        p = h.predict_all(datasets.TinyWineData.Xte)
        acc = np.mean(p == datasets.TinyWineData.Yte)
        # Check if accuracy is roughly 60%
        self.assertAlmostEqual(acc, 0.60, delta=0.02)

    def test_ava_depth1_tiny(self):
        print("\n[Part 1] Testing AVA depth 1 on TinyWineData...")
        h = multiclass.AVA(5, lambda: DecisionTreeClassifier(max_depth=1, random_state=42))
        h.train(datasets.TinyWineData.X, datasets.TinyWineData.Y)
        p = h.predict_all(datasets.TinyWineData.Xte)
        acc = np.mean(p == datasets.TinyWineData.Yte)
        # Check if accuracy is roughly 52%
        self.assertAlmostEqual(acc, 0.52, delta=0.02)

    def test_ava_depth3_tiny(self):
        print("\n[Part 1] Testing AVA depth 3 on TinyWineData...")
        h = multiclass.AVA(5, lambda: DecisionTreeClassifier(max_depth=3, random_state=42))
        h.train(datasets.TinyWineData.X, datasets.TinyWineData.Y)
        p = h.predict_all(datasets.TinyWineData.Xte)
        acc = np.mean(p == datasets.TinyWineData.Yte)
        # Check if accuracy is roughly 55%
        self.assertAlmostEqual(acc, 0.55, delta=0.02)

    def test_mctree_depth3_tiny(self):
        print("\n[Part 1] Testing MCTree depth 3 on TinyWineData...")
        t = multiclass.make_balanced_tree(range(5))
        h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3, random_state=42))
        h.train(datasets.TinyWineData.X, datasets.TinyWineData.Y)
        p = h.predict_all(datasets.TinyWineData.Xte)
        acc = np.mean(p == datasets.TinyWineData.Yte)
        # Check if accuracy is roughly 56%
        self.assertAlmostEqual(acc, 0.56, delta=0.02)


class TestPart2GD(unittest.TestCase):
    def test_gd_simple_convex(self):
        print("\n[Part 2] Testing GD on x^2...")
        func = lambda x: x**2
        grad = lambda x: 2*x
        # Starting at 10, expecting to get near 0
        x, traj = gd.gd(func, grad, 10.0, 100, 0.2)
        self.assertAlmostEqual(float(x), 0.0, delta=0.1)

    def test_gd_multidimensional(self):
        print("\n[Part 2] Testing GD on ||x||^2...")
        func = lambda x: np.sum(x**2)
        grad = lambda x: 2*x
        x0 = np.array([10.0, 5.0])
        x, traj = gd.gd(func, grad, x0, 100, 0.2)
        self.assertTrue(np.all(np.abs(x) < 0.1))

class TestPart3Linear(unittest.TestCase):
    def setUp(self):
        # Simple linearly separable 2D data
        self.X = np.array([[1, 2], [2, 1], [-1, -2], [-2, -1]])
        self.Y = np.array([1, 1, -1, -1])

    def test_logistic_loss(self):
        print("\n[Part 3] Testing LogisticLoss...")
        loss_fn = linear.LogisticLoss()
        opts = {'lossFunction': loss_fn, 'lambda': 0.1, 'numIter': 100, 'stepSize': 0.1}
        clf = linear.LinearClassifier(opts)
        clf.train(self.X, self.Y)
        preds = clf.predict(self.X)
        self.assertTrue(np.array_equal(preds, self.Y))

    def test_hinge_loss(self):
        print("\n[Part 3] Testing HingeLoss...")
        loss_fn = linear.HingeLoss()
        opts = {'lossFunction': loss_fn, 'lambda': 0.1, 'numIter': 100, 'stepSize': 0.1}
        clf = linear.LinearClassifier(opts)
        clf.train(self.X, self.Y)
        preds = clf.predict(self.X)
        self.assertTrue(np.array_equal(preds, self.Y))

    def test_exp_loss(self):
        print("\n[Part 3] Testing ExpLoss...")
        loss_fn = linear.ExpLoss()
        opts = {'lossFunction': loss_fn, 'lambda': 0.1, 'numIter': 100, 'stepSize': 0.1}
        clf = linear.LinearClassifier(opts)
        clf.train(self.X, self.Y)
        preds = clf.predict(self.X)
        self.assertTrue(np.array_equal(preds, self.Y))

if __name__ == '__main__':
    # Use verbosity 2 for cleaner output in a classroom/grading setting
    unittest.main(verbosity=2)
