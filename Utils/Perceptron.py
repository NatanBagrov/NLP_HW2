import time
import numpy as np
from Utils.History import History
from Utils.Tagger import Tagger


class Perceptron():
    def __init__(self, featureVectorBuilder, tagger: Tagger):
        self.featureVectorBuilder = featureVectorBuilder
        self.tagger = tagger

    def perceptron(self, N, tuples: (History, list), old_w=None):
        w = old_w
        if old_w is None:
            w = np.zeros(self.featureVectorBuilder.size)
        for n in range(N):
            for history, tree in tuples:
                start = time.time()
                opt_tree = self.tagger.historyToOptParseTree(self.featureVectorBuilder, history, w)
                if not self._isTreeEq(opt_tree, tree):
                    w = self.update_weights(history, tree, opt_tree, w)
            print("Finished Iteration #", n)
        return w

    def update_weights(self, history, tree, opt_tree, w):
        for (i, j) in tree:
            w[self.featureVectorBuilder.getFeatureVector(history, i, j)] += 1
        for (i, j) in opt_tree:
            w[self.featureVectorBuilder.getFeatureVector(history, i, j)] -= 1
        return w

    def _isTreeEq(self, t1, t2):
        return set(t1) == set(t2)
