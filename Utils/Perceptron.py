from itertools import chain

import numpy as np

from Utils.Gen import Gen
from Utils.History import History
from Utils.chu_liu import Digraph


class Perceptron():
    def __init__(self, featureVectorBuilder):
        self.featureVectorBuilder = featureVectorBuilder

    def perceptron(self, N, tuples: (History, list)):
        w = np.zeros(self.featureVectorBuilder.size)
        for n in range(N):
            for history, tree in tuples:
                opt_tree = self._historyToOptParseTree(history, w)
                if not self._isTreeEq(opt_tree, tree):
                    w = self.update_weights(history, tree, opt_tree, w)
        return w

    def _historyToOptParseTree(self, history: History, w):
        successors = {}
        N = len(history.tokens)
        for i in range(N):
            successors[i] = [idx for idx in range(1, N) if idx != i]
        get_score = lambda i, j: np.sum(w[self.featureVectorBuilder.getFeatureVector(history, i, j)])
        graph = Digraph(successors, get_score)
        return self._mstToTuples(graph.mst().successors)

    def _mstToTuples(self, successors):
        res = [[(h, m) for m in successors[h]] for h in range(len(successors))]
        res = list(chain.from_iterable(res))
        return res

    def update_weights(self, history, tree, opt_tree, w):
        for (i, j) in tree:
            w[self.featureVectorBuilder.getFeatureVector(history, i, j)] += 1
        for (i, j) in opt_tree:
            w[self.featureVectorBuilder.getFeatureVector(history, i, j)] -= 1
        return w

    def _isTreeEq(self, t1, t2):
        return set(t1) == set(t2)
