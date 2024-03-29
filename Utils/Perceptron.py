import random
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
        start = time.time()
        if old_w is None:
            w = np.zeros(self.featureVectorBuilder.size)
        for n in range(N):
            tmpFileName = "perceptron_verycomplex_iteration_" + str(n) + "_weights.txt"
            np.savetxt(tmpFileName, w)
            random.shuffle(tuples)
            for history, tree in tuples:
                # sentenceStart = time.time()
                opt_tree = self.tagger.historyToOptParseTree(self.featureVectorBuilder, history, w)
                # print((time.time() - sentenceStart), "seconds took for sentence")
                if not self._isTreeEq(opt_tree, tree):
                    w = self.update_weights(history, tree, opt_tree, w)
            print("Finished Iteration #", n, "took: ", (time.time() - start)/60,"minutes total")
        return w

    def update_weights(self, history, tree, opt_tree, w):
        tree = set(tree)
        opt_tree = set(opt_tree)
        intersection = tree.intersection(opt_tree)
        to_add = (tree - intersection)
        to_remove = (opt_tree - intersection)
        for (i, j) in to_add:
            vector = self.featureVectorBuilder.getFeatureVector(history, i, j)
            w[vector] += 1
        for (i, j) in to_remove:
            vector = self.featureVectorBuilder.getFeatureVector(history, i, j)
            w[vector] -= 1
        return w

    def _isTreeEq(self, t1, t2):
        return set(t1) == set(t2)
