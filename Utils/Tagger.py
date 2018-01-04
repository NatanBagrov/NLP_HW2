from itertools import chain

import numpy as np

from Utils.History import History
from Utils.chu_liu import Digraph


class Tagger:

    def historyToOptParseTree(self, featureVectorBuilder, history: History, w):
        successors = {}
        N = len(history.tokens)
        for i in range(N):
            successors[i] = [idx for idx in range(1, N) if idx != i]
        get_score = lambda i, j: np.sum(w[featureVectorBuilder.getFeatureVector(history, i, j)])
        graph = Digraph(successors, get_score)
        return self._mstToTuples(graph.mst().successors)

    def _mstToTuples(self, successors):
        res = [[(h, m) for m in successors[h]] for h in range(len(successors))]
        res = list(chain.from_iterable(res))
        return res

    def compareTrees(self, t1, t2):
        t1, t2 = set(t1), set(t2)
        assert len(t1)==len(t2)
        size = len(t1)
        return len(t1.intersection(t2))

