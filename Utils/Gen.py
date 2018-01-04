from itertools import chain

import numpy as np

from Utils import History
from Utils.chu_liu import Digraph


class Gen:
    def __init__(self, limit):
        self.edgeToRandWeightDict = {}
        self.limit = limit

    def _createRandomGraph(self, history):
        N = len(history.tokens)
        for i in range(N):
            for j in range(1, N):
                if i == j:
                    continue
                self.edgeToRandWeightDict[(i, j)] = np.random.random_sample()

    def get_score(self, i, j):
        return self.edgeToRandWeightDict[(i, j)]

    def gen(self, history: History):
        print('nope')
        succseors = {}
        N = len(history.tokens)
        for i in range(N):
            succseors[i] = [idx for idx in range(1, N) if idx != i]
        for idx in range(self.limit):
            self._createRandomGraph(history)
            graph = Digraph(succseors, self.get_score)
            mst = graph.mst()
            yield self._mstToTuples(mst.successors)

    def _mstToTuples(self, successors):
        res = [[(h, m) for m in successors[h]] for h in range(len(successors))]
        res = list(chain.from_iterable(res))
        return res
