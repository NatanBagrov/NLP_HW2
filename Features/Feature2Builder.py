import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser


class Feature2Builder(FeatureBuilderBase):
    def __init__(self, parser : MyParser, offset) -> None:
        heads = parser.getHeadWordsFromHeadTokens()
        super().__init__(len(heads), offset)
        self.heads = {}
        for index in range(0, self.size):
            self.heads[heads[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        head_token = history[head]
        head_word = head_token.token
        if head_word in self.heads:
            return np.array([self.heads[head_word]])
        return np.array([])