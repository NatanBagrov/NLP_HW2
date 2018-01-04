import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from Utils.MyParser import MyParser


class Feature1Builder(FeatureBuilderBase):
    def __init__(self, parser : MyParser, offset) -> None:
        head_tuples = parser.getTupleOfPosAndWordFromHeadTokens()
        super().__init__(len(head_tuples), offset)
        self.head_tuples = {}
        for index in range(0, self.size):
            self.head_tuples[head_tuples[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        head_token = history.tokens[head]
        tpl = head_token.token, head_token.pos
        if tpl in self.head_tuples:
            return np.array([self.head_tuples[tpl]])
        return np.array([])
