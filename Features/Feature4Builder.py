import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser


class Feature4Builder(FeatureBuilderBase):
    def __init__(self, parser : MyParser, offset) -> None:
        modifier_tuples = parser.getTupleOfPosAndWordFromModifierTokens()
        super().__init__(len(modifier_tuples), offset)
        self.modifier_tuples = {}
        for index in range(0, self.size):
            self.modifier_tuples[modifier_tuples[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        modifier_token = history[modifier]
        tpl = modifier_token.token, modifier_token.pos
        if tpl in self.modifier_tuples:
            return np.array([self.modifier_tuples[tpl]])
        return np.array([])
