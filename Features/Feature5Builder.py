import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser


class Feature5Builder(FeatureBuilderBase):
    def __init__(self, parser : MyParser, offset) -> None:
        modifiers = parser.getModifierWordsFromModifierTokens()
        super().__init__(len(modifiers), offset)
        self.modifiers = {}
        for index in range(0, self.size):
            self.modifiers[modifiers[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        modifier_token = history[modifier]
        modifier_word = modifier_token.token
        if modifier_word in self.modifiers:
            return np.array([self.modifiers[modifier_word]])
        return np.array([])