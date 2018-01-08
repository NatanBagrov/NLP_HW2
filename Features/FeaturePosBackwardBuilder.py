import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from Utils.MyParser import MyParser


class FeaturePosBackwardBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        pos = parser.getModifierPosFromModifierToken()
        super().__init__(len(pos), offset)
        self.pos = {}
        for index in range(0, self.size):
            self.pos[pos[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        modifier_token = history.tokens[modifier]
        modifier_pos = modifier_token.pos
        if head > modifier and modifier_pos in self.pos:
            return np.array([self.pos[modifier_pos]])
        return np.array([])