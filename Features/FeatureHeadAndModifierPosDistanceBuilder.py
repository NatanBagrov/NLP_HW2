import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from Utils.MyParser import MyParser


class FeatureHeadAndModifierPosDistanceBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        tokens = parser.getHeadAndModifierTokensAndDistance()
        tpl = sorted(set([(h, m, d) for (h, m, d) in tokens]))
        super().__init__(len(tpl), offset)
        self.headsAndModifiersPosAndDistance = {}
        for index in range(0, self.size):
            self.headsAndModifiersPosAndDistance[tpl[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        head_pos = history.tokens[head].pos
        modifier_pos = history.tokens[modifier].pos
        distance = head - modifier
        tpl = (head_pos, modifier_pos, distance)
        if tpl in self.headsAndModifiersPosAndDistance:
            return np.array([self.headsAndModifiersPosAndDistance[tpl]])
        return np.array([])
