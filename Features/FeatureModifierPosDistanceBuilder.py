import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from Utils.MyParser import MyParser


class FeatureModifierPosDistanceBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        tokens = parser.getHeadAndModifierTokensAndDistance()
        modifiers = sorted(set([(m,d) for (h,m,d) in tokens]))
        super().__init__(len(modifiers), offset)
        self.modifiersPosAndDistance = {}
        for index in range(0, self.size):
            self.modifiersPosAndDistance[modifiers[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        modifier_pos = history.tokens[modifier].pos
        distance = head - modifier
        tpl = (modifier_pos, distance)
        if tpl in self.modifiersPosAndDistance:
            return np.array([self.modifiersPosAndDistance[tpl]])
        return np.array([])