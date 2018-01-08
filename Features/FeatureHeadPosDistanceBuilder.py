import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from Utils.MyParser import MyParser


class FeatureHeadPosDistanceBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        tokens = parser.getHeadAndModifierTokensAndDistance()
        heads = sorted(set([(h,d) for (h,m,d) in tokens]))
        super().__init__(len(heads), offset)
        self.headsPosAndDistance = {}
        for index in range(0, self.size):
            self.headsPosAndDistance[heads[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        head_pos = history.tokens[head].pos
        distance = head - modifier
        tpl = (head_pos, distance)
        if tpl in self.headsPosAndDistance:
            return np.array([self.headsPosAndDistance[tpl]])
        return np.array([])