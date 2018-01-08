import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from Utils.MyParser import MyParser


class FeatureHeadAndModifierPosClassDistanceBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        self.parser = parser
        tpls = sorted(set(parser.getHeadAndModifierPosClassAndDistance()))
        super().__init__(len(tpls), offset)
        self.d = {}
        for index in range(0, self.size):
            self.d[tpls[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        modifier_pos_class = self.parser.getPosClass(history.tokens[modifier].pos)
        head_pos_class = self.parser.getPosClass(history.tokens[head].pos)
        tpl = head_pos_class, modifier_pos_class, head - modifier
        if tpl in self.d:
            return np.array([self.d[tpl]])
        return np.array([])
