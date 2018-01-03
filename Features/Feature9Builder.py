import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from MyParser import MyParser


class Feature9Builder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        tuples = parser.getTupleOfHeadAndModifier()
        tuples = sorted(list(set([(h.token, m.token, m.pos) for (h, m) in tuples])))
        super().__init__(len(tuples), offset)
        self.h_m_tuples = {}
        for index in range(0, self.size):
            self.h_m_tuples[tuples[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        head_token = history[head]
        modifier_token = history[modifier]
        tpl = head_token.token, modifier_token.token, modifier_token.pos
        if tpl in self.h_m_tuples:
            return np.array([self.h_m_tuples[tpl]])
        return np.array([])
