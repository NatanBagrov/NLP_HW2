import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from Utils.MyParser import MyParser


class FeaturePosNeighborsBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        self.parser = parser
        tpls = parser.getTupleOfHeadAndModifierAndNeighbors()
        res1 = [(h.pos, hp.pos, dm.pos, d.pos, h.idx - d.idx) for (hm, h, hp, dm, d, dp) in tpls]
        res2 = [(hm.pos, h.pos, dm.pos, d.pos, h.idx - d.idx) for (hm, h, hp, dm, d, dp) in tpls]
        res3 = [(h.pos, hp.pos, d.pos, dp.pos, h.idx - d.idx) for (hm, h, hp, dm, d, dp) in tpls]
        res4 = [(hm.pos, h.pos, d.pos, dp.pos, h.idx - d.idx) for (hm, h, hp, dm, d, dp) in tpls]
        tpls1 = sorted(set(res1))
        tpls2 = sorted(set(res2))
        tpls3 = sorted(set(res3))
        tpls4 = sorted(set(res4))
        tpls = tpls1 + tpls2 + tpls3 + tpls4
        super().__init__(len(tpls), offset)
        self.d = {}
        for index in range(0, self.size):
            self.d[tpls[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        hp_pos = hm_pos = mp_pos = '_'
        if head != 0:
            hp_pos = history.tokens[head - 1].pos
        size = len(history.tokens) - 1
        if head < size:
            hm_pos = history.tokens[head + 1].pos
        if modifier < size:
            mp_pos = history.tokens[modifier + 1].pos
        mm_pos = history.tokens[modifier - 1]

        m_pos = history.tokens[modifier].pos
        h_pos = history.tokens[head].pos
        d = head - modifier
        res = []

        tpl = (h_pos, hp_pos, mm_pos, m_pos, d)
        if tpl in self.d:
            res.append(self.d[tpl])
        tpl = (hm_pos, h_pos, mm_pos, m_pos, d)
        if tpl in self.d:
            res.append(self.d[tpl])
        tpl = (h_pos, hp_pos, m_pos, mp_pos, d)
        if tpl in self.d:
            res.append(self.d[tpl])
        tpl = (hm_pos, h_pos, m_pos, mp_pos, d)
        if tpl in self.d:
            res.append(self.d[tpl])
        return np.array(res)
