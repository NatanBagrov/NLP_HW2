import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from Utils.MyParser import MyParser


class FeaturePosInBetweenBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        self.parser = parser
        tpls = parser.getTupleOfHeadAndModifierAndTokensInBetween()
        res = []
        res2 = []
        res3 = []
        for h, m, l in tpls:
            for token in l:
                res.append((h.pos, m.pos, token.pos))
                res2.append((h.pos, m.pos, token.pos, h.idx - m.idx))
                res3.append((self.parser.getPosClass(h.pos), self.parser.getPosClass(m.pos),
                             self.parser.getPosClass(token.pos)))
        tpls1 = sorted(set(res))
        tpls2 = sorted(set(res2))
        tpls3 = sorted(set(res3))
        tpls = sorted(set(tpls1 + tpls2 + tpls3))
        super().__init__(len(tpls), offset)
        self.d = {}
        for index in range(0, self.size):
            self.d[tpls[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        start, end = min(head, modifier) + 1, max(head, modifier)
        inbetween = [token.pos for token in history.tokens[start:end]]
        modifier_pos = history.tokens[modifier].pos
        head_pos = history.tokens[head].pos
        dist = head - modifier
        tpls = [(head_pos, modifier_pos, pos, dist) for pos in inbetween]
        res = []
        for h, m, b, d in tpls:
            if (h, m, b, d) in self.d:
                res.append(self.d[(h, m, b, d)])
            if (h, m, b) in self.d:
                res.append(self.d[(h, m, b)])
            tpl = (self.parser.getPosClass(h), self.parser.getPosClass(m), self.parser.getPosClass(b))
            if tpl in self.d:
                res.append(self.d[tpl])
        return np.array(res)
