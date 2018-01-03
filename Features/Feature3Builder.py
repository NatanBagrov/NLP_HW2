import numpy as np

from Features.FeatureBuilderBase import FeatureBuilderBase
from Utils.MyParser import MyParser


class Feature3Builder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        pos = parser.getHeadPosFromHeadToken()
        super().__init__(len(pos), offset)
        self.pos = {}
        for index in range(0, self.size):
            self.pos[pos[index]] = index + self.offset

    def getFeatureVector(self, history, head, modifier):
        head_token = history[head]
        head_pos = head_token.pos
        if head_pos in self.pos:
            return np.array([self.pos[head_pos]])
        return np.array([])