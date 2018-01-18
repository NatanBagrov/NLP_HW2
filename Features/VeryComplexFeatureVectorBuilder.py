import numpy as np

from Features.ComplexFeatureVectorBuilder import ComplexFeatureVectorBuilder
from Features.FeatureBuilderBase import FeatureBuilderBase
from Features.FeaturePosInBetweenBuilder import FeaturePosInBetweenBuilder
from Features.FeaturePosNeighborsBuilder import FeaturePosNeighborsBuilder
from Utils.MyParser import MyParser


class VeryComplexFeatureVectorBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        self.parser = parser
        self.fComplex = ComplexFeatureVectorBuilder(parser, 0)
        size = self.fComplex.size
        self.fInbetween = FeaturePosInBetweenBuilder(parser, size)
        size += self.fInbetween.size
        print("fInbetween interval = ", size - self.fInbetween.size, size)
        self.fNeighbors = FeaturePosNeighborsBuilder(parser, size)
        size += self.fNeighbors.size
        print("fNeighbors interval = ", size - self.fNeighbors.size, size)
        super().__init__(size, offset)
        print("VeryComplex Total size:", self.size)

    def getFeatureVector(self, history, head, modifier):
        vecComplex = self.fComplex.getFeatureVector(history, head, modifier)
        myVec = np.concatenate((self.fInbetween.getFeatureVector(history, head, modifier),
                               self.fNeighbors.getFeatureVector(history, head, modifier)))
        return np.concatenate((vecComplex, myVec)).astype(int)
