import numpy as np

from Features.ComplexFeatureVectorBuilder import ComplexFeatureVectorBuilder
from Features.Feature11Builder import Feature11Builder
from Features.Feature12Builder import Feature12Builder
from Features.Feature1Builder import Feature1Builder
from Features.Feature2Builder import Feature2Builder
from Features.Feature3Builder import Feature3Builder
from Features.Feature4Builder import Feature4Builder
from Features.Feature5Builder import Feature5Builder
from Features.Feature6Builder import Feature6Builder
from Features.Feature7Builder import Feature7Builder
from Features.Feature8Builder import Feature8Builder
from Features.Feature10Builder import Feature10Builder
from Features.Feature13Builder import Feature13Builder
from Features.Feature9Builder import Feature9Builder
from Features.FeatureBuilderBase import FeatureBuilderBase
from Features.FeatureHeadAndModifierPosClassBuilder import FeatureHeadAndModifierPosClassBuilder
from Features.FeatureHeadAndModifierPosClassDistanceBuilder import FeatureHeadAndModifierPosClassDistanceBuilder
from Features.FeatureHeadAndModifierPosDistanceBuilder import FeatureHeadAndModifierPosDistanceBuilder
from Features.FeatureHeadPosDistanceBuilder import FeatureHeadPosDistanceBuilder
from Features.FeatureModifierPosDistanceBuilder import FeatureModifierPosDistanceBuilder
from Features.FeaturePosBackwardBuilder import FeaturePosBackwardBuilder
from Features.FeaturePosForwardBuilder import FeaturePosForwardBuilder
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
        self.fNeighbors = FeaturePosNeighborsBuilder(parser, size)
        size += self.fNeighbors.size
        super().__init__(size, offset)

    def getFeatureVector(self, history, head, modifier):
        vecComplex = self.fComplex.getFeatureVector(history, head, modifier)
        myVec = np.concatenate((self.fInbetween.getFeatureVector(history, head, modifier),
                               self.fNeighbors.getFeatureVector(history, head, modifier)))
        return np.concatenate((vecComplex, myVec)).astype(int)
