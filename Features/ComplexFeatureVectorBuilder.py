import numpy as np

from Features.Feature10Builder import Feature10Builder
from Features.Feature11Builder import Feature11Builder
from Features.Feature12Builder import Feature12Builder
from Features.Feature13Builder import Feature13Builder
from Features.Feature1Builder import Feature1Builder
from Features.Feature2Builder import Feature2Builder
from Features.Feature3Builder import Feature3Builder
from Features.Feature7Builder import Feature7Builder
from Features.Feature8Builder import Feature8Builder
from Features.Feature9Builder import Feature9Builder
from Features.FeatureBuilderBase import FeatureBuilderBase
from Features.FeatureHeadAndModifierPosClassBuilder import FeatureHeadAndModifierPosClassBuilder
from Features.FeatureHeadAndModifierPosClassDistanceBuilder import FeatureHeadAndModifierPosClassDistanceBuilder
from Features.FeatureHeadAndModifierPosDistanceBuilder import FeatureHeadAndModifierPosDistanceBuilder
from Features.FeatureHeadPosDistanceBuilder import FeatureHeadPosDistanceBuilder
from Features.FeatureModifierPosDistanceBuilder import FeatureModifierPosDistanceBuilder
from Features.FeaturePosBackwardBuilder import FeaturePosBackwardBuilder
from Features.FeaturePosForwardBuilder import FeaturePosForwardBuilder
from Utils.MyParser import MyParser


class ComplexFeatureVectorBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        self.parser = parser
        self.f1 = Feature1Builder(parser, 0)
        size = self.f1.size
        print("f1 interval = ", size - self.f1.size, size)
        self.f2 = Feature2Builder(parser, size)
        size += self.f2.size
        print("f2 interval = ", size - self.f2.size, size)
        self.f3 = Feature3Builder(parser, size)
        size += self.f3.size
        print("f3 interval = ", size - self.f3.size, size)
        self.f7 = Feature7Builder(parser, size)
        size += self.f7.size
        print("f7 interval = ", size - self.f7.size, size)
        self.f8 = Feature8Builder(parser, size)
        size += self.f8.size
        print("f8 interval = ", size - self.f8.size, size)
        self.f9 = Feature9Builder(parser, size)
        size += self.f9.size
        print("f9 interval = ", size - self.f9.size, size)
        self.f10 = Feature10Builder(parser, size)
        size += self.f10.size
        print("f10 interval = ", size - self.f10.size, size)
        self.f11 = Feature11Builder(parser, size)
        size += self.f11.size
        print("f11 interval = ", size - self.f11.size, size)
        self.f12 = Feature12Builder(parser, size)
        size += self.f12.size
        print("f12 interval = ", size - self.f12.size, size)
        self.f13 = Feature13Builder(parser, size)
        size += self.f13.size
        print("f13 interval = ", size - self.f13.size, size)
        self.fHMPosClass = FeatureHeadAndModifierPosClassBuilder(parser, size)
        size += self.fHMPosClass.size
        print("fHMPosClass interval = ", size - self.fHMPosClass.size, size)
        self.fHMPosClassDistance = FeatureHeadAndModifierPosClassDistanceBuilder(parser, size)
        size += self.fHMPosClassDistance.size
        print("fHMPosClassDistance interval = ", size - self.fHMPosClassDistance.size, size)
        self.fHMPosDistnace = FeatureHeadAndModifierPosDistanceBuilder(parser, size)
        size += self.fHMPosDistnace.size
        print("fHMPosDistnace interval = ", size - self.fHMPosDistnace.size, size)
        self.fHPosDistance = FeatureHeadPosDistanceBuilder(parser, size)
        size += self.fHPosDistance.size
        print("fHPosDistance interval = ", size - self.fHPosDistance.size, size)
        self.fMPosDistance = FeatureModifierPosDistanceBuilder(parser, size)
        size += self.fMPosDistance.size
        print("fMPosDistance interval = ", size - self.fMPosDistance.size, size)
        self.fPosBack = FeaturePosBackwardBuilder(parser, size)
        size += self.fPosBack.size
        print("fPosBack interval = ", size - self.fPosBack.size, size)
        self.fPosForward = FeaturePosForwardBuilder(parser, size)
        size += self.fPosForward.size
        print("fPosForward interval = ", size - self.fPosForward.size, size)
        super().__init__(size, offset)

    def getFeatureVector(self, history, head, modifier):
        vec1 = self.f1.getFeatureVector(history, head, modifier)
        vec2 = self.f2.getFeatureVector(history, head, modifier)
        vec3 = self.f3.getFeatureVector(history, head, modifier)
        vec7 = self.f7.getFeatureVector(history, head, modifier)
        vec8 = self.f8.getFeatureVector(history, head, modifier)
        vec9 = self.f9.getFeatureVector(history, head, modifier)
        vec10 = self.f10.getFeatureVector(history, head, modifier)
        vec11 = self.f11.getFeatureVector(history, head, modifier)
        vec12 = self.f12.getFeatureVector(history, head, modifier)
        vec13 = self.f13.getFeatureVector(history, head, modifier)
        vec14 = self.fHMPosClass.getFeatureVector(history, head, modifier)
        vec15 = self.fHMPosClassDistance.getFeatureVector(history, head, modifier)
        vec16 = self.fHMPosDistnace.getFeatureVector(history, head, modifier)
        vec17 = self.fHPosDistance.getFeatureVector(history, head, modifier)
        vec18 = self.fMPosDistance.getFeatureVector(history, head, modifier)
        vec19 = self.fPosBack.getFeatureVector(history, head, modifier)
        vec20 = self.fPosForward.getFeatureVector(history, head, modifier)
        return np.concatenate(
            (vec1, vec2, vec3, vec7, vec8, vec9, vec10, vec11, vec12, vec13,
             vec14, vec15, vec16, vec17, vec18, vec19, vec20)).astype(int)
