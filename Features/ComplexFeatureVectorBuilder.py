import numpy as np

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
from Utils.MyParser import MyParser


class ComplexFeatureVectorBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        self.parser = parser
        self.f1 = Feature1Builder(parser, 0)
        size = self.f1.size
        print(self.f1.size)
        self.f2 = Feature2Builder(parser, size)
        size += self.f2.size
        print(self.f2.size)
        self.f3 = Feature3Builder(parser, size)
        size += self.f3.size
        print(self.f3.size)
        self.f4 = Feature4Builder(parser, size)
        size += self.f4.size
        print(self.f4.size)
        self.f5 = Feature5Builder(parser, size)
        size += self.f5.size
        print(self.f5.size)
        self.f6 = Feature6Builder(parser, size)
        size += self.f6.size
        print(self.f6.size)
        self.f7 = Feature7Builder(parser, size)
        size += self.f7.size
        print(self.f7.size)
        self.f8 = Feature8Builder(parser, size)
        size += self.f8.size
        print(self.f8.size)
        self.f9 = Feature9Builder(parser, size)
        size += self.f9.size
        print(self.f9.size)
        self.f10 = Feature10Builder(parser, size)
        size += self.f10.size
        print(self.f10.size)
        self.f11 = Feature11Builder(parser, size)
        size += self.f11.size
        print(self.f11.size)
        self.f12 = Feature12Builder(parser, size)
        size += self.f12.size
        print(self.f12.size)
        self.f13 = Feature13Builder(parser, size)
        size += self.f13.size
        print(self.f13.size)
        super().__init__(size, offset)

    def getFeatureVector(self, history, head, modifier):
        vec1 = self.f1.getFeatureVector(history, head, modifier)
        vec2 = self.f2.getFeatureVector(history, head, modifier)
        vec3 = self.f3.getFeatureVector(history, head, modifier)
        vec4 = self.f4.getFeatureVector(history, head, modifier)
        vec5 = self.f5.getFeatureVector(history, head, modifier)
        vec6 = self.f6.getFeatureVector(history, head, modifier)
        vec7 = self.f7.getFeatureVector(history, head, modifier)
        vec8 = self.f8.getFeatureVector(history, head, modifier)
        vec9 = self.f9.getFeatureVector(history, head, modifier)
        vec10 = self.f10.getFeatureVector(history, head, modifier)
        vec11 = self.f11.getFeatureVector(history, head, modifier)
        vec12 = self.f12.getFeatureVector(history, head, modifier)
        vec13 = self.f13.getFeatureVector(history, head, modifier)
        return np.concatenate(
            (vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11, vec12, vec13)).astype(int)
