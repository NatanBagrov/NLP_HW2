import numpy as np

from Features.Feature1Builder import Feature1Builder
from Features.Feature2Builder import Feature2Builder
from Features.Feature3Builder import Feature3Builder
from Features.Feature4Builder import Feature4Builder
from Features.Feature5Builder import Feature5Builder
from Features.Feature6Builder import Feature6Builder
from Features.Feature8Builder import Feature8Builder
from Features.Feature10Builder import Feature10Builder
from Features.Feature13Builder import Feature13Builder
from Features.FeatureBuilderBase import FeatureBuilderBase
from Utils.MyParser import MyParser


class BasicFeatureVectorBuilder(FeatureBuilderBase):
    def __init__(self, parser: MyParser, offset) -> None:
        self.parser = parser
        self.f1 = Feature1Builder(parser, 0)
        size = self.f1.size
        print("F1 size:", self.f1.size)
        self.f2 = Feature2Builder(parser, size)
        size += self.f2.size
        print("F2 size:", self.f2.size)
        self.f3 = Feature3Builder(parser, size)
        size += self.f3.size
        print("F3 size:", self.f3.size)
        self.f4 = Feature4Builder(parser, size)
        size += self.f4.size
        print("F4 size:", self.f4.size)
        self.f5 = Feature5Builder(parser, size)
        size += self.f5.size
        print("F5 size:", self.f5.size)
        self.f6 = Feature6Builder(parser, size)
        size += self.f6.size
        print("F6 size:", self.f6.size)
        self.f8 = Feature8Builder(parser, size)
        size += self.f8.size
        print("F8 size:", self.f8.size)
        self.f10 = Feature10Builder(parser, size)
        size += self.f10.size
        print("F10 size:", self.f10.size)
        self.f13 = Feature13Builder(parser, size)
        size += self.f13.size
        print("F13 size:", self.f13.size)
        super().__init__(size, offset)

    def getFeatureVector(self, history, head, modifier):
        vec1 = self.f1.getFeatureVector(history, head, modifier)
        vec2 = self.f2.getFeatureVector(history, head, modifier)
        vec3 = self.f3.getFeatureVector(history, head, modifier)
        vec4 = self.f4.getFeatureVector(history, head, modifier)
        vec5 = self.f5.getFeatureVector(history, head, modifier)
        vec6 = self.f6.getFeatureVector(history, head, modifier)
        vec8 = self.f8.getFeatureVector(history, head, modifier)
        vec10 = self.f10.getFeatureVector(history, head, modifier)
        vec13 = self.f13.getFeatureVector(history, head, modifier)
        return np.concatenate((vec1, vec2, vec3, vec4, vec5, vec6, vec8, vec10, vec13)).astype(int)
