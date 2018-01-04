from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from Utils.MyParser import MyParser
from Utils.Token import Token


def test():
    parser = MyParser('test.txt')
    res = parser.getTupleOfHeadAndModifier()
    vectorBuilder = BasicFeatureVectorBuilder(parser, 0)
    vec = vectorBuilder.getFeatureVector(parser.histories[0], 3, 1)
    assert vectorBuilder.f1.size == 8
    assert vectorBuilder.f2.size == 8
    assert vectorBuilder.f3.size == 6
    assert vectorBuilder.f4.size == 20
    assert vectorBuilder.f5.size == 20
    assert vectorBuilder.f6.size == 15
    assert vectorBuilder.f8.size == 22
    assert vectorBuilder.f10.size == 21
    assert vectorBuilder.f13.size == 19
    assert vectorBuilder.size == 139
    assert vec.size == 9
    print (vec)
    pass


if __name__ == '__main__':
    test()
