import numpy as np
import time

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from Features.ComplexFeatureVectorBuilder import ComplexFeatureVectorBuilder
from Features.VeryComplexFeatureVectorBuilder import VeryComplexFeatureVectorBuilder
from Utils.MyParser import MyParser
from Utils.Perceptron import Perceptron
from Utils.Tagger import Tagger


def fit_complex_model(continueFile, numIterations):
    resFile = "finish_complex_w" + str(numIterations) + ".txt"
    parser = MyParser("../train.labeled")
    featureBuilder = ComplexFeatureVectorBuilder(parser, 0)
    tagger = Tagger()
    perceptron = Perceptron(featureBuilder, tagger)
    v = None
    if continueFile is not None:
        v = np.loadtxt(continueFile)
    v = fit_model_aux(resFile, parser, perceptron, numIterations, v)

def fit_very_complex_model(continueFile, numIterations):
    resFile = "finish_very_complex_w" + str(numIterations) + ".txt"
    parser = MyParser("../train.labeled")
    featureBuilder = VeryComplexFeatureVectorBuilder(parser, 0)
    tagger = Tagger()
    perceptron = Perceptron(featureBuilder, tagger)
    v = None
    if continueFile is not None:
        v = np.loadtxt(continueFile)
    v = fit_model_aux(resFile, parser, perceptron, numIterations, v)


def fit_basic_model(continueFile, numIterations):
    resFile = "finish_basic_w" + str(numIterations) + ".txt"
    parser = MyParser("../train.labeled")
    featureBuilder = BasicFeatureVectorBuilder(parser, 0)
    tagger = Tagger()
    perceptron = Perceptron(featureBuilder, tagger)
    v = None
    if continueFile is not None:
        v = np.loadtxt(continueFile)
    v = fit_model_aux(resFile, parser, perceptron, numIterations, v)
    # for history in parser.histories:
    #     print(tagger.historyToOptParseTree(featureBuilder,history,v))
    #     pass


def fit_model_aux(resFile, parser: MyParser, perceptron: Perceptron, iterationsNum, initv=None):
    v = None
    if initv is not None:
        print("Will continue training given init vector")
        v = initv
    start = time.time()
    best_v = perceptron.perceptron(iterationsNum, parser.getAllHistoriesWithParseTrees(), v)
    np.savetxt(resFile, best_v)
    print("Training took: ", (time.time() - start) / 60, "minutes")
    print("######################################################")
    return best_v


if __name__ == "__main__":
    #fit_basic_model(None, 50)
    # fit_complex_model(None, 1)
    fit_very_complex_model(None, 100)
