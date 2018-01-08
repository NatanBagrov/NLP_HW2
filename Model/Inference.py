import numpy as np
import time
import os

from Features.BasicFeatureVectorBuilder import BasicFeatureVectorBuilder
from Utils.History import History
from Utils.MyParser import MyParser
from Utils.Tagger import Tagger


def infer_basic(fileToInfer, outputFile):
    infer_prepare_params("basic", fileToInfer, outputFile)


def infer_complex(fileToInfer):
    pass
    # infer_prepare_params("complex", fileToInfer)


def infer_prepare_params(basic_or_complex, fileToInfer, outputFile):
    train_parser = MyParser("../train.labeled")
    featureBuilder = None
    weight_file = None
    if basic_or_complex:
        featureBuilder = BasicFeatureVectorBuilder(train_parser, 0)
        weight_file = "Weights/perceptron_iteration_100_weights.txt"
    else:
        pass
    parser = MyParser(fileToInfer)
    tagger = Tagger()
    w = np.loadtxt(weight_file)
    tot_acc = 0
    tot_size = 0
    output = open(outputFile,'w')
    for (history, tree), idx in zip(parser.getAllHistoriesWithParseTrees(), range(len(parser.histories))):
        opt_tree = tagger.historyToOptParseTree(featureBuilder, history, w)
        tmpHistory = History(history.tokens)
        for (h,m) in opt_tree:
            tmpHistory.tokens[m].head=h
        output.write(tmpHistory.toString())
        tot_size += len(tree)
        tot_acc += tagger.compareTrees(tree, opt_tree)
        print("Sentence ", idx, " Total acc: ", 100 * (tot_acc / tot_size))
    output.close()

if __name__ == "__main__":
    infer_basic("../test.labeled", "infered_test.labeled")
