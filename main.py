from CoTrain.BaseCoTrain import BaseCoTrain
from algorithm.SemiMLP import SemiMLP
from dataset.ArffRegDataset import ArffRegDataset
from util import RegUtil


def main():
    n_epoch = 1000
    batch_size = 256
    label_batch_size = 50
    consistency_rampup = int(n_epoch * 0.8)
    consistency_scale = 50
    lr = 0.05

    verbose = True
    abalone = ArffRegDataset.abalone()
    labelData, unlabelData, testData, evalData = abalone.total_split(n_label=0.1, random_state=1)
    baseModel = BaseCoTrain.getDefaultModel()
    baseModel.fit(labelData.X, labelData.y, unlabelData.X)
    model = SemiMLP(n_epoch, lr, consistency_scale, consistency_rampup, baseModel, evalData,
                    batch_size=batch_size, label_batch_size=label_batch_size, verbose=verbose)
    model.fit(labelData.X, labelData.y, unlabelData.X)
    rmse = RegUtil.evalRegTestRmse(model, testData)
    print("SemiMLP rmse: {}".format(rmse))
    print("baseModel rmse: {}".format(RegUtil.evalRegTestRmse(baseModel, testData)))


if __name__ == '__main__':
    RegUtil.setup_seed(10)
    main()
