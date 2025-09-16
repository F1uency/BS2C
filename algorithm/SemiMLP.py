import math
from abc import ABC

import math
import torch
from torch import optim
from torch.utils.data import DataLoader

from CoTrain import BaseCoTrain
from algorithm.SklearnInterface import SklearnInterface
from algorithm.Sampler import TwoStreamBatchSampler
from algorithm.ramps import sigmoid_rampup
from algorithm.NeuralNetworks import MLPDropoutReg
from algorithm.SemiDataset import SemiDataset
from util import RegUtil


class SemiMLP(SklearnInterface, ABC):
    def __init__(self, n_epoch, lr, consistency_scale, consistency_rampup, baseModel, evalData,
                 batch_size=50, label_batch_size=10, verbose=False, ablation=False, addGaussianNoise=True,
                 noise_scale=0.001) -> None:
        super().__init__()
        self.verbose = verbose
        self.label_batch_size = label_batch_size
        self.consistency_rampup = consistency_rampup
        self.epochs = n_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self.baseModel = baseModel
        self.consistency_scale = consistency_scale
        self.evalData = evalData
        self.ablation = ablation
        self.addGaussianNoise = addGaussianNoise
        self.noise_scale = noise_scale

    def fit(self, X_l, y_l, X_u):
        self.baseModel.fit(X_l, y_l, X_u)
        self.model = MLPDropoutReg.getDefaultModel(X_l.shape[1], 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        lossFunc = torch.nn.SmoothL1Loss().cuda()
        semiDataset = SemiDataset(X_l, y_l, X_u, addGaussianNoise=self.addGaussianNoise, noise_scale=self.noise_scale)
        labelIdx, unlabelIdx = semiDataset.getSemiIndex()
        sampler = TwoStreamBatchSampler(primary_indices=unlabelIdx,
                                        secondary_indices=labelIdx,
                                        batch_size=self.batch_size,
                                        secondary_batch_size=self.label_batch_size)
        dataLoader = DataLoader(semiDataset, batch_sampler=sampler)
        return self.train(self.epochs, self.model, dataLoader, optimizer, lossFunc, self.baseModel, device)

    def train(self, n_epoch, model, dataLoader, optimizer, lossFunc, baseModel, device):
        trainRmseList = []
        trainBatchIdxList = []
        evalRmseList = []
        evalBatchIdxList = []
        for epoch in range(n_epoch):
            if self.verbose:
                print('Epoch [{}/{}]\n'.format(epoch + 1, n_epoch))
            model.train()
            sum_loss = 0.0
            total = 0.0
            for batch_idx, (trains, labels) in enumerate(dataLoader):
                length = len(dataLoader)
                trains = trains.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                predicts = model(trains)
                labelMask = labels > 0
                unlabelMask = labels == -1
                supervise_loss = lossFunc(predicts[labelMask].ravel(), labels[labelMask])

                if self.ablation:
                    loss = supervise_loss
                else:
                    pseudoLabel = baseModel.predict(trains[unlabelMask].cpu())
                    pseudoLabelTensor = torch.Tensor(pseudoLabel).to(device)

                    # Calculate u_i
                    u_i = torch.abs(predicts[unlabelMask].ravel() - pseudoLabelTensor)

                    # Select top-k pseudo labels with smallest u_i
                    top50_idx = torch.argsort(u_i)[:200]
                    top50_u_i = u_i[top50_idx]
                    top50_pseudoLabel = pseudoLabelTensor[top50_idx]
                    top50_predicts = predicts[unlabelMask][top50_idx]

                    # Calculate weights
                    weights = top50_u_i / torch.sum(top50_u_i)

                    # Calculate weighted loss
                    weighted_loss = weights * lossFunc(top50_predicts.ravel(), top50_pseudoLabel)
                    unsupervise_loss = torch.sum(weighted_loss)

                    # Debug
                    # if self.verbose:
                    #     print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}:")
                    #     print(f"Selected pseudo labels: {top50_pseudoLabel}")
                    #     print(f"Corresponding weights: {weights}")
                    #     print(f"Unsupervised loss: {unsupervise_loss.item()}")

                    consistency_weight = self.consistency_scale * sigmoid_rampup(epoch, self.consistency_rampup)
                    loss = supervise_loss + consistency_weight * unsupervise_loss

                loss.backward()
                optimizer.step()

                if self.verbose:
                    sum_loss += supervise_loss.item()
                    total += labels[labelMask].size(0)
                    totalBatchIdx = batch_idx + 1 + epoch * length
                    trainRmse = math.sqrt(sum_loss / total)
                    print("[epoch:%d, iter:%d] Loss: %.09f"
                          % (epoch + 1, totalBatchIdx, trainRmse))
                    trainBatchIdxList.append(totalBatchIdx)
                    trainRmseList.append(trainRmse)
                    if totalBatchIdx % 10 == 0:
                        evalRmse = RegUtil.evalNNLoss(model, self.evalData, device)
                        evalRmseList.append(evalRmse)
                        evalBatchIdxList.append(totalBatchIdx)
                        model.train()

        if self.verbose:
            return trainBatchIdxList, trainRmseList, evalBatchIdxList, evalRmseList

    def predict(self, X):
        self.model.eval()
        tensorX = torch.Tensor(X).cuda()
        return self.model(tensorX).detach().cpu().numpy()

    @staticmethod
    def getDefaultModel(n_epoch=2000, batch_size=256, label_batch_size=50,
                        consistency_scale=0.9, lr=0.05):
        consistency_rampup = int(n_epoch * 0.8)
        baseModel = BaseCoTrain.getDefaultModel()
        model = SemiMLP(n_epoch, lr, consistency_scale, consistency_rampup, baseModel, None,
                        batch_size=batch_size, label_batch_size=label_batch_size)
        return model

    @staticmethod
    def getDefaultAdjustModel(n_epoch=1000, batch_size=256, label_batch_size=50,
                              consistency_scale=50, lr=0.05, ablation=False, addGaussianNoise=True, noise_scale=0.001):
        consistency_rampup = int(n_epoch * 0.8)
        baseModel = BaseCoTrain.getDefaultModel()
        model = SemiMLP(n_epoch, lr, consistency_scale, consistency_rampup, baseModel, None,
                        batch_size=batch_size, label_batch_size=label_batch_size, ablation=ablation,
                        addGaussianNoise=addGaussianNoise, noise_scale=noise_scale)
        return model
