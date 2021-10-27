import operator
from functools import reduce
from typing import List, Type

import numpy as np
import torch
import pytorch_lightning as pl
from pl_bolts.datamodules import STL10DataModule
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform
from .models import ExtendedSimCLR, AlgebraicMixin
from torch.optim.swa_utils import update_bn

from os.path import join



def Distribute_Dataset(to, dataset, name, train=True, test=True):
    def Distribute_Dataset2(message=None):
        for id, model in enumerate(to):
            if train:
                model.datasets[name] = dataset.train_dataloader(id)
            if test:
                model.datasets["test"] = dataset.test_dataloader()
        return message
    return Distribute_Dataset2

def Distribute_Model(source, target):
    def Distribute_Model2(message=None):
        for model in target:
            model.load_state_dict(source[0].state_dict())
        return message
    return Distribute_Model2

def Train(on, epochs, mode, logger=None):
    def Train2(message=None):
        for model in on:
            pl.Trainer(max_epochs=epochs, logger=logger, weights_summary=None).fit(model.mode(mode))
        return message
    return Train2

def Pretrain(on, epochs, dataset, logger=None):
    dm = STL10DataModule(data_dir=join('./datasets', 'STL10'))
    dm.train_transforms = SimCLRTrainDataTransform(32)
    dm.val_transforms = SimCLREvalDataTransform(32)

    simclr_model = ExtendedSimCLR(num_samples=dm.num_unlabeled_samples, batch_size=dm.batch_size, model=on[0], gpus=1, dataset=None, hidden_mlp=128)

    def Pretrain2(message=None):
        for model in on:
            pl.Trainer(max_epochs=2, logger=logger, weights_summary=None, gpus=1).fit(simclr_model, datamodule=dm)
        return message
    return Pretrain2

def Test(on, logger=None):
    def Test2(message=None):
        for model in on:
            pl.Trainer(logger=logger).test(model.mode("local"))
        return message
    return Test2

def Aggregate(source, target: List[Type[AlgebraicMixin]], type='model', method='Average'):
    def Aggregate2(message=None):
        for target_model in target:
            print(1.0/len(source))
            target_model = 1.0/len(source) * reduce(operator.add, source, target_model.set_additive_identity())
        return message
    return Aggregate2





