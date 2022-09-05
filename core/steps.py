import operator
from typing import List, Type

import numpy as np
import torch
from torch import nn

def Distribute_Dataset(to, dataset, name, split, **split_kwargs):
    if split == "dirichlet":
        _train_data = [Subset(train_data, subset_idx) for subset_idx in
                            split_dirichlet(train_data.targets, n_clients=len(n_clients), **self.split_kwargs)]
    else:
        _train_data = train_data

    return message

def Distribute_Model(source, target):

    assert issubclass(source, torch.nn.Module), "The source must be a Pytorch model which has a state_dict() function"
    assert all([issubclass(target_model, torch.nn.Module) for target_model in target]), "The target must be a sequence of Pytorch models which has a load_state_dict() function"

    for model in target:
        model.load_state_dict(source.state_dict())

def Train(on, epochs, trainer, logger=None):
    for model in on:
        trainer(max_epochs=epochs, logger=logger, weights_summary=None).fit(model)

def Test(on, tester, logger=None):
    for model in on:
        Tester(logger=logger).test(model)

def Aggregate(source, target, aggregator):

    for target_model in target:
        target_model = 1.0/len(source) * reduce(operator.add, source, target_model.set_additive_identity())





