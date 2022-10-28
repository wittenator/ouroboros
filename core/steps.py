import operator
from pyexpat import model
from typing import List, Type

import numpy as np
import torch
from torch.utils.data import Subset
from torch import nn
from tqdm import tqdm, trange

from .utils import split_dirichlet

def Distribute_Dataset(to, dataset, name, split, **split_kwargs):

    if split == "dirichlet":
        split_dataset = [Subset(dataset, subset_idx) for subset_idx in
                            split_dirichlet(dataset.targets, n_clients=len(to), **split_kwargs)]
        for model, dataset_part in zip(to, split_dataset) :
            model.datasets[name] = dataset_part
    else:
        for model in to :
            model.datasets[name] = dataset

def Distribute_Model(source, target):

    assert issubclass(source, torch.nn.Module), "The source must be a Pytorch model which has a state_dict() function"
    assert all([issubclass(target_model, torch.nn.Module) for target_model in target]), "The target must be a sequence of Pytorch models which has a load_state_dict() function"

    for model in target:
        model.load_state_dict(source.state_dict())

def Train(on, epochs, device, logger=None):
    for participant in on:
        trainer_instance = participant.trainer(
            model = participant.model,
            device = device,
            dataset = participant.datasets["train"],
            batch_size = 16
        )
        for epoch in trange(epochs):
            epoch_acc = trainer_instance.train_for_one_epoch()
            print(epoch_acc)

def Test(on, tester, logger=None):
    for model in on:
        Tester(logger=logger).test(model)

def Aggregate(source, target, aggregator):

    for target_model in target:
        target_model = 1.0/len(source) * reduce(operator.add, source, target_model.set_additive_identity())





