from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm, trange
from torch.utils.tensorboard.writer import SummaryWriter


from core.aggregator import ModelConsolidationStrategy

from .utils import Participant, split_dirichlet

def Distribute_Dataset(to, dataset, name, split, **split_kwargs):
    rng = np.random.default_rng()
    
    if not isinstance(to, Iterable):
        to = [to]

    if split == "dirichlet":
        split_dataset = []
        for subset_idx in split_dirichlet(dataset.targets, n_clients=len(to), **split_kwargs):
            split_dataset.append(
                Subset(
                    dataset,
                    rng.choice(
                        subset_idx,
                        replace=False,
                        size=split_kwargs['datapoints_per_split'] if split_kwargs.get('datapoints_per_split', 0) > 0 else len(subset_idx)
                    )
                )   
            )
        for model, dataset_part in zip(to, split_dataset) :
            model.datasets[name] = dataset_part
    else:
        for model in to :
            model.datasets[name] = Subset(dataset, rng.choice(range(len(dataset)), replace=False, size=split_kwargs['datapoints_per_split'] if split_kwargs.get('datapoints_per_split', 0) > 0 else len(dataset)))

def Distribute_Model(source, target):

    for participant in target:
        participant.model.load_state_dict(source.model.state_dict())  # type: ignore

def Train(on: List[Participant], epochs: int, device: torch.device, batch_size: int, communication_round: int, logger: SummaryWriter):
    for participant in on:
        trainer_instance = participant.trainer(
            model = participant.model,
            device = device,
            dataset = participant.datasets["train"],
            batch_size = batch_size
            )  # type: ignore
        for epoch in trange(epochs):
            epoch_acc = trainer_instance.train_for_one_epoch()
        logger.add_scalar(f'train/{participant.group}/{participant.group}-{participant.id}/accuracy', epoch_acc, communication_round)


def Validate(on, device, batch_size, communication_round: int, logger: SummaryWriter):

    if not isinstance(on, Iterable):
        on = [on]

    for participant in on:
        acc = participant.validator(
            model = participant.model,
            device = device,
            dataset = participant.datasets["validate"],
            batch_size = batch_size
        ).validate()

        logger.add_scalar(f'val/{participant.group}/{participant.group}-{participant.id}/accuracy', acc, communication_round)


def Aggregate(sources, target, aggregator: ModelConsolidationStrategy):
    aggregator.consolidate_models(target=target, sources=sources)





