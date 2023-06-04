import copy
from typing import Callable, Iterable, List

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm, trange
from torch.utils.tensorboard.writer import SummaryWriter


from core.aggregator import ModelConsolidationStrategy

from .utils import Participant, split_dirichlet, noop

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
        print(f"Loading model from {source.group}-{source.id} to {participant.group}-{participant.id}")
        participant.model.load_state_dict(source.model.state_dict())  # type: ignore

def Train(on: List[Participant], epochs: int, device: torch.device, batch_size: int, communication_round: int, logger: SummaryWriter):
    for participant in on:
        trainer_instance = participant.trainer(
            model = participant.model,
            optimizer = participant.optimizer,
            device = device,
            dataset = participant.datasets["train"],
            batch_size = batch_size,
            number_of_classes = participant.number_of_classes,
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
            batch_size = batch_size,
            number_of_classes = participant.number_of_classes,
        ).validate()

        logger.add_scalar(f'val/{participant.group}/{participant.group}-{participant.id}/accuracy', acc, communication_round)


def Aggregate(sources, target, aggregator: ModelConsolidationStrategy):
    aggregator.consolidate_models(target_model=target.model, source_models=[source.model for source in sources])

def Train_and_Aggregate(
    on: List[Participant],
    target: Participant,
    epochs: int,
    device: torch.device,
    batch_size: int,
    communication_round: int,
    logger: SummaryWriter,
    aggregator: ModelConsolidationStrategy,
    pre_training_hook: Callable = noop,
    post_training_hook: Callable = noop,
    pre_aggregation_hook: Callable = noop,
    ):
    central_model = target.model
    central_model = central_model.to(device)
    temp_model = copy.deepcopy(central_model)
    temp_model = temp_model.to(device)
    client_model = copy.deepcopy(target.model)


    for idx, participant in enumerate(on):
        client_model.load_state_dict(central_model.state_dict())
        participant.model = client_model
        trainer_instance = participant.trainer(
            model = participant.model,
            device = device,
            dataset = participant.datasets["train"],
            batch_size = batch_size
            )  # type: ignore
        # execute hook before trainig starts
        pre_training_hook(participant=participant, communication_round=communication_round, logger=logger)

        for epoch in trange(epochs):
            epoch_acc = trainer_instance.train_for_one_epoch()
        logger.add_scalar(f'train/{participant.group}/{participant.group}-{participant.id}/accuracy', epoch_acc, communication_round)


        # execute hook after trainig ends
        post_training_hook(participant=participant, communication_round=communication_round, logger=logger)

        # execute hook before aggregation ends
        pre_aggregation_hook(participant=participant, communication_round=communication_round, logger=logger)
        
        aggregator.consolidate_models(
            target_model=temp_model,
            source_models=participant.model,
            num_sources=len(on),
            add_to_target=(idx!=0)
        )

        # remove reference to temporary client model
        participant.model = None
    target.model.load_state_dict(temp_model.state_dict())

def Create_Differential_Update(source, target):
    source_weights = {key : value for key, value in source.model.named_parameters()}
    for target_model in target:
        target_weights = {key : value for key, value in target_model.model.named_parameters()}
        for name in target_weights.keys():
            target_weights[name].data = (target_weights[name].detach() - source_weights[name].detach()).clone()

def Restore_Differential_Update(source, target):
    source_weights = {key : value for key, value in source.model.named_parameters()}
    for target_model in target:
        target_weights = {key : value for key, value in target_model.model.named_parameters()}
        for name in target_weights.keys():
            target_weights[name].data = (target_weights[name].detach() + source_weights[name].detach()).clone()



