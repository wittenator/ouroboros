"""Represents a module, which contains the primitives for federated learning."""

from enum import Enum
from typing import Dict, List, Union
from abc import ABC, abstractmethod

import torch
from torch import nn
import torchvision
import numpy

from .utils import Participant


class ModelConsolidationStrategy(ABC):
    """Represents the abstract base class for all model consolidation strategies. The model consolidation strategy is the algorithm that is used to
    update the global model from several other models.
    """

    @abstractmethod
    def consolidate_models(self, target: nn.Module, sources: List[nn.Module]) -> None:
        """Consolidates the models of the specified federated learning source into a new global model.

        Args:
            target (nn.Module): The target model, which is to be updated from the source models.
            sources (list[nn.Module]): The federated learning source models, whose models are to be consolidated into an updated global model.
        """

        raise NotImplementedError()


class FederatedAveragingParameterConsolidationMethod(Enum):
    """Represents an enumeration for the different methods that are used by the federated averaging algorithm to consolidate the parameters of
    different neural network layer types.
    """

    MEAN = 'mean'
    SUM = 'sum'


class FederatedAveraging(ModelConsolidationStrategy):
    """Represents the federated averaging (FedAvg) algorithm, which can be used to consolidate the models of the federated learning sources into an
    updated global model.
    """

    def consolidate_models(self, target_model: nn.Module, source_models: Union[List[nn.Module], nn.Module], num_sources: int = None, add_to_target=False) -> None:
        """Consolidates the models of the specified federated learning source into a new global model.

        Args:
            target_model (nn.Module): The federated learning central server that contains the global model, which is to be
                updated from the source models.
            sources (list[nn.Module]): The federated learning sources, whose models are to be consolidated into an updated global model.
        """

        assert not (isinstance(source_models, nn.Module) and num_sources is None), "If incremental federated averaging is performed, the number of all source models must be passed"

        if isinstance(source_models, List):
            num_sources = len(source_models)
        else:
            source_models = [source_models]

        # Determines which method of consolidating parameters is used based on the layer types of the model
        global_model_state_dict = target_model.state_dict()
        parameter_consolidation_methods = self.get_parameter_consolidation_methods(target_model)
        with torch.no_grad():
            for parameter_name, parameter_consolidation_method in parameter_consolidation_methods.items():

                # When the parameter consolidation method is "mean", the parameter is averaged over the sources
                if parameter_consolidation_method is FederatedAveragingParameterConsolidationMethod.MEAN:
                    source_parameter_sum = None
                    for source_model in source_models:
                        if source_parameter_sum is None:
                            source_parameter_sum = source_model.state_dict()[parameter_name].detach().clone() / num_sources
                        else:
                            source_parameter_sum += source_model.state_dict()[parameter_name].detach().clone() / num_sources
                    if add_to_target:
                        global_model_state_dict[parameter_name].add_(source_parameter_sum)
                    else:
                        global_model_state_dict[parameter_name].copy_(source_parameter_sum)

                # When the parameter consolidation method is "sum", the parameter is summed over the sources
                elif parameter_consolidation_method is FederatedAveragingParameterConsolidationMethod.SUM:
                    source_parameter_sum = None
                    for source_model in source_models:
                        if source_parameter_sum is None:
                            source_parameter_sum = source_model.state_dict()[parameter_name].detach().clone()
                        else:
                            source_parameter_sum += source_model.state_dict()[parameter_name].detach().clone()
                    if add_to_target:
                        global_model_state_dict[parameter_name].add_(source_parameter_sum)
                    else:
                        global_model_state_dict[parameter_name].copy_(source_parameter_sum)
                else:
                    raise ValueError(f'The parameter consolidation method "{parameter_consolidation_method.value}" is not supported.')

    def get_parameter_consolidation_methods(self, module: torch.nn.Module, parent_name: str = None) -> Dict[str, str]:
        """Different neural network layer types need to be handled differently when consolidating the model. For example, the weights and biases of
        linear layers must be averaged, while the number of tracked batches in a batchnorm layer have to summed up. This method goes through all
        layers (called modules in PyTorch) and determines the method by which their parameters can be consolidated. Since some modules contain other
        modules themselves, the method goes through all modules recursively.

        Args:
            module (torch.nn.Module): The module for which the consolidation method of their child modules have to be determined.
            parent_name (str, optional): The name of the parent module. When calling this method on a neural network model, nothing needs to be
                specified. This parameter is only used when by the method itself, when it goes through child modules recursively. Defaults to None.

        Raises:
            ValueError: When a layer type (module) is detected, which is not supported by this implementation of federated averaging, then an
                exception is raised. This indicates that the consolidation method for the parameters of this module kind still need to be implemented.

        Returns:
            dict[str, str]: Returns a dictionary, which maps the name of a parameter (which is also the exact name of the parameter in the state
                dictionary of the model) to the method that must be used to consolidate this parameter.
        """

        # Initializes the dictionary that maps the consolidation method for each parameter of the module
        parameter_consolidation_methods = {}

        # Cycles through the child modules of the specified module to determine the consolidation method that must be used for their parameters
        for child_name, child_module in module.named_children():

            # Composes the name of the current module (which corresponds to the name of the module in the state dictionary of the model)
            child_name = child_name if parent_name is None else f'{parent_name}.{child_name}'

            # For different module types, different methods are needed to consolidate their parameters
            if isinstance(
                child_module,
                (
                    torch.nn.Sequential,
                    torchvision.models.mobilenetv3.InvertedResidual,
                    torchvision.ops.SqueezeExcitation,
                    torchvision.models.resnet.BasicBlock,
                    torchvision.models.resnet.Bottleneck,
                )
            ):

                # Sequential modules contains other modules, which are invoked in order, sequential modules do not have any parameters of their own,
                # so the consolidation method for the parameters of their child modules need to be determined recursively
                parameter_consolidation_methods.update(self.get_parameter_consolidation_methods(child_module, parent_name=child_name))

            elif isinstance(child_module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)):

                # Linear layers, convolutional layers, and transpose convolutional layers have a weight and a bias parameter, which can be
                # consolidated by averaging them
                parameter_consolidation_methods[f'{child_name}.weight'] = FederatedAveragingParameterConsolidationMethod.MEAN
                if 'bias' in child_module.__dict__ and child_module.bias:
                    parameter_consolidation_methods[f'{child_name}.bias'] = FederatedAveragingParameterConsolidationMethod.MEAN

            elif isinstance(child_module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):

                # BatchNorm layers have a gamma and a beta parameter (the parameters are called 'weight' and 'bias' respectively), a running mean, and
                # a running variance, which can be consolidated by averaging them, they track the number of batches that they have processed so far,
                # which can be consolidated by summation
                if 'bias' in child_module.__dict__ and child_module.bias:
                    parameter_consolidation_methods[f'{child_name}.weight'] = FederatedAveragingParameterConsolidationMethod.MEAN
                    parameter_consolidation_methods[f'{child_name}.bias'] = FederatedAveragingParameterConsolidationMethod.MEAN
                parameter_consolidation_methods[f'{child_name}.running_mean'] = FederatedAveragingParameterConsolidationMethod.MEAN
                parameter_consolidation_methods[f'{child_name}.running_var'] = FederatedAveragingParameterConsolidationMethod.MEAN
                parameter_consolidation_methods[f'{child_name}.num_batches_tracked'] = FederatedAveragingParameterConsolidationMethod.SUM

            elif isinstance(
                    child_module,
                    (
                        torch.nn.Flatten,
                        torch.nn.Unflatten,
                        torch.nn.ReLU,
                        torch.nn.Sigmoid,
                        torch.nn.LeakyReLU,
                        torch.nn.MaxPool2d,
                        torch.nn.modules.pooling.AdaptiveAvgPool2d,
                        torch.nn.modules.dropout.Dropout,
                        torch.nn.modules.activation.Hardswish,
                        torch.nn.modules.linear.Identity,
                        torch.nn.modules.activation.Hardsigmoid
                    )
                ):

                # Flatten, Unflatten, ReLU activations, Sigmoid activations, LeakeReLU activations, MaxPool2d Layers and Interpolate layers have no
                # parameters, therefore, nothing needs to be done
                continue

            else:

                # Since this current module type is not supported, yet, an exception is raised
                raise ValueError(f'The module {child_name} of type {type(child_module)} is not supported by the federated averaging.')

        # Returns the parameter consolidation methods that were determined
        return parameter_consolidation_methods

'''
class FederatedDistillation(ModelConsolidationStrategy):
    """Represents the federated distillation algorithm, which can be used to consolidate the models of the federated learning sources into an
    updated global model.
    """

    def __init__(self, public_data: torch.utils.data.Dataset, device: Union[str, torch.device], number_of_server_epochs: int = 1):
        """Initializes a new FederatedDistillation instance.

        Args:
            public_data (torch.utils.data.Dataset): The public dataset that is used to perform the distillation.
            device (Union[str, torch.device]): The device on which the distillation is to be performed.
            number_of_server_epochs (int, optional): The number of epochs for which the global model is to be trained during the distillation.
                Defaults to 1.

        Raises:
            ValueError: If no public dataset was provided, an exception is raised.
        """

        if public_data is None:
            raise ValueError("Public Data must be supplied for Federated Distillation")
        self.public_data = public_data
        self.device = device
        self.number_of_server_epochs = number_of_server_epochs

    def consolidate_source_models(self, target: 'FederatedLearningCentralServer', sources: list['FederatedLearningsource']) -> None:
        """Consolidates the models of the specified federated learning source into a new global model for the federated learning central server.

        Args:
            target (FederatedLearningCentralServer): The federated learning central server that contains the global model, which is to be
                updated from the source models.
            sources (list[FederatedLearningsource]): The federated learning sources, whose models are to be consolidated into an updated global model.
        """

        public_dataloader = torch.utils.data.DataLoader(
            self.public_data,
            batch_size=128,
            shuffle=False
        )

        mean_source_anomaly_scores = []
        with torch.no_grad():
            for inputs, _ in public_dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                logits_source = torch.mean(torch.stack([source.model.predict_anomaly_scores(inputs) for source in sources]), axis=0)
                mean_source_anomaly_scores.append(logits_source)
        mean_source_anomaly_scores = torch.cat(mean_source_anomaly_scores)
        self.public_data.targets = mean_source_anomaly_scores

        distill_trainer = DistillationTrainer(
            model=target.model,
            device=self.device,
            public_data=self.public_data,
            batch_size=256,
            learning_rate=0.001,
            model_name="Distillation Server"
        )

        for _ in range(self.number_of_server_epochs):
            distill_trainer.train_for_one_epoch()


class ComposeModelConsolidationStrategies(ModelConsolidationStrategy):
    """Composes multiple consolidation methods into one by concatenation."""

    def __init__(self, *args: list[ModelConsolidationStrategy]) -> None:
        """Initializes a new ComposeModelConsolidationStrategies instance.

        Args:
            *args (list[ModelConsolidationStrategy]): A list of model consolidation strategies that are being concatenated.
        """

        for step in args:
            assert isinstance(step, ModelConsolidationStrategy), "All steps must be ModelConsolidationStrategies"
        self.steps = args

    def consolidate_source_models(self, target: 'FederatedLearningCentralServer', sources: list['FederatedLearningsource']) -> None:
        """Consolidates the models of the specified federated learning source into a new global model for the federated learning central server.

        Args:
            target (FederatedLearningCentralServer): The federated learning central server that contains the global model, which is to be
                updated from the source models.
            sources (list[FederatedLearningsource]): The federated learning sources, whose models are to be consolidated into an updated global model.
        """

        for step in self.steps:
            step.consolidate_source_models(target=target, sources=sources)
'''