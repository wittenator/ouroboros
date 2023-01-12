"""Represents a module, which contains APIs for managing the lifecycle of machine learning models, from training to validation and inference."""

from typing import Any, Union
from abc import ABC, abstractmethod

import numpy
import torch
from tqdm import tqdm
import torchmetrics


class ModelTrainer(ABC):
    """Represents the abstract base class for model trainers. Model trainers can be used to train machine learning models and save the trained models
    in checkpoint files.
    """

    @abstractmethod
    def train_for_one_epoch(self) -> float:
        """Trains the model for a single epoch.

        Returns:
            float: Returns the average training loss over the epoch.
        """

        raise NotImplementedError()


class ModelValidator(ABC):
    """Represents the abstract base class for model validators. Model validators can be used to compute validation metrics on a trained neural network
    model.
    """

    @abstractmethod
    def validate(self) -> tuple:
        """Validates the model on the validation subset of the dataset.

        Returns:
            tuple: Returns a tuple containing the computed validation metrics.
        """

        raise NotImplementedError()


class ModelInferer(ABC):
    """Represents the abstract base class for model inferers. Model inferers can be used to perform inference on trained neural network models."""

    @abstractmethod
    def infer(self, inputs: torch.Tensor) -> Any:
        """Performs inference using a trained neural network model.

        Args:
            inputs (torch.Tensor): The inputs to the neural network.

        Returns:
            Any: Returns the outputs of the neural network model.
        """

        raise NotImplementedError()


class ClassificationModelTrainer(ModelTrainer):
    """Represents a model trainer, which trains a classification model."""

    def __init__(
            self,
            model: torch.nn.Module,
            device: Union[str, torch.device],
            dataset: torch.utils.data.Dataset,  # type: ignore
            batch_size: int,
            learning_rate: float = 0.001,
            ) -> None:
        """Initializes a new ReconstructionBasedAnomalyDetectionModelTrainer instance.

        Args:
            model (torch.nn.Module): The reconstruction-based anomaly detection model that is to be trained.
            device (Union[str, torch.device]): The device on which the training is performed.
            training_subset (torch.utils.data.Dataset): The training split of the dataset on which the model is to be trained.
            batch_size (int): The size of the mini-batch used during training.
            learning_rate (float): The initial learning rate of the optimizer.
            weight_decay (float): The factor for the L2 regularization that is added by the optimizer to promote smaller weights.
            model_name (str): The name of the model that is to be trained.
        """

        # Stores the arguments for later use
        self.model = model
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Creates the data loaders
        self.training_data_loader = torch.utils.data.DataLoader(  # type: ignore
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Moves the model to the specified device
        self.model = self.model.to(self.device)

        # Creates the loss function
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)

        # Creates the optimizer for the training
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate
        )

    def train_for_one_epoch(self) -> float:
        """Trains the reconstruction-based anomaly detection model for a single epoch.

        Returns:
            float: Returns the average training loss over the epoch.
        """

        # Puts the model into training mode, because some parts of the model may behave differently in training as compared to
        # validation/testing/inference
        self.model.train()

        # Cycles through the entire training subset of the dataset and trains the model on the samples (this is for one epoch of training)
        metric = torchmetrics.Accuracy().to(self.device)
        for inputs, classes in self.training_data_loader:

            # Resets the gradients of the optimizer (otherwise the gradients would accumulate)
            self.optimizer.zero_grad()

            # Moves the inputs to the selected device
            inputs = inputs.to(self.device, non_blocking=True)
            classes = classes.to(self.device, non_blocking=True)

            # Performs a forward pass through the neural network
            class_probs = self.model(inputs)
            loss = self.loss_function(class_probs, classes)  # pylint: disable=not-callable

            # Performs the backward pass and the optimization step
            loss.backward()
            self.optimizer.step()

            # Updates the sum of losses from which the mean loss will be computed
            acc = metric(class_probs.argmax(axis=1), classes)

        # metric on all batches using custom accumulation
        acc = metric.compute()

        # Reseting internal state such that metric ready for new data
        metric.reset()
        return acc.item()


class ClassificationModelValidator(ModelValidator):
    """Represents a model validator, which validates a reconstruction-based anomaly detection model on the validation subset of the dataset."""

    def __init__(
            self,
            model: torch.nn.Module,
            device: Union[str, torch.device],
            dataset: torch.utils.data.Dataset,  # type: ignore
            batch_size: int) -> None:
        """Initializes a new ReconstructionBasedAnomalyDetectionModelValidator instance.

        Args:
            model (torch.nn.Module): The reconstruction-based anomaly detection model that is to be validated.
            device (Union[str, torch.device]): The device on which the validation is performed.
            validation_subset (torch.utils.data.Dataset): The validation split of the dataset on which the model is to be validated.
            batch_size (int): The size of the mini-batch used during validation.
            model_name (str): The name of the model that is to be validated.
        """

        # Stores the arguments for later use
        self.model = model
        self.device = device
        self.validation_subset = dataset
        self.batch_size = batch_size

        # Creates the data loaders
        self.validation_data_loader = torch.utils.data.DataLoader(
            self.validation_subset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True
        )

        # Moves the model to the specified device
        self.model = self.model.to(self.device)

        # Creates the loss function
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)

    def validate(self) -> float:
        """Validates the model on the validation subset of the dataset.

        Returns:
            tuple[float, float, float, float, float]: Returns a tuple containing the validation loss, the AUC, the mean anomaly score of the inliers,
                the mean anomaly score of the outliers, and the p-value of the Mann-Whitney U test, which determines how significant the difference
                between the anomaly scores of the inliers and outliers is, i.e., the lower the p-value, the more significant is the difference between
                the anomaly scores of the inliers and outliers.
        """

        # Puts the model into evaluation mode, because some parts of the model may behave differently in training as compared to
        # validation/testing/inference
        self.model.eval()

        # Since we are only evaluating the model, the gradient does not have to be computed
        with torch.no_grad():

            # Cycles through the whole validation subset of the dataset and performs the validation
            metric = torchmetrics.Accuracy().to(self.device)
            for inputs, classes in self.validation_data_loader:

                # Transfers the batch to the selected device
                inputs = inputs.to(self.device, non_blocking=True)
                classes = classes.to(self.device, non_blocking=True)

                # Performs the forward pass through the neural network
                class_probs = self.model(inputs)
                loss = self.loss_function(class_probs, classes)  # pylint: disable=not-callable

                # Updates the validation metrics
                acc = metric(class_probs.argmax(axis=1), classes)

            # metric on all batches using custom accumulation
            acc = metric.compute()

            # Reseting internal state such that metric ready for new data
            metric.reset()
            return acc.item()

'''
class ReconstructionBasedAnomalyDetectionModelInferer(ModelInferer):
    """Represents a model inferer, which performs inference on a reconstruction-based anomaly detection model."""

    def __init__(
            self,
            model: torch.nn.Module,
            device: Union[str, torch.device]) -> None:
        """Initializes a new ReconstructionBasedAnomalyDetectionModelInferer instance.

        Args:
            model (torch.nn.Module): The trained reconstruction-based anomaly detection model on which the inference is to be performed.
            device (Union[str, torch.device]): The device on which the inference is performed.
        """

        # Stores the arguments for later use
        self.model = model
        self.device = device

    def infer(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs inference using a trained neural network model.

        Args:
            inputs (torch.Tensor): The inputs to the neural network.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Returns a tuple containing two PyTorch tensors. The first contains the predictions of the model and the
                second contains the reconstructions that the model created from the inputs.
        """

        # Since we are only performing inference on the model, the gradient does not have to be computed
        with torch.no_grad():

            # Transfers the inputs to the selected device
            inputs = inputs.to(self.device, non_blocking=True)

            # Performs the forward pass through the neural network
            reconstructions = self.model(inputs)
            reconstruction_error = torch.pow(reconstructions - inputs, 2).view(inputs.shape[0], -1).sum(axis=1)
            predictions = (reconstruction_error > self.model.classification_threshold).type(torch.float32)

            # Returns the predictions and the reconstructions
            return predictions, reconstructions
'''
