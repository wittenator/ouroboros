from __future__ import annotations

import numbers
from abc import ABCMeta, abstractmethod

import torch
from sklearn.utils.validation import check_is_fitted
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from collections import OrderedDict
from pl_bolts.models.self_supervised import SimCLR
from torchmetrics.classification.accuracy import Accuracy

from sklearn.mixture import BayesianGaussianMixture

class AlgebraicMixin(metaclass=ABCMeta):

    @abstractmethod
    def set_additive_identity(self):
        raise NotImplementedError(f"Setting to additive identity of {self.__name__} is not implemented")

    @abstractmethod
    def __add__(self, other):
        raise NotImplementedError(f"Summation of {self.__name__} is not implemented")

    @abstractmethod
    def __mul__(self, other):
        raise NotImplementedError(f"Scalar multiplication of {self.__name__} is not implemented")

    @abstractmethod
    def __rmul__(self, other):
        raise NotImplementedError(f"Reverse scalar multiplication of {self.__name__} is not implemented")

class Model(pl.LightningModule, AlgebraicMixin):
    def __init__(self, role, id=0, **kwargs):
        super().__init__()
        self.role = role
        self.kwargs = kwargs
        self.datasets = {}
        self._mode = "local"
        self.id = id

    def train_dataloader(self):
        if self._mode == "local":
            return self.datasets["local"]
        elif self._mode == "distill":
            return self.datasets["distill"]
        elif self._mode == "pretrain":
            return self.datasets["public"]

    def test_dataloader(self):
        return self.datasets["test"]

    def training_step(self, batch, batch_idx):
        if self._mode == "local":
            return self.local_training(batch, batch_idx)
        elif self._mode == "distill":
            return self.distill_training(batch, batch_idx)

    def mode(self, m):
        self._mode = m
        return self

class NNModel(Model):
    def __init__(self, role, id=0, **kwargs):
        super().__init__(role, id=0, **kwargs)
        self.accuracy = Accuracy()

    def extract_features(self, X):
        return torch.flatten(self.f(X), start_dim=1)

    def forward(self, X):
        features = self.extract_features(X)
        if self._mode == "pretrain":
            return features
        else:
            return self.classification_layer(features)

    def local_training(self, batch, batch_idx):
        self.train()
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log(f'{self.role}_{self.id}/train_loss/local', loss, on_step=False, on_epoch=True)
        return loss

    def distill_training(self, batch, batch_idx):
        x, y = batch
        loss = F.kl_div(self(x), y)
        self.log(f'{self.role}_{self.id}/train_loss/distill', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        out = self.forward(x)
        y_hat = torch.argmax(out, dim=1)
        self.accuracy(y_hat, y)

    def test_epoch_end(self, outputs):
        self.log('accuracy', self.accuracy.compute())

    def configure_optimizers(self):
        optimizer = self.kwargs[f"optimizer_{self.role}"](self.parameters(), lr=self.kwargs[f"lr_{self.role}"])
        return optimizer

    def set_additive_identity(self) -> NNModel:
        for param in self.parameters():
            param.detach().copy_(torch.zeros_like(param))
        return self

    def __add__(self, other):
        if isinstance(other, type(self)):
            for p_target, p_source in zip(self.parameters(), other.parameters()):
                p_target.detach().copy_(p_target.detach() + p_source.detach())
        else:
            raise NotImplementedError(f"Reverse addition is not implemented for {self.__name__}")
        return self

    def __rmul__(self, other: numbers.Number) -> NNModel:
        if isinstance(other, numbers.Number):
            for param in self.parameters():
                param.detach().copy_(param.detach() * other)
        else:
            raise NotImplementedError(f"Reverse multiplication is not implemented for {self.__name__}")
        return self

    def __mul__(self, other):
        return self.__rmul__(other)

class MOG(Model):

    def __init__(self, role, **kwargs):
        super().__init__(role, **kwargs)
        self.model = BayesianGaussianMixture()
        self.l1 = nn.Linear(1, 1)

    def forward(self, *args, **kwargs) -> Any:
        pass

    def training_step(self, batch, batch_idx):
        if not hasattr(self.model, "weights_"):
            data = np.concatenate([torch.flatten(x, start_dim=1).numpy() for (x,y) in self.train_dataloader()])
            self.model.fit(data)

        return torch.randn(1, requires_grad=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def set_additive_identity(self) -> MOG:
        self.model.weights_ = None
        self.model.means_ = None
        self.model.covariances_ = None
        return self

    def __add__(self, other) -> MOG:
        if isinstance(other, type(self)):
            for attribute in ['weights_', 'means_', 'covariances_']:
                if getattr(self.model, attribute, None) is None:
                    setattr(self.model, attribute, getattr(other.model, attribute))
                else:
                    setattr(self.model, attribute, np.concatenate([getattr(o, attribute) for o in [self.model, other.model]]))
        else:
            raise NotImplementedError(f"Reverse addition is not implemented for {self.__name__}")
        return self

    def __rmul__(self, other: numbers.Number) -> MOG:
        return self

    def __mul__(self, other):
        return self.__rmul__(other)


class VGG(NNModel):

    def __init__(self, cfg, size=512, num_classes=10, group_norm=False, **kwargs):
        super(VGG, self).__init__(**kwargs)

        self.f = self.make_layers(cfg)

        self.classification_layer = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(size, size),
            nn.ReLU(True),
            nn.Linear(size, num_classes),
        )

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

        if group_norm:
            apply_gn(self)


    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

def VGG11s(num_classes=10, **kwargs):
    return VGG([32, 'M', 64, 'M', 128, 128, 'M', 128, 128, 'M', 128, 128, 'M'], size=128, num_classes=num_classes, **kwargs)


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

class ExtendedSimCLR(SimCLR):
    def __init__(self, model, **kwargs):
        super(ExtendedSimCLR, self).__init__(**kwargs)
        self.encoder = model.mode("pretrain")

    def init_model(self):
        return None

    def forward(self, x):
        return self.encoder(x)