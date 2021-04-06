import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from collections import OrderedDict
from pl_bolts.models.self_supervised import SimCLR

class Model(pl.LightningModule):
    def __init__(self, role , id=0,  **kwargs):
        super().__init__()
        self.role = role
        self.kwargs = kwargs
        self.datasets = {}
        self._mode = "local"
        self.id = id
        self.accuracy = pl.metrics.Accuracy()

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

    def extract_features(self, X):
        return torch.flatten(self.f(X), start_dim=1)

    def forward(self, X):
        features = self.extract_features(X)
        if self._mode == "pretrain":
            return features
        else:
            return self.classification_layer(features)

    def local_training(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log(f'{self.role}_{self.id}/train_loss/local', loss, on_step=False, on_epoch=True)
        return loss

    def distill_training(self, batch, batch_idx):
        x, y = batch
        loss = F.kl_div(self(x), y)
        self.log(f'{self.role}_{self.id}/train_loss/distill', loss, on_step=False, on_epoch=True)
        return loss

    def mode(self, m):
        self._mode = m
        return self

    def test_step(self, batch, batch_idx):
        self.train()
        x, y = batch
        out = self.forward(x)
        print(out)
        y_hat = torch.argmax(out, dim=1)
        print(y_hat)
        self.accuracy(y_hat, y)

    def test_epoch_end(self, outputs):
        self.log('accuracy', self.accuracy.compute())


    def configure_optimizers(self):
        optimizer = self.kwargs[f"optimizer_{self.role}"](self.parameters(), lr=self.kwargs[f"lr_{self.role}"])
        return optimizer




class VGG(Model):

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


    def make_layers(self, cfg, batch_norm=False):
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