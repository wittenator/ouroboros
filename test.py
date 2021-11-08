import operator
from functools import reduce

from core.models import VGG11s, MOG, Autoencoder
from core.datasets import CIFAR10DataModule, STL10DataModule, GaussianBlobAnomalyDataset
from core.steps import Distribute_Dataset, Distribute_Model, Train, Pretrain, Test, Aggregate
from core.config import Config
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

import torch
from torch.optim import Adam

config = Config(
    seed=3,
    alpha=100.0,
    models=Autoencoder,
    n_clients=2,
    optimizer_client=Adam,
    lr_server=0.001,
    optimizer_server=Adam,
    lr_client=0.001,
    batch_size=32,
    anomaly_class_labels=[1,2,3]
)

# Seeding everything
pl.seed_everything(config.seed)

#logger = TestTubeLogger('.', create_git_tag=False)

clients = [config.models(role="client", id=i, **config) for i in range(config.n_clients)]
server = [config.models(role="client", **config)]

g = GaussianBlobAnomalyDataset(anomaly_class_labels=[1,2], local_alpha=0.01, anomaly_alpha=100, n_clients=3)
g.setup()

Distribute_Dataset(to=clients, dataset=g, name="local", train=True, test=True)()
Distribute_Dataset(to=server, dataset=g, name="local", train=False, test=True)()
for round in range(10):
    Distribute_Model(source=server, target=clients)()
    Train(on=clients, mode="local", epochs=1)()
    Aggregate(source=clients, target=server, type='model')()
    Test(on=clients)()
    Test(on=server)()
