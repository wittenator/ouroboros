import operator
from functools import reduce

from core.models import VGG11s, MOG
from core.datasets import CIFAR10DataModule, STL10DataModule
from core.steps import Distribute_Dataset, Distribute_Model, Train, Pretrain, Test, Aggregate
from core.config import Config
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

import torch
from torch.optim import Adam

config = Config(
    seed=3,
    alpha=100.0,
    models=MOG,
    n_clients=2,
    optimizer_client=Adam,
    lr_server=0.001,
    optimizer_server=Adam,
    lr_client=0.001,
    batch_size=32
)

# Seeding everything
pl.seed_everything(config.seed)


#logger = TestTubeLogger('.', create_git_tag=False)

clients = [config.models(role="client", id=i, **config) for i in range(config.n_clients)]
server = [config.models(role="client", **config)]


cifar = CIFAR10DataModule(**config)

Distribute_Dataset(to=clients, dataset=cifar, name="local", train=True, test=True)()
Distribute_Dataset(to=server, dataset=cifar, name="local", train=False, test=True)()
for round in range(1):
    Distribute_Model(source=server, target=clients)()
    Train(on=clients, mode="local", epochs=1)()
    Aggregate(source=clients, target=server, type='model')()
    Test(on=clients)()
    Test(on=server)()


