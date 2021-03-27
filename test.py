from core.models import VGG11s
from core.datasets import CIFAR10DataModule, STL10DataModule
from core.steps import Distribute_Dataset, Distribute_Model, Train, Pretrain, Test, Aggregate
from core.config import Config
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

from torch.optim import Adam

config = Config(
    alpha=100.0,
    models=VGG11s,
    n_clients=2,
    optimizer_client=Adam,
    lr_server=0.0001,
    optimizer_server=Adam,
    lr_client=0.0001
)

logger = TestTubeLogger('.', create_git_tag=False)

clients = [config.models(role="client", id=i, **config) for i in range(config.n_clients)]
server = [config.models(role="server", **config)]

#cifar = CIFAR10DataModule(alpha=0.01, parts=len(clients))
stl = STL10DataModule()

#print model
server[0].summarize(mode="full")

Pretrain(on=server, epochs=1, dataset=STL10DataModule)
exit()
Distribute_Dataset(to=clients, dataset=cifar, name="local", train=True, test=True)()
Distribute_Dataset(to=server, dataset=cifar, name="local", test=True)()
for i in range(3):
    Train(on=clients, mode="local", epochs=1, logger=logger)()
    Aggregate(source=clients, target=server, type='model')()
    Distribute_Model(server[0], clients)
    Test(on=server, logger=logger)()



