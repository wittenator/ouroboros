from datetime import datetime
import os
import torch
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange
from core.aggregator import FederatedAveraging
from core.utils import Participant
from core.steps import Aggregate, Distribute_Dataset, Distribute_Model, Train, Validate
from core.lifecycle import ClassificationModelTrainer, ClassificationModelValidator

logdir = os.getenv("LOGDIR") or "logs/scalars/" + datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)
logger = SummaryWriter(log_dir=logdir)

use_cuda = torch.cuda.is_available()

device = None
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transforms = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))
                                    ])
train_data = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True, download=True, transform=transforms)
test_data = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=transforms)

server = Participant(
    model = torchvision.models.vgg11(num_classes=10),
    group = 'server',
    id = 1,
    trainer = ClassificationModelTrainer,
    validator=ClassificationModelValidator
)

clients = [
    Participant(
        model = torchvision.models.vgg11(num_classes=10),
        group = 'client',
        id = i,
        trainer = ClassificationModelTrainer,
        validator=ClassificationModelValidator
    ) for i in range(5)
]

Distribute_Dataset(
    to = clients,
    dataset = train_data,
    name = 'train',
    split = 'dirichlet',
    alpha = 0.01
)

Distribute_Dataset(
    to = clients + [server],
    dataset = test_data,
    name = 'validate',
    split = 'complete',
)

for communication_round in trange(2):
    Distribute_Model(
        source = server,
        target = clients
    )

    Train(
        on = clients,
        epochs = 2,
        device = device,
        batch_size=64,
        logger=logger,
        communication_round=communication_round
    )

    Aggregate(
        sources = clients,
        target = server,
        aggregator = FederatedAveraging()
    )

    Validate(
        on = clients + [server],
        device = device,
        batch_size = 64,
        logger=logger,
        communication_round=communication_round
    )

logger.close()

