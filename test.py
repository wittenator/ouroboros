import torch
import torchvision
from ouroboros.utils import Participant
from ouroboros.steps import Distribute_Dataset

transforms = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010))
                                    ])
train_data = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transforms)
test_data = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transforms)

server = Participant(
    model = torchvision.models.vgg11(),
    group = 'server',
    id = 1,
)

clients = [
    Participant(
        model = torchvision.models.vgg11(),
        group = 'client',
        id = i,
    ) for i in range(5)
]

Distribute_Dataset(
    to = clients,
    dataset = train_data,
    name = 'train',
    split = 'dirichlet',
    alpha = 0.1
)

