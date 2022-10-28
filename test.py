import torch
import torchvision
from core.utils import Participant
from core.steps import Distribute_Dataset, Train
from core.lifecycle import ClassificationModelTrainer

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
)

clients = [
    Participant(
        model = torchvision.models.vgg11(num_classes=10),
        group = 'client',
        id = i,
        trainer = ClassificationModelTrainer,
    ) for i in range(5)
]

Distribute_Dataset(
    to = clients,
    dataset = train_data,
    name = 'train',
    split = 'dirichlet',
    alpha = 0.1
)

Train(
    on = clients,
    epochs = 1,
    device = device
)

