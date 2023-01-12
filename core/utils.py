import torch
from torch import nn
import torchvision
from os.path import join
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from typing import List, Any, Dict, Type
from PIL import Image
from dataclasses import dataclass, field

from .lifecycle import ModelTrainer, ModelValidator

@dataclass
class Participant:
    """Class for keeping track of an item in inventory."""
    group: Any
    trainer: Type[ModelTrainer]
    validator: Type[ModelValidator]
    model: nn.Module = None
    id: Any = None
    datasets: Dict = field(default_factory=dict)

def split_dirichlet(labels, n_clients, alpha, double_stochstic=True, **kwargs):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    classes = np.unique(labels)
    n_classes = len(classes)

    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    if double_stochstic:
        label_distribution = make_double_stochastic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels) == y).flatten()
                  for y in classes]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    print_split(client_idcs, labels)

    return client_idcs

def make_double_stochastic(x):
    rsum = None
    csum = None

    n = 0
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1
    return x

def print_split(idcs, labels):
  n_labels = np.max(labels) + 1
  print("Data split:")
  splits = []
  for i, idccs in enumerate(idcs):
    split = np.sum(np.array(labels)[idccs].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
    splits += [split]
    if len(idcs) < 30 or i < 10 or i>len(idcs)-10:
      print(" - Client {}: {:55} -> sum={}".format(i,str(split), np.sum(split)), flush=True)
    elif i==len(idcs)-10:
      print(".  "*10+"\n"+".  "*10+"\n"+".  "*10)

  print(" - Total:     {}".format(np.stack(splits, axis=0).sum(axis=0)))
  print()


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

noop = lambda *args, **kwargs: None
'''
class PublicDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str = "datasets", batch_size: int = 32, n_public: int = 80000, n_distill: int = 80000, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms
        self.n_distill = n_distill
        self.n_public = n_public
        self.public_data = None
        self.distill_data = None
        self.setup()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_data = self.train_data()
            idx = np.random.permutation(len(train_data))
            self.public_data = Subset(train_data, idx[:self.n_public])
            self.distill_data = Subset(train_data, idx[self.n_public: self.n_public+self.n_distill])

    def public_dataloader(self):
        return DataLoader(self.public_data, batch_size=self.batch_size, num_workers=4, shuffle=True, pin_memory=True)

    @property
    def num_public_samples(self):
        return len(self.public_data)

    def distill_dataloader(self):
        return DataLoader(self.distill_data, batch_size=self.batch_size, num_workers=4, shuffle=True, pin_memory=True)

class LocalDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str = "datasets", batch_size: int = 32, split: str ='dirichlet', n_clients: int = 10, transforms=None, **split_kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms
        self.split = split
        self.n_clients = n_clients
        self.split_kwargs = split_kwargs
        self._train_data = None
        self._test_data = None

    def setup(self, stage=None):



    def train_dataloader(self, id=0):
        assert id < self.n_clients
        return DataLoader(self._train_data[id], batch_size=self.batch_size, shuffle=True, num_workers=10, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self._test_data, batch_size=self.batch_size, num_workers=10, pin_memory=True)

class SyntheticAnomalyDataset(LocalDataset):
    def __init__(self, anomaly_class_labels: List[int], data_dir: str = "datasets", batch_size: int = 32, split: str ='dirichlet', n_clients: int = 10, transforms=None, local_alpha=100, anomaly_alpha=100,  **split_kwargs):
        super().__init__(data_dir=data_dir, batch_size=batch_size, split=split, n_clients=n_clients, transforms=transforms, **split_kwargs)
        self.anomaly_class_labels = anomaly_class_labels
        self.local_alpha = local_alpha
        self.anomaly_alpha = anomaly_alpha

    def setup(self, stage=None):
        train_data = self.train_data()
        anomaly_class_binary_index = sum([train_data.targets == anomaly_class for anomaly_class in self.anomaly_class_labels]).numpy().astype(bool)
        anomaly_class_index = np.where(anomaly_class_binary_index)[0]

        local_data_class_index = np.where(~anomaly_class_binary_index)[0]
        anomaly_class_index = np.random.choice(anomaly_class_index, size=int(len(local_data_class_index)*0.1))
        if stage == 'fit' or stage is None:
            if self.split == "dirichlet":
                anomaly_data_iterator = zip(
                    split_dirichlet(train_data.targets[local_data_class_index], n_clients=self.n_clients, alpha=self.local_alpha, **self.split_kwargs),
                    split_dirichlet(train_data.targets[anomaly_class_index], n_clients=self.n_clients, alpha=self.anomaly_alpha, **self.split_kwargs)
                )

                # reset the class labels for anomaly detection
                train_data.targets = anomaly_class_binary_index.astype(int)

                self._train_data = []
                for subset_idx_local_data, subset_idx_global_anomalies in anomaly_data_iterator:
                    merged_idx = np.concatenate((local_data_class_index[subset_idx_local_data], anomaly_class_index[subset_idx_global_anomalies]))
                    client_subset = Subset(train_data, merged_idx)
                    self._train_data.append(client_subset)
                    print(1)
            else:
                self._train_data = train_data

        if stage == 'test' or stage is None:
            self._test_data = self.train_data()
            self._test_data.targets = anomaly_class_binary_index.astype(int)

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.data = X
        self.targets = y

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)




class CIFAR10DataModule(LocalDataset):
    def __init__(self, **kwargs):
        transforms = torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))
                                          ])
        super(CIFAR10DataModule, self).__init__(transforms=transforms,**kwargs)

    def train_data(self):
        return torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transforms)

    def test_data(self):
        return torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transforms)

class STL10DataModule(PublicDataset):
    def __init__(self, **kwargs):
        if 'transforms' not in kwargs:
            transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                      (0.2023, 0.1994, 0.2010))
                                                     ])
            super(STL10DataModule, self).__init__(transforms=transforms, **kwargs)
        super(STL10DataModule, self).__init__(**kwargs)

    def train_data(self):
        return torchvision.datasets.STL10(root=join(self.data_dir, 'STL10'), split='unlabeled', folds=None, download=True, transform=self.transforms)
'''