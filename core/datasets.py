import torch
import torchvision
from os.path import join
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset



def split_dirichlet(labels, n_clients, alpha, double_stochstic=True, seed=0, **kwargs):
    '''Splits data among the clients according to a dirichlet distribution with parameter alpha'''

    np.random.seed(seed)

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    n_classes = np.max(labels) + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    if double_stochstic:
        label_distribution = make_double_stochstic(label_distribution)

    class_idcs = [np.argwhere(np.array(labels) == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    print_split(client_idcs, labels)

    return client_idcs

def make_double_stochstic(x):
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
        return DataLoader(self.public_data, batch_size=self.batch_size, num_workers=4)

    def distill_dataloader(self):
        return DataLoader(self.distill_data, batch_size=self.batch_size, num_workers=4)

class LocalDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str = "datasets", batch_size: int = 32, split: str ='dirichlet', parts: int = 10, transforms=None, **split_kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms
        self.split = split
        self.parts = parts
        self.split_kwargs = split_kwargs
        self._train_data = None
        self._test_data = None
        self.setup()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_data = self.train_data()
            if self.split == "dirichlet":
                self._train_data = [Subset(train_data, subset_idx) for subset_idx in
                                   split_dirichlet(train_data.targets, self.parts, **self.split_kwargs)]
            else:
                self._train_data = train_data

        if stage == 'test' or stage is None:
            self._test_data = self.test_data()


    def train_dataloader(self, id=0):
        assert id < self.parts
        return DataLoader(self._train_data[id], batch_size=self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self._test_data, batch_size=self.batch_size, num_workers=4)


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
            super(STL10DataModule, self).__init__(transforms=transforms,**kwargs)
        super(STL10DataModule, self).__init__(**kwargs)

    def train_data(self):
        print("Downloading")
        return torchvision.datasets.STL10(root=self.data_dir, split='unlabeled', folds=None, download=True, transform=self.transforms)
