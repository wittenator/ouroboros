from itertools import product
from collections.abc import Iterable
from torch.optim import Adam

class Config(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __iter__(self):
        if any(isinstance(x, Iterable) for x in self.__dict__):
            kwargs = {k:v if isinstance(v, Iterable) else [v] for k,v in self.__dict__.items()}
            keys = kwargs.keys()
            vals = kwargs.values()
            return (Config(**dict(zip(keys, instance))) for instance in product(*vals))
        else:
            return (x for x in self.__dict__.values())

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]


