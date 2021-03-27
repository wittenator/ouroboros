import * from ouroboros.models
import MNIST, EMNIST from ouroboros.datasets
import * from ouroboros.steps
import settings from settings

server =
clients = [settings.client]*settings.clients

#Fedaux
fedaux = Process([
    Distribute_dataset(to=clients, dataset=cifar, split='dirichlet', split_kwargs{'alpha':0,01}),
    Distribute_dataset(to=clients + server, dataset=cifar100, name='distill'),
    Pretraining(on=server, dataset='distill'),
    Distribute_model(from=server, to=clients),
    compute_outlier_scores(on=clients, dataset='distill'),
    Loop([
            Select_Subset(on=clients, p=0.1, message="subset"),
            Local_training(on=clients, filter="subset"),
            aggregate(from=clients, to=server, filter="subset", type='softlabel', method='Fedaux'),
            aggregate(from=clients, to=server, filter="subset", type='model', method='Average'),
            Distillation(on=server),
            Distribute_model(from=server, to=clients)
        ],
        iterations=5
    )
])

# fedavg
fedavg = Process([
    Distribute_dataset(to=clients, dataset=cifar, split='dirichlet', split_kwargs{'alpha':0,01}),
    Distribute_model(from=server, to=clients),
    Loop([
            Select_Subset(on=clients, p=0.1, message="subset"),
            Local_training(on="subset", dataset="local"),
            aggregate(from="subset", to=server, type='model', method='Average'),
            Distribute_model(from=server, to=clients)
        ],
        iterations=5
    )
])

Experiment(**config).run()