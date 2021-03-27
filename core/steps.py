import numpy as np
import torch
import pytorch_lightning as pl
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform
from .models import ExtendedSimCLR


def Distribute_Dataset(to, dataset, name, train=True, test=True):
    def Distribute_Dataset2(message=None):
        for id, model in enumerate(to):
            model.datasets[name] = dataset.train_dataloader(id)
            if test:
                model.datasets["test"] = dataset.test_dataloader()
        return message
    return Distribute_Dataset2

def Distribute_Model(source, destination):
    def Distribute_Model2(message=None):
        for id, model in enumerate(destination):
            model.W = source.W
        return message
    return Distribute_Model2

def Train(on, epochs, mode, logger=None):
    def Train2(message=None):
        for model in on:
            pl.Trainer(max_epochs=epochs, logger=logger, weights_summary=None).fit(model.mode(mode))
        return message
    return Train2

def Pretrain(on, epochs, dataset, logger=None):

    dataset_train = dataset(transforms=SimCLRTrainDataTransform).public_dataloader()

    simclr_model = ExtendedSimCLR(num_samples=10, batch_size=64, model=on[0].mode("pretrain"))

    def Pretrain2(message=None):
        for model in on:
            pl.Trainer(max_epochs=epochs, logger=logger, weights_summary=None).fit(simclr_model, train_dataloader=dataset_train)
        return message
    return Pretrain2

def Test(on, logger=None):
    def Test2(message=None):
        for model in on:
            pl.Trainer(logger=logger).test(model)
        return message
    return Test2

def Aggregate(source, target, type='model', method='Average'):
    def Aggregate2(message=None):
        if type == 'model':
            for target_model in target:
                target_model.W = sum(source_model.W for source_model in source)/len(source)
        return message
    return Aggregate2

def Compute_Outlier_Score(on):
    def Compute_Outlier_Score2(message=None):
        for model in on:

            ### Train Scoring Model
            local_features = torch.cat([model.extract_features(x.cuda()).detach() for x, _ in model.datasets["local"]])
            public_features = torch.cat([model.extract_features(x_pub.cuda()).detach() for x_pub, _ in model.datasets["distill"]])

            X = torch.cat([local_features, public_features]).cpu().numpy()
            y = np.concatenate([np.zeros(local_features.shape[0]), np.ones(public_features.shape[0])])

            norm = np.linalg.norm(X, axis=1).max()
            X_normalized = X / norm

            clf = LogisticRegression(penalty="l2", C=1 / lambda_reg, max_iter=1000).fit(X_normalized, y)

            ### Add Differential Privacy
            n = X.shape[0]
            sensitivity = 2 / (n * lambda_reg)

            if eps_delt is not None:
                epsilon, delta = eps_delt

                sig2 = 2 * np.log(1.25 / delta) * sensitivity ** 2 / epsilon ** 2

                # print(lambda_reg, n, sensitivity, sig2, np.linalg.norm(X_normalized, axis=1).max())

                clf.coef_ = clf.coef_ + np.sqrt(sig2) * np.random.normal(size=clf.coef_.shape)
                clf.intercept_ = clf.intercept_ + np.sqrt(sig2) * np.random.normal(size=clf.intercept_.shape)

            ### Compute Scores
            scores = []
            idcs = []
            for (x, _), idx in loader_distill:
                x = self.model.extract_features(x.cuda()).detach() / norm
                scores += [torch.Tensor(clf.predict_proba(x.cpu().numpy())[:, 0])]
                idcs += [idx]

            argidx = torch.argsort(torch.cat(idcs, dim=0))
            scores = torch.cat(scores, dim=0)[argidx].detach() + 1e-8

        return message
    return Compute_Outlier_Score2





