import os
import math
import copy

from sklearn.neighbors import KNeighborsClassifier

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ignite.utils import setup_logger, convert_tensor
import ignite.distributed as idist

import models


def get_logger(args):
    if idist.get_rank() == 0:
        os.makedirs(args.logdir)
        logger = setup_logger(name='logging', filepath=os.path.join(args.logdir, 'log.txt'))
        logger.info(args)
        logger.info(' '.join(os.sys.argv))
        tb_logger = SummaryWriter(log_dir=args.logdir)
    else:
        logger, tb_logger = None, None

    idist.barrier()
    return logger, tb_logger


@idist.one_rank_only()
def log(engine):
    if engine.state.iteration % 10 == 0:
        engine.logger.info(f'[Epoch {engine.state.epoch:4d}] '
                           f'[Iter {engine.state.iteration:6d}] '
                           f'[Loss {engine.state.output["loss"].item():.4f}]')
        for k, v in engine.state.output.items():
            engine.tb_logger.add_scalar(k, v, engine.state.iteration)


@idist.one_rank_only()
def save_checkpoint(engine, args, **kwargs):
    state = { k: v.state_dict() for k, v in kwargs.items() }
    state['engine'] = engine.state_dict()
    # torch.save(state, os.path.join(args.logdir, f'ckpt-{engine.state.epoch}.pth'))
    torch.save(state, os.path.join(args.logdir, f'last.pth'))


@torch.no_grad()
def collect_features(model, loader):
    model.eval()
    device = idist.device()
    X, Y = [], []
    for batch in loader:
        x, y = convert_tensor(batch, device=device)
        x = model(x, mode='feature')
        X.append(x.detach())
        Y.append(y.detach())
    X = torch.cat(X).detach()
    Y = torch.cat(Y).detach()
    return X, Y


@torch.no_grad()
def evaluate_nn(model, trainloader, testloader):
    X_train, Y_train = [idist.all_gather(_) for _ in collect_features(model, trainloader)]
    X_test,  Y_test  = collect_features(model, testloader)

    BATCH_SIZE = 256
    corrects = []
    d_train = X_train.T.pow(2).sum(dim=0, keepdim=True)
    for i in range(0, X_test.shape[0], BATCH_SIZE):
        X_batch = X_test[i:i+BATCH_SIZE]
        Y_batch = Y_test[i:i+BATCH_SIZE]
        d_batch = X_batch.pow(2).sum(dim=1)[:, None]
        distance = d_batch - torch.mm(X_batch, X_train.T) * 2 + d_train
        corrects.append((Y_batch == Y_train[distance.argmin(dim=1)]).detach())
    corrects = idist.all_gather(torch.cat(corrects))
    acc = corrects.float().mean()
    return acc


@torch.no_grad()
def evaluate_fewshot(model, loader, metric):
    model.eval()
    device = idist.device()

    N = loader.batch_sampler.N
    K = loader.batch_sampler.K
    Q = loader.batch_sampler.Q
    accuracies = []
    for cnt, task in enumerate(loader):
        x, _ = convert_tensor(task, device=device)
        input_shape = x.shape[1:]
        x = x.view(N, K+Q, *input_shape)

        shots   = x[:, :K].reshape(N*K, *input_shape)
        queries = x[:, K:].reshape(N*Q, *input_shape)
        if metric == 'knn':
            knn = KNeighborsClassifier(n_neighbors=K, metric='cosine')
            shots_knn   = F.normalize(model(shots,   mode='feature')).detach().cpu().numpy()
            queries_knn = F.normalize(model(queries, mode='feature')).detach().cpu().numpy()

            y_shots = np.tile(np.expand_dims(np.arange(N), 1), K).reshape(-1)
            knn.fit(shots_knn, y_shots)

            preds = np.array(knn.predict(queries_knn))
            labels = np.tile(np.expand_dims(np.arange(N), 1), Q).reshape(-1)
            accuracies.append((preds == labels).mean().item())

        elif metric == 'supcon':
            queries_supcon = F.normalize(model(queries, mode='feature', momentum=False, projection=True, prediction=True))
            shots_supcon   = F.normalize(model(shots,   mode='feature', momentum=False, projection=True))

            prototypes = F.normalize(shots_supcon.view(N, K, shots_supcon.shape[-1]).mean(dim=1))
            preds = torch.mm(queries_supcon, prototypes.T).argmax(dim=1)
            labels = torch.arange(N, device=preds.device).repeat_interleave(Q)
            accuracies.append((preds == labels).float().mean().item())

        else:
            raise NotImplementedError

        print(f'{cnt:4d} {sum(accuracies)/len(accuracies):.4f}', end='\r')

    accuracies = idist.all_gather(torch.tensor(accuracies))
    return accuracies.mean(), accuracies.std()*1.96/math.sqrt(accuracies.numel())


def get_projector(dim_in):
    mlp = []
    for l in range(2):
        dim1 = dim_in if l == 0 else 4096
        dim2 = 256 if l == 1 else 4096

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        else:
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)


def get_predictor():
    mlp = []
    for l in range(2):
        dim1 = 256 if l == 0 else 4096
        dim2 = 256 if l == 1 else 4096

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))

    return nn.Sequential(*mlp)


def evaluate_fewshot_ft_supcon(model, loader, n_iters=50):
    device = idist.device()

    N = loader.batch_sampler.N
    K = loader.batch_sampler.K
    Q = loader.batch_sampler.Q

    accuracies = []
    for cnt, task in enumerate(loader):
        x, _ = convert_tensor(task, device=device)
        input_shape = x.shape[1:]
        x = x.view(N, K+Q, *input_shape)

        shots   = x[:, :K].reshape(N*K, *input_shape)
        queries = x[:, K:].reshape(N*Q, *input_shape)

        A = torch.zeros(N*K, N*K, device=device)
        for i in range(N):
            A[i*K:(i+1)*K, i*K:(i+1)*K] = 1.

        net = copy.deepcopy(model)
        net.eval()
        net.predictor.train()
        net.projector.train()

        optimizer = optim.SGD(list(net.projector.parameters()) + list(net.predictor.parameters()), lr=0.01, momentum=0.9, weight_decay=0.001)

        with torch.no_grad():
            shots_train = net.backbone(shots).detach()

        for _ in range(n_iters):
            with torch.no_grad():
                shots_train = shots_train.detach()

            z = net.projector(shots_train)
            p = net.predictor(z)
            z = F.normalize(z)
            p = F.normalize(p)
            logits = torch.mm(p, z.T).div(net.temperature2)
            loss = (logits.logsumexp(dim=1) - logits.mul(A.clone().detach()).sum(dim=1).div(K)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            queries = F.normalize(net(queries, mode='feature', momentum=False, projection=True, prediction=True))
            shots   = F.normalize(net(shots,   mode='feature', momentum=False, projection=True))

            prototypes = F.normalize(shots.view(N, K, shots.shape[-1]).mean(dim=1))
            preds = torch.mm(queries, prototypes.T).argmax(dim=1)
            labels  = torch.arange(N, device=shots.device).repeat_interleave(Q)
            acc = (preds == labels).float().mean().item()
        accuracies.append(acc)
        print(f'{cnt:4d} {sum(accuracies)/len(accuracies):.4f}', end='\r')

    accuracies = idist.all_gather(torch.tensor(accuracies))
    return accuracies.mean(), accuracies.std()*1.96/math.sqrt(accuracies.numel())


def evaluate_fewshot_linear(model, loader):
    device = idist.device()

    N = loader.batch_sampler.N
    K = loader.batch_sampler.K
    Q = loader.batch_sampler.Q

    accuracies = []
    for cnt, task in enumerate(loader):
        x, _ = convert_tensor(task, device=device)
        input_shape = x.shape[1:]
        x = x.view(N, K+Q, *input_shape)

        shots   = x[:, :K].reshape(N*K, *input_shape)
        queries = x[:, K:].reshape(N*Q, *input_shape)
        y_shots = torch.arange(N, device=shots.device).repeat_interleave(K)
        labels  = torch.arange(N, device=shots.device).repeat_interleave(Q)

        net        = copy.deepcopy(model.backbone)
        classifier = nn.Linear(net.out_dim, N).to(device)
        net.eval()
        classifier.train()

        optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            shots   = net(shots)
            queries = net(queries)

        for _ in range(100):
            with torch.no_grad():
                shots   = shots.detach()
                queries = queries.detach()

            rand_id = np.random.permutation(N*K)
            batch_indices = [rand_id[i*4:(i+1)*4] for i in range(rand_id.size//4)]
            for id in batch_indices:
                x_train = shots[id]
                y_train = y_shots[id]
                shots_pred = classifier(x_train)
                loss = criterion(shots_pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        net.eval()
        classifier.eval()
        with torch.no_grad():
            preds = classifier(queries).argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            accuracies.append(acc)
        print(f'{cnt:4d} {sum(accuracies)/len(accuracies):.4f}', end='\r')

    accuracies = idist.all_gather(torch.tensor(accuracies))
    return accuracies.mean(), accuracies.std()*1.96/math.sqrt(accuracies.numel())

