import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import ignite.distributed as idist


def get_mlp(*layers, last_bn=False):
    modules = []
    for i in range(len(layers)-1):
        modules.append(nn.Linear(layers[i], layers[i+1], bias=False))
        if i < len(layers) - 2:
            modules.append(nn.BatchNorm1d(layers[i+1]))
            modules.append(nn.ReLU(inplace=True))
        elif last_bn:
            modules.append(nn.BatchNorm1d(layers[i+1], affine=False))
    return nn.Sequential(*modules)


def get_backbone(backbone, input_shape):
    if backbone in ['resnet18', 'resnet50']:
        net = torchvision.models.__dict__[backbone](zero_init_residual=True)
        net.out_dim = net.fc.weight.shape[1]
        if input_shape[1:] == (32, 32):
            net.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Identity()

    elif backbone == 'conv4':
        layers = []
        in_channels = input_shape[0]
        for _ in range(4):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, ceil_mode=(input_shape==(1, 28, 28)))))
            in_channels = 64
        layers.append(nn.Flatten())
        net = nn.Sequential(*layers)
        if input_shape[1:] == (84, 84):
            net.out_dim = 1600
        elif input_shape[1:] == (28, 28):
            net.out_dim = 256

    elif backbone == 'conv5':
        layers = []
        in_channels = input_shape[0]
        for _ in range(5):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)))
            in_channels = 64
        layers.append(nn.Flatten())
        net = nn.Sequential(*layers)
        net.out_dim = 256

    else:
        raise NotImplementedError

    return net


def get_model(args, input_shape):
    if args.model == 'simclr':
        model = SimCLR(backbone=args.backbone,
                       input_shape=input_shape,
                       temperature=args.temperature)

    elif args.model == 'swav':
        model = SwAV(backbone=args.backbone,
                     input_shape=input_shape,
                     temperature=args.temperature,
                     n_prototypes=args.n_prototypes,
                     sinkhorn_iter=args.sinkhorn_iter,
                     )
    elif args.model == 'moco':
        model = MoCo(backbone=args.backbone,
                     input_shape=input_shape,
                     momentum=args.momentum,
                     temperature=args.temperature,
                     queue_size=args.queue_size,
                     prediction=args.prediction,
                     )

    elif args.model == 'byol':
        model = BYOL(backbone=args.backbone,
                     input_shape=input_shape,
                     momentum=args.momentum,
                     temperature=args.temperature,
                     queue_size=args.queue_size,
                     prediction=args.prediction,
                     )

    elif args.model == 'psco':
        model = PsCoMoCo(backbone=args.backbone,
                         input_shape=input_shape,
                         momentum=args.momentum,
                         temperature=args.temperature,
                         queue_size=args.queue_size,
                         prediction=args.prediction,
                         num_shots=args.num_shots,
                         shot_sampling=args.shot_sampling,
                         temperature2=args.temperature2,
                         sinkhorn_iter=args.sinkhorn_iter,
                         )

    elif args.model == 'psco_byol':
        model = PsCoBYOL(backbone=args.backbone,
                         input_shape=input_shape,
                         momentum=args.momentum,
                         temperature=args.temperature,
                         queue_size=args.queue_size,
                         prediction=args.prediction,
                         num_shots=args.num_shots,
                         shot_sampling=args.shot_sampling,
                         temperature2=args.temperature2,
                         sinkhorn_iter=args.sinkhorn_iter,
                         )

    elif args.model == 'sup':
        model = Supervised(backbone=args.backbone)

    return model


class Supervised(nn.Module):
    def __init__(self,
                 backbone: str = 'resnet50',
                 ):
        super().__init__()
        self.backbone = torchvision.models.__dict__[backbone](pretrained=True)
        self.backbone.out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()


class BaseModel(nn.Module):
    def __init__(self,
                 backbone: str = 'resnet18',
                 input_shape: tuple[int] = (3, 32, 32),
                 momentum: float = 0.999,
                 prediction: bool = False,
                 ):
        super().__init__()
        self.ema_modules = []
        self.build_ema_module('backbone', get_backbone, backbone=backbone, input_shape=input_shape)
        self.build_ema_module('projector', get_mlp, self.backbone.out_dim, 4096, 256, last_bn=True)
        self.predictor = get_mlp(256, 4096, 256, last_bn=False) if prediction else nn.Identity()
        self.momentum = momentum

    def build_ema_module(self, name, fn, *args, **kwargs):
        module     = fn(*args, **kwargs)
        ema_module = fn(*args, **kwargs)
        ema_module.load_state_dict(module.state_dict())
        ema_module.requires_grad_(False)
        setattr(self, name, module)
        setattr(self, f'ema_{name}', ema_module)
        self.ema_modules.append(name)

    @torch.no_grad()
    def update_ema_modules(self):
        for name in self.ema_modules:
            module_dst = getattr(self, f'ema_{name}')
            module_src = getattr(self, name)
            params_dst = dict(module_dst.named_parameters())
            params_src = dict(module_src.named_parameters())
            buf_dst = dict(module_dst.named_buffers())
            buf_src = dict(module_src.named_buffers())

            for k in params_dst.keys():
                params_dst[k].data.mul_(self.momentum).add_(params_src[k].data, alpha=1-self.momentum)
            for k in buf_dst.keys():
                buf_dst[k].data.copy_(buf_src[k].data)

    def forward(self, batch, mode='train', **kwargs):
        if mode == 'train':
            self.update_ema_modules()
            return self.compute_loss(batch, **kwargs)
        elif mode == 'feature':
            return self.extract_features(batch, **kwargs)
        elif mode == 'fewshot':
            return self.predict_fewshot(batch, **kwargs)

    def compute_loss(self, batch, **kwargs):
        raise NotImplementedError

    def extract_features(self, batch, momentum=False, projection=False, prediction=False):
        if not momentum:
            z = self.backbone(batch)
            if projection:
                z = self.projector(z)
                if prediction:
                    z = self.predictor(z)
        else:
            z = self.ema_backbone(batch)
            if projection:
                z = self.ema_projector(z)
        return z

    def predict_fewshot(self, batch, **kwargs):
        raise NotImplementedError


class SimCLR(BaseModel):
    def __init__(self,
                 backbone: str = 'resnet18',
                 input_shape: tuple[int] = (3, 32, 32),
                 temperature: float = 0.2,
                 ):
        super().__init__(backbone=backbone, input_shape=input_shape)
        self.temperature = temperature

    def compute_loss(self, batch):
        (x1, x2), _ = batch
        n = x1.shape[0]
        z1 = F.normalize(self.projector(self.backbone(x1)))
        z2 = F.normalize(self.projector(self.backbone(x2)))
        z = torch.cat([z1, z2])
        logits = torch.mm(z, z.T).div(self.temperature)
        logits.fill_diagonal_(float('-inf'))
        labels = torch.tensor(list(range(n, 2*n)) + list(range(n)), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return dict(loss=loss)


class SwAV(BaseModel):
    def __init__(self,
                 backbone: str = 'resnet18',
                 input_shape: tuple[str] = (3, 32, 32),
                 temperature: float = 0.2,
                 n_prototypes: int = 2048,
                 sinkhorn_iter: int = 3,
                 ):
        super().__init__(backbone=backbone, input_shape=input_shape)
        self.prototypes = nn.Linear(128, n_prototypes, bias=False)
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter

    def compute_loss(self, batch):
        (x1, x2), _ = batch
        with torch.no_grad():
            w = F.normalize(self.prototypes.weight.data.clone())
            self.prototypes.weight.copy_(w)
        z1 = F.normalize(self.projector(self.backbone(x1)))
        z2 = F.normalize(self.projector(self.backbone(x2)))
        scores1 = self.prototypes(z1)
        scores2 = self.prototypes(z2)
        z1 = z1.detach()
        z2 = z2.detach()
        q1 = distributed_sinkhorn(scores1.detach(), normalization='row', num_iterations=self.sinkhorn_iter)
        q2 = distributed_sinkhorn(scores2.detach(), normalization='row', num_iterations=self.sinkhorn_iter)
        logp1 = F.log_softmax(scores1.div(self.temperature), dim=1)
        logp2 = F.log_softmax(scores2.div(self.temperature), dim=1)
        loss = -(torch.sum(q1.mul(logp2), dim=1)+torch.sum(q2.mul(logp1), dim=1)).mean()
        return dict(loss=loss)


class MoCo(BaseModel):
    def __init__(self,
                 backbone: str = 'resnet18',
                 input_shape: tuple[int] = (3, 32, 32),
                 momentum: float = 0.999,
                 queue_size: int = 8192,
                 temperature: float = 0.2,
                 prediction: bool = False,
                 ):
        super().__init__(backbone=backbone, input_shape=input_shape, momentum=momentum, prediction=prediction)
        self.register_buffer('queue', F.normalize(torch.randn(queue_size, 256)))
        self.register_buffer('queue_ptr', torch.tensor([0]))
        self.temperature = temperature

    @torch.no_grad()
    def update_queue(self, z):
        z = idist.all_gather(z)
        batch_size = z.shape[0]
        ptr = self.queue_ptr.item()
        self.queue[ptr:ptr+batch_size] = z
        self.queue_ptr[0] = (ptr + batch_size) % self.queue.shape[0]

    def compute_loss(self, batch):
        (x1, x2), _ = batch
        z1 = F.normalize(self.predictor(self.projector(self.backbone(x1))))
        with torch.no_grad():
            z2 = F.normalize(self.ema_projector(self.ema_backbone(x2))).detach()
        l_pos = torch.mul(z1, z2).sum(dim=1, keepdim=True)
        l_neg = torch.mm(z1, self.queue.clone().T.detach())
        logits = torch.cat([l_pos, l_neg], dim=1).div(self.temperature)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        self.update_queue(z2)
        return dict(loss=loss)


class BYOL(BaseModel):
    def __init__(self,
                 backbone: str = 'resnet18',
                 input_shape: tuple[int] = (3, 32, 32),
                 momentum: float = 0.999,
                 queue_size: int = 8192,
                 temperature: float = 0.2,
                 prediction: bool = False,
                 ):
        super().__init__(backbone=backbone, input_shape=input_shape, momentum=momentum, prediction=prediction)
        self.register_buffer('queue', F.normalize(torch.randn(queue_size, 256)))
        self.register_buffer('queue_ptr', torch.tensor([0]))
        self.temperature = temperature

    @torch.no_grad()
    def update_queue(self, z):
        z = idist.all_gather(z)
        batch_size = z.shape[0]
        ptr = self.queue_ptr.item()
        self.queue[ptr:ptr+batch_size] = z
        self.queue_ptr[0] = (ptr + batch_size) % self.queue.shape[0]

    def compute_loss(self, batch):
        (x1, x2), _ = batch
        z1 = F.normalize(self.predictor(self.projector(self.backbone(x1))))
        with torch.no_grad():
            z2 = F.normalize(self.ema_projector(self.ema_backbone(x2))).detach()
        loss = - F.cosine_similarity(z1, z2).mean() * 4
        self.update_queue(z2)
        return dict(loss=loss)


class PsCoMoCo(MoCo):
    def __init__(self,
                 backbone: str = 'resnet18',
                 input_shape: tuple[int] = (3, 32, 32),
                 momentum: float = 0.999,
                 queue_size: int = 8192,
                 temperature: float = 0.2,
                 prediction: bool = False,
                 num_shots: int = 16,
                 shot_sampling: str = 'topk',
                 temperature2: float = 1.,
                 sinkhorn_iter: int = 3,
                 ):
        super().__init__(backbone=backbone,
                         input_shape=input_shape,
                         momentum=momentum,
                         queue_size=queue_size,
                         temperature=temperature,
                         prediction=prediction)
        self.num_shots= num_shots
        self.shot_sampling = shot_sampling
        self.temperature2 = temperature2
        self.sinkhorn_iter = sinkhorn_iter

    def compute_loss(self, batch):
        (x1, x2), _ = batch
        p1 = F.normalize(self.predictor(self.projector(self.backbone(x1))))
        with torch.no_grad():
            z2 = F.normalize(self.ema_projector(self.ema_backbone(x2))).detach()
            sim = torch.mm(z2, self.queue.T).detach()
            assignments = distributed_sinkhorn(sim, normalization='row', num_iterations=self.sinkhorn_iter).detach()

            if self.shot_sampling == 'topk':
                samples = torch.topk(assignments, k=self.num_shots, dim=1).indices.reshape(-1)
            elif self.shot_sampling == 'prob':
                samples = torch.multinomial(assignments, num_samples=self.num_shots).reshape(-1)

            shots = self.queue[samples].clone().detach()
            prototypes = shots.reshape(assignments.shape[0], self.num_shots, -1).mean(dim=1)

        # SupCon with N-way K-shot task
        loss = (torch.mm(p1, shots.T).div(self.temperature2).logsumexp(dim=1) - p1.mul(prototypes).div(self.temperature2).sum(dim=1)).mean()

        # MoCo
        logits = torch.cat([
                torch.mul(p1, z2).sum(dim=1, keepdim=True),
                torch.mm(p1, self.queue.clone().detach().T),
        ], dim=1).div(self.temperature)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss += F.cross_entropy(logits, labels)

        self.update_queue(z2)
        return dict(loss=loss)


class PsCoBYOL(MoCo):
    def __init__(self,
                 backbone: str = 'resnet18',
                 input_shape: tuple[int] = (3, 32, 32),
                 momentum: float = 0.999,
                 queue_size: int = 8192,
                 temperature: float = 0.2,
                 prediction: bool = False,
                 num_shots: int = 16,
                 shot_sampling: str = 'topk',
                 temperature2: float = 1.,
                 sinkhorn_iter: int = 3,
                 ):
        super().__init__(backbone=backbone,
                         input_shape=input_shape,
                         momentum=momentum,
                         queue_size=queue_size,
                         temperature=temperature,
                         prediction=prediction)
        self.num_shots= num_shots
        self.shot_sampling = shot_sampling
        self.temperature2 = temperature2
        self.sinkhorn_iter = sinkhorn_iter

    def compute_loss(self, batch):
        (x1, x2), _ = batch
        p1 = F.normalize(self.predictor(self.projector(self.backbone(x1))))
        with torch.no_grad():
            z2 = F.normalize(self.ema_projector(self.ema_backbone(x2))).detach()
            sim = torch.mm(z2, self.queue.T).detach()
            assignments = distributed_sinkhorn(sim, normalization='row', num_iterations=self.sinkhorn_iter).detach()

            if self.shot_sampling == 'topk':
                samples = torch.topk(assignments, k=self.num_shots, dim=1).indices.reshape(-1)
            elif self.shot_sampling == 'prob':
                samples = torch.multinomial(assignments, num_samples=self.num_shots).reshape(-1)

            shots = self.queue[samples].clone().detach()
            prototypes = shots.reshape(assignments.shape[0], self.num_shots, -1).mean(dim=1)

        # SupCon with N-way K-shot task
        loss = (torch.mm(p1, shots.T).div(self.temperature2).logsumexp(dim=1) - p1.mul(prototypes).div(self.temperature2).sum(dim=1)).mean()

        # BYOL
        loss += - F.cosine_similarity(p1, z2).mean() * 4

        self.update_queue(z2)
        return dict(loss=loss)


@torch.no_grad()
def distributed_sinkhorn(out, epsilon=0.05, num_iterations=3, normalization='col'):
    # https://github.com/facebookresearch/swav/blob/main/main_swav.py

    Q = torch.exp(out / epsilon) # Q is B-by-K (B = batch size, K = queue size)
    B = Q.shape[0] * idist.get_world_size()
    K = Q.shape[1]

    # make the matrix sums to 1
    Q /= idist.all_reduce(torch.sum(Q))

    if normalization == 'col':
        for it in range(num_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

            # normalize each column: total weight per sample must be 1/B
            Q /= idist.all_reduce(torch.sum(Q, dim=0, keepdim=True))
            Q /= K

        Q *= K # the colomns must sum to 1 so that Q is an assignment
    else:
        for it in range(num_iterations):
            # normalize each column: total weight per sample must be 1/B
            Q /= idist.all_reduce(torch.sum(Q, dim=0, keepdim=True))
            Q /= K

            # normalize each row: total weight per prototype must be 1/K
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q

