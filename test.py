import argparse

import torch
import torch.backends.cudnn as cudnn

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor, setup_logger

import utils
import models
import datasets


def main(local_rank, args):
    device = idist.device()

    if not args.pretrained_dataset == args.dataset:
        dataset = [args.pretrained_dataset, args.dataset]
    else:
        dataset = args.dataset
    dataset = datasets.get_dataset(dataset, args.datadir)
    loader  = datasets.get_loader(args, dataset)

    model = models.get_model(args, input_shape=dataset['input_shape'])
    model = idist.auto_model(model, sync_bn=True)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model_state = ckpt['model']

    if args.backbone in ['resnet18', 'resnet50']:
        model_state = { k[len('module.'):]: v for k, v in model_state.items() }
    model.load_state_dict(model_state)

    logger = setup_logger(name='logging')

    if args.dataset not in datasets.FEWSHOT_BENCHMARKS:
        acc = utils.evaluate_nn(model, loader['val'], loader['test'])
        logger.info(f'[NN Acc {acc:.4f}]')

    else:
        if args.eval_fewshot_metric == 'ft-supcon':
            val  = 0, 0 # utils.evaluate_fewshot_ft_supcon(model, loader['val'])
            test = utils.evaluate_fewshot_ft_supcon(model, loader['test'], args.ft_supcon_test_iter)
        elif args.eval_fewshot_metric == 'linear-eval':
            val  = 0, 0 # utils.evaluate_fewshot_linear(model, loader['val'])
            test = utils.evaluate_fewshot_linear(model, loader['test'])
        else:
            val  = 0, 0 # utils.evaluate_fewshot(model, loader['val'],  args.eval_fewshot_metric)
            test = utils.evaluate_fewshot(model, loader['test'], args.eval_fewshot_metric)

        logger.info(f'[Model: {args.model}] [dataset: {args.dataset}]'
                    f'[{args.N} way {args.K} shot] [FewShot {args.eval_fewshot_metric}]'
                    f'[{val[0]:.4f}±{val[1]:.4f}] | {test[0]:.4f}±{test[1]:.4f}]')


if __name__ == "__main__":
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pretrained-dataset', type=str, default='miniimagenet')
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--datadir', type=str, default='/data/miniimagenet')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='psco')
    parser.add_argument('--backbone', type=str, default='conv5')

    parser.add_argument('--prediction', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--queue-size', type=int, default=16384)
    parser.add_argument('--num-shots', type=int, default=4)
    parser.add_argument('--shot-sampling', type=str, default='topk', choices=['topk', 'prob'])
    parser.add_argument('--temperature2', type=float, default=1.)
    parser.add_argument('--sinkhorn-iter', type=int, default=3)
    parser.add_argument('--n-prototypes', type=int, default=2048)

    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Q', type=int, default=15)
    parser.add_argument('--num-tasks', type=int, default=2000)
    parser.add_argument('--ft-supcon-test-iter', type=int, default=50)

    parser.add_argument('--eval-fewshot-metric', type=str, default='supcon')

    args = parser.parse_args()

    with idist.Parallel() as parallel:
        parallel.run(main, args)

