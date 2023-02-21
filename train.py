import os
import argparse

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor

import utils
import models
import datasets


def main(local_rank, args):
    device = idist.device()
    logger, tb_logger = utils.get_logger(args)
    dataset = datasets.get_dataset(args.dataset, args.datadir, augmentations=args.aug)
    loader  = datasets.get_loader(args, dataset)

    model = models.get_model(args, input_shape=dataset['input_shape'])
    model = idist.auto_model(model, sync_bn=True)

    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr, momentum=0.9, weight_decay=args.wd)
    optimizer = idist.auto_optim(optimizer)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs*len(loader['train']))

    def training_step(engine, batch):
        model.train()
        batch = convert_tensor(batch, device=device, non_blocking=True)
        outputs = model(batch)
        optimizer.zero_grad()
        outputs['loss'].backward()
        optimizer.step()
        scheduler.step()
        return outputs

    trainer = Engine(training_step)
    if logger is not None:
        trainer.logger = logger
        trainer.tb_logger = tb_logger
    trainer.add_event_handler(Events.ITERATION_COMPLETED, utils.log)

    if args.dataset not in datasets.FEWSHOT_BENCHMARKS:
        @trainer.on(Events.EPOCH_COMPLETED(every=args.eval_freq))
        def evaluation_step(engine):
            acc = utils.evaluate_nn(model, loader['val'], loader['test'])
            if idist.get_rank() == 0:
                epoch = engine.state.epoch
                engine.logger.info(f'[Epoch {epoch:4d}] [NN Acc {acc:.4f}]')
                engine.tb_logger.add_scalar('nn', acc, epoch)
            idist.barrier()

    else:
        @idist.one_rank_only()
        @trainer.on(Events.EPOCH_COMPLETED(every=args.eval_freq))
        def evaluation_step(engine):
            metric = args.eval_fewshot_metric
            val  = utils.evaluate_fewshot(model, loader['val'],  metric)
            test = utils.evaluate_fewshot(model, loader['test'], metric)

            if idist.get_rank() == 0:
                epoch = engine.state.epoch
                engine.logger.info(f'[Epoch {epoch:4d}] '
                                   f'[FewShot {metric} {val[0]:.4f}±{val[1]:.4f}] | {test[0]:.4f}±{test[1]:.4f}]')
                engine.tb_logger.add_scalar(f'fewshot_{metric}/val',  val[0],  epoch)
                engine.tb_logger.add_scalar(f'fewshot_{metric}/test', test[0], epoch)
            idist.barrier()

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=args.save_freq), utils.save_checkpoint, args,
                              model=model, optimizer=optimizer, scheduler=scheduler)

    trainer.run(loader['train'], max_epochs=args.num_epochs)
    if tb_logger is not None:
        tb_logger.close()


if __name__ == "__main__":
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--datadir', type=str, default='/data/miniimagenet')
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--base-lr', type=float, default=0.03)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--aug', type=str, default=['strong', 'weak'], nargs='+')

    parser.add_argument('--save-freq', type=int, default=10)
    parser.add_argument('--eval-freq', type=int, default=10)
    parser.add_argument('--eval-fewshot-metric', type=str, default='supcon')

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

    # for evaluation
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Q', type=int, default=15)
    parser.add_argument('--num-tasks', type=int, default=600)

    # for multiprocessing
    parser.add_argument('--master-port', type=int, default=2222)

    args = parser.parse_args()
    args.lr = args.base_lr * args.batch_size / 256

    n = torch.cuda.device_count()
    if n == 1:
        with idist.Parallel() as parallel:
            parallel.run(main, args)
    else:
        with idist.Parallel(backend='nccl', nproc_per_node=n, master_port=os.environ.get('MASTER_PORT', args.master_port)) as parallel:
            parallel.run(main, args)

