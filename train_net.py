from lib.config import cfg, args
from lib.datasets.make_dataset import make_render_data_loader
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network, load_pretrain
from lib.evaluators import make_evaluator
import torch.multiprocessing
import torch
import torch.distributed as dist
import os
torch.autograd.set_detect_anomaly(True)

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_wandb(project_name='sdf', exp_name=''):
    import wandb
    wandb.init(project=project_name, name=exp_name, sync_tensorboard=True)


def train(cfg, network):
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    begin_epoch = load_model(
        network,
        optimizer,
        scheduler,
        recorder,
        cfg.trained_model_dir,
        resume=cfg.resume
    
    )
    if begin_epoch == 0 and cfg.pretrain != '':
        load_pretrain(network, cfg.pretrain, strict=False)
        
    set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=cfg.distributed,
        max_iter=cfg.ep_iter
    )
    test_loader = make_data_loader(cfg, is_train=False)
    render_loader = make_render_data_loader(cfg, is_train=False)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        if render_loader is not None and epoch % cfg.render_interval == 0 and (epoch > 0 or cfg.pretrain != ''):
            save_dir = os.path.join(render_loader.dataset.instance_dir, cfg.exp_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            return_sep_plane = cfg.train.render_sep_plane
            plane_masks, non_plane_masks, plane_masks_sp, non_plane_masks_sp = trainer.render(epoch, render_loader, save_dir, return_sep_plane)
            train_loader.dataset.update_rendered_plane(plane_masks, non_plane_masks)
            train_loader.dataset.update_rendered_plane_sp(plane_masks_sp, non_plane_masks_sp)

        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder, cfg.trained_model_dir, epoch)

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(
                network,
                optimizer,
                scheduler,
                recorder,
                cfg.trained_model_dir,
                epoch,
                last=True
            )

        if (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            trainer.val(epoch, test_loader)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    test_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)

    epoch = load_network(
        network,
        cfg.trained_model_dir,
        resume=cfg.resume,
        epoch=cfg.test.epoch
    )
    trainer.val(epoch, test_loader, evaluate_mesh=True, evaluator=evaluator)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        if args.wandb:
            set_wandb(project_name='sdf', exp_name=cfg.exp_name)
        train(cfg, network)


if __name__ == "__main__":
    main()